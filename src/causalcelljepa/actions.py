# Deterministic target-to-protein mapping and frozen ESM-2 mean pooling.
# Resource acquisition remains in scripts/action_resources.py, separate from model inference.
import re
from collections import defaultdict

import torch


def learned_target_id_payload(targets, training_targets):
    """Encode training identities one-hot and collapse every unseen target to unknown."""
    targets, training_targets = list(targets), sorted(training_targets)
    assert len(targets) == len(set(targets))
    assert len(training_targets) == len(set(training_targets))
    assert set(training_targets) <= set(targets)
    target_index = {target: index for index, target in enumerate(targets)}
    known = torch.zeros(len(targets), dtype=torch.bool)
    embedding = torch.zeros(len(targets), len(training_targets))
    for identity, target in enumerate(training_targets):
        row = target_index[target]
        known[row] = True
        embedding[row, identity] = 1.0
    return known, embedding


def protein_symbols(value):
    """Parse UniProt's whitespace/semicolon-delimited gene-name fields."""
    return frozenset(filter(None, re.split(r"[;\s]+", value.strip())))


def map_targets(target_gene_ids, proteins, id_mapping_rows):
    """Map targets conservatively, requiring name agreement before Ensembl acceptance."""
    by_gene_id = defaultdict(set)
    for row in id_mapping_rows:
        if row["Entry"] in proteins:
            by_gene_id[row["From"]].add(row["Entry"])
    primary, synonym = defaultdict(set), defaultdict(set)
    for accession, row in proteins.items():
        for symbol in protein_symbols(row["Gene Names (primary)"]):
            primary[symbol].add(accession)
        for symbol in protein_symbols(row["Gene Names (synonym)"]):
            synonym[symbol].add(accession)

    mapped, unknown = {}, {}
    for target, gene_id in sorted(target_gene_ids.items()):
        candidates = by_gene_id[gene_id]
        matching = {
            accession
            for accession in candidates
            if target
            in protein_symbols(
                proteins[accession]["Gene Names (primary)"]
                + " "
                + proteins[accession]["Gene Names (synonym)"]
            )
        }
        if len(matching) == 1:
            accession, method = next(iter(matching)), "ensembl_name_match"
        elif len(primary[target]) == 1:
            accession, method = next(iter(primary[target])), "primary_symbol_fallback"
        elif len(synonym[target]) == 1:
            accession, method = next(iter(synonym[target])), "synonym_symbol_fallback"
        else:
            unknown[target] = {
                "gene_id": gene_id,
                "candidate_accessions": sorted(matching or candidates),
                "reason": "no_unique_reviewed_canonical_protein",
            }
            continue
        mapped[target] = {"gene_id": gene_id, "accession": accession, "method": method}
    return mapped, unknown


def embed_proteins(model, alphabet, sequences, layer=6, chunk_residues=1022, token_budget=4096):
    """Mean-pool all canonical residues with bounded, deterministic ESM-2 chunks."""
    chunks = [
        (accession, start, sequence[start : start + chunk_residues])
        for accession, sequence in sorted(sequences.items())
        for start in range(0, len(sequence), chunk_residues)
    ]
    chunks.sort(key=lambda item: (len(item[2]), item[0], item[1]))
    converter, pooled = alphabet.get_batch_converter(), {}
    offset = 0
    with torch.inference_mode():
        while offset < len(chunks):
            stop = offset + 1
            while stop < len(chunks) and (len(chunks[stop][2]) + 2) * (stop - offset + 1) <= token_budget:
                stop += 1
            batch = chunks[offset:stop]
            labels = [(f"{accession}:{start}", sequence) for accession, start, sequence in batch]
            _, _, tokens = converter(labels)
            representations = model(tokens, repr_layers=[layer])["representations"][layer]
            assert torch.isfinite(representations).all()
            for index, (accession, start, sequence) in enumerate(batch):
                pooled[accession, start] = representations[index, 1 : len(sequence) + 1].sum(
                    0, dtype=torch.float64
                )
            offset = stop

    embeddings = {}
    for accession, sequence in sorted(sequences.items()):
        starts = range(0, len(sequence), chunk_residues)
        embeddings[accession] = (
            sum(
                (pooled[accession, start] for start in starts),
                torch.zeros(model.embed_dim, dtype=torch.float64),
            )
            / len(sequence)
        ).float()
    return embeddings, {"proteins": len(sequences), "chunks": len(chunks)}
