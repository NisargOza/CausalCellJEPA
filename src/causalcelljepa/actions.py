# Deterministic target-to-protein mapping and frozen ESM-2 mean pooling.
# Resource acquisition remains in scripts/action_resources.py, separate from model inference.
import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import yaml

from causalcelljepa.resources import (
    derive_hvg_programs,
    file_sha256,
    parse_go_basic,
    parse_human_bp_annotations,
)
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


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


def multiteacher_action_payload(action, programs, rank=64):
    """Fuse frozen ESM vectors with a canonically oriented low-rank GO teacher."""
    targets = list(action["targets"])
    target_index = {target: index for index, target in enumerate(targets)}
    go_ids = sorted(programs)
    membership = np.zeros((len(targets), len(go_ids)), dtype=np.float32)
    for column, go_id in enumerate(go_ids):
        for target in programs[go_id]:
            membership[target_index[target], column] = 1
    centered = membership - membership.mean(0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:rank]
    anchors = np.abs(components).argmax(1)
    components *= np.where(components[np.arange(rank), anchors] < 0, -1, 1)[:, None]
    go_embedding = centered @ components.T
    embedding = torch.cat((action["embedding"].float(), torch.from_numpy(go_embedding)), 1)
    return {
        "targets": targets,
        "embedding": embedding,
        "known": action["known"].bool(),
        "modality_dims": [action["embedding"].shape[1], rank],
        "modalities": ["esm2_t6_8M_UR50D", "go_bp_lsa"],
    }, {
        "go_terms": len(go_ids),
        "go_rank": rank,
        "targets_with_go": int((membership.sum(1) > 0).sum()),
        "target_coverage": float((membership.sum(1) > 0).mean()),
        "go_term_sha256": sha256("\n".join(go_ids).encode()).hexdigest(),
    }


def contextual_multiteacher_action_payload(action, programs, rank=64):
    """Append explicit teacher availability while preserving frozen teacher features."""
    payload, report = multiteacher_action_payload(action, programs, rank)
    go_targets = set().union(*programs.values()) if programs else set()
    availability = torch.tensor(
        [
            [bool(action["known"][index]), target in go_targets]
            for index, target in enumerate(action["targets"])
        ],
        dtype=torch.bool,
    )
    payload["embedding"] = torch.cat((payload["embedding"], availability.float()), 1)
    payload["known"] = availability.any(1)
    payload["modality_availability"] = True
    report["targets_known_from_any_modality"] = int(payload["known"].sum())
    return payload, report


def prepare_multiteacher_action(
    config_path="configs/multiteacher_action.yaml", include_availability=False
):
    """Build the checksum-pinned ESM+GO action cache without reading outcomes."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    source, go = config["source_action"], config["go_teacher"]
    assert file_sha256(source["path"]) == source["sha256"]
    action = torch.load(source["path"], map_location="cpu", weights_only=True)
    assert file_sha256(go["ontology_path"]) == go["ontology_sha256"]
    assert file_sha256(go["annotations_path"]) == go["annotations_sha256"]
    _, terms, aliases = parse_go_basic(go["ontology_path"])
    _, annotations, stats = parse_human_bp_annotations(
        go["annotations_path"], terms, aliases, excluded_evidence=go["excluded_evidence_codes"]
    )
    programs = derive_hvg_programs(
        terms,
        annotations,
        action["targets"],
        go["minimum_targets"],
        go["maximum_targets"],
    )
    builder = (
        contextual_multiteacher_action_payload
        if include_availability
        else multiteacher_action_payload
    )
    payload, report = builder(action, programs, go["rank"])
    output = Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "input_dim": payload["embedding"].shape[1],
        "modality_dims": payload["modality_dims"],
    }
    if include_availability:
        artifact["modality_availability"] = True
    manifest = {
        "format_version": 1,
        "architecture": (
            "frozen_esm_go_contextual_multiteacher_action"
            if include_availability
            else "frozen_esm_go_multiteacher_action"
        ),
        "artifact": artifact,
        "report": {**report, "annotation_stats": stats},
        "source": {
            "config_sha256": file_sha256(config_path),
            "action_sha256": source["sha256"],
            "ontology_sha256": go["ontology_sha256"],
            "annotations_sha256": go["annotations_sha256"],
            "outcomes_read": False,
        },
        "provenance": {
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    manifest["manifest_sha256"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    Path(config["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


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
