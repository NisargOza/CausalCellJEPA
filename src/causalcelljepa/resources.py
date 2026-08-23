"""Pinned Gene Ontology resource acquisition and deterministic GMT derivation."""

import gzip
import json
import os
import urllib.request
from collections import defaultdict
from hashlib import sha256
from pathlib import Path


def file_sha256(path, block_size=1 << 20):
    """Return a streaming SHA-256 digest without loading large resources in memory."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_download(url, destination, expected_bytes, expected_sha256):
    """Download a pinned file atomically, or verify an already-present copy."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = (destination.stat().st_size, file_sha256(destination))
        expected = (expected_bytes, expected_sha256)
        if actual != expected:
            raise ValueError(
                f"Resource integrity failure for {destination}: {actual} != {expected}"
            )
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            while block := response.read(1 << 20):
                output.write(block)
        actual = (partial.stat().st_size, file_sha256(partial))
        expected = (expected_bytes, expected_sha256)
        if actual != expected:
            raise ValueError(f"Resource integrity failure for {url}: {actual} != {expected}")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def _commit_obo_term(raw, terms, aliases):
    if (
        not raw
        or raw.get("namespace") != "biological_process"
        or raw.get("obsolete")
        or "gocheck_do_not_annotate" in raw.get("subsets", ())
    ):
        return
    go_id = raw["id"]
    terms[go_id] = {
        "name": raw["name"],
        "parents": tuple(sorted(set(raw.get("parents", ())))),
    }
    for alias in raw.get("alt_ids", ()):
        aliases[alias] = go_id


def parse_go_basic(path):
    """Parse non-obsolete BP terms with conservative is-a/part-of ancestry."""
    terms, aliases, header = {}, {}, {}
    current = None
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                _commit_obo_term(current, terms, aliases)
                current = {"parents": [], "alt_ids": [], "subsets": []}
            elif line.startswith("["):
                _commit_obo_term(current, terms, aliases)
                current = None
            elif current is None:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    if key in {"data-version", "format-version"}:
                        header[key] = value
            elif line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:]
            elif line == "is_obsolete: true":
                current["obsolete"] = True
            elif line.startswith("alt_id: "):
                current["alt_ids"].append(line[8:])
            elif line.startswith("subset: "):
                current["subsets"].append(line[8:])
            elif line.startswith("is_a: "):
                current["parents"].append(line[6:].split()[0])
            elif line.startswith("relationship: part_of "):
                current["parents"].append(line[22:].split()[0])
    _commit_obo_term(current, terms, aliases)
    valid = set(terms)
    for term in terms.values():
        term["parents"] = tuple(parent for parent in term["parents"] if parent in valid)
    return header, terms, aliases


def _ancestor_map(terms):
    memo, visiting = {}, set()

    def ancestors(go_id):
        if go_id in memo:
            return memo[go_id]
        if go_id in visiting:
            raise ValueError(f"Cycle in go-basic ontology at {go_id}")
        visiting.add(go_id)
        result = {go_id}
        for parent in terms[go_id]["parents"]:
            result.update(ancestors(parent))
        visiting.remove(go_id)
        memo[go_id] = frozenset(result)
        return memo[go_id]

    return {go_id: ancestors(go_id) for go_id in terms}


def parse_human_bp_annotations(
    path, terms, aliases, taxon="taxon:9606", excluded_evidence=("ND", "RCA")
):
    """Read valid human BP annotations, excluding explicitly negated assertions."""
    annotations = defaultdict(set)
    header = {}
    stats = defaultdict(int)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                if ": " in line:
                    key, value = line[1:].rstrip().split(": ", 1)
                    header[key] = value
                continue
            stats["rows"] += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 17:
                stats["malformed"] += 1
                continue
            symbol, qualifier, go_id, evidence, aspect, annotation_taxon = (
                fields[2],
                fields[3],
                fields[4],
                fields[6],
                fields[8],
                fields[12],
            )
            if "NOT" in qualifier.split("|"):
                stats["negated"] += 1
                continue
            if evidence in excluded_evidence:
                stats["excluded_evidence"] += 1
                continue
            if aspect != "P" or taxon not in annotation_taxon.split("|"):
                stats["non_bp_or_taxon"] += 1
                continue
            go_id = aliases.get(go_id, go_id)
            if go_id not in terms or not symbol:
                stats["unknown_term_or_symbol"] += 1
                continue
            annotations[go_id].add(symbol)
            stats["accepted_rows"] += 1
    return header, annotations, dict(stats)


def derive_hvg_programs(terms, annotations, hvg_genes, minimum=3, maximum=500):
    """Propagate direct annotations and retain useful programs in frozen HVG space."""
    ancestors = _ancestor_map(terms)
    propagated = defaultdict(set)
    for go_id, genes in annotations.items():
        for ancestor in ancestors[go_id]:
            propagated[ancestor].update(genes)
    hvg_genes = set(hvg_genes)
    programs = {}
    for go_id, genes in propagated.items():
        selected = tuple(sorted(genes & hvg_genes))
        if go_id != "GO:0008150" and minimum <= len(selected) <= maximum:
            programs[go_id] = selected
    return programs


def write_gmt(programs, terms, destination):
    """Write stable GO-ID-ordered, HVG-only GMT records."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for go_id in sorted(programs):
        label = f"{go_id} {terms[go_id]['name']}"
        description = f"https://amigo.geneontology.org/amigo/term/{go_id}"
        lines.append("\t".join((label, description, *programs[go_id])))
    destination.write_text("\n".join(lines) + "\n")
    return destination


def build_go_bp_resource(config, hvg_genes):
    """Verify sources and produce a deterministic, Stage-1-specific GMT and manifest."""
    resource = config["resource"]
    supported = {
        "namespace": "biological_process",
        "ancestor_relations": ["is_a", "part_of"],
        "exclude_qualifier": "NOT",
        "excluded_term_subsets": ["gocheck_do_not_annotate"],
    }
    for key, value in supported.items():
        if resource[key] != value:
            raise ValueError(f"Unsupported GO derivation setting {key}={resource[key]!r}")
    raw_directory = Path(resource["raw_directory"])
    source_paths = {}
    for name, source in resource["sources"].items():
        source_paths[name] = ensure_download(
            source["url"],
            raw_directory / source["filename"],
            source["bytes"],
            source["sha256"],
        )
    obo_header, terms, aliases = parse_go_basic(source_paths["ontology"])
    gaf_header, annotations, annotation_stats = parse_human_bp_annotations(
        source_paths["human_annotations"],
        terms,
        aliases,
        resource["taxon"],
        resource["excluded_evidence_codes"],
    )
    minimum, maximum = resource["hvg_program_size"]
    programs = derive_hvg_programs(terms, annotations, hvg_genes, minimum, maximum)
    gmt_path = write_gmt(programs, terms, resource["gmt_path"])
    gene_union = set().union(*programs.values()) if programs else set()
    payload = {
        "collection": resource["collection"],
        "release": resource["release"],
        "ontology_version": resource["ontology_version"],
        "archive_doi": resource["archive_doi"],
        "license": resource["license"],
        "license_url": resource["license_url"],
        "sources": resource["sources"],
        "derivation": {
            "namespace": resource["namespace"],
            "taxon": resource["taxon"],
            "ancestor_relations": resource["ancestor_relations"],
            "exclude_qualifier": resource["exclude_qualifier"],
            "include_evidence_codes": resource["include_evidence_codes"],
            "excluded_evidence_codes": resource["excluded_evidence_codes"],
            "excluded_term_subsets": resource["excluded_term_subsets"],
            "hvg_program_size": resource["hvg_program_size"],
            "root_excluded": "GO:0008150",
            "genes_are_frozen_hvg_symbols": True,
        },
        "headers": {"obo": obo_header, "gaf": gaf_header},
        "annotation_stats": annotation_stats,
        "output": {
            "gmt_path": str(gmt_path),
            "gmt_sha256": file_sha256(gmt_path),
            "programs": len(programs),
            "unique_hvg_genes": len(gene_union),
            "minimum_program_size": min(map(len, programs.values())),
            "maximum_program_size": max(map(len, programs.values())),
        },
    }
    payload["manifest_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = Path(resource["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_gmt_gene_indices(path, hvg_genes):
    """Load GMT programs as deterministic gene-index tensors without model metadata."""
    import torch

    index = {gene: position for position, gene in enumerate(hvg_genes)}
    programs = []
    with Path(path).open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            genes = sorted({index[gene] for gene in fields[2:] if gene in index})
            if genes:
                programs.append(torch.tensor(genes, dtype=torch.long))
    return tuple(programs)
