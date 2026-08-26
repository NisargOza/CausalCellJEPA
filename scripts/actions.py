# Cache frozen ESM-2 target representations after action-resource validation.
# Unknown targets retain zero placeholders plus a mask for the later learned shared action.
import argparse
import csv
import gzip
import json
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import esm
import torch
import yaml

from causalcelljepa.actions import embed_proteins
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config_path = Path("configs/action.yaml")
config = yaml.safe_load(config_path.read_text())
resources, action = config["resources"], config["action"]
esm_config, uniprot = resources["esm2"], resources["uniprot"]
manifest_path = Path(action["manifest_path"])
manifest = json.loads(manifest_path.read_text())
declared_hash = manifest.pop("manifest_sha256")
assert declared_hash == sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
manifest["manifest_sha256"] = declared_hash
assert manifest["runtime"]["config_sha256"] == file_sha256(config_path)
assert manifest["runtime"]["actions_source_sha256"] == file_sha256(
    "src/causalcelljepa/actions.py"
)
for kind in ("model", "regression"):
    assert file_sha256(esm_config[f"{kind}_path"]) == esm_config[f"{kind}_sha256"]
assert file_sha256(uniprot["proteome_path"]) == uniprot["proteome_sha256"]
assert file_sha256(uniprot["id_mapping_path"]) == uniprot["id_mapping_sha256"]

with gzip.open(uniprot["proteome_path"], "rt") as handle:
    protein_rows = {row["Entry"]: row for row in csv.DictReader(handle, delimiter="\t")}
sequences = {}
for entry in manifest["targets"].values():
    accession = entry["accession"]
    sequence = protein_rows[accession]["Sequence"]
    assert len(sequence) == entry["sequence_length"]
    assert sha256(sequence.encode()).hexdigest() == entry["sequence_sha256"]
    sequences[accession] = sequence

with torch.serialization.safe_globals([argparse.Namespace]):
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(
        Path(esm_config["model_path"])
    )
model.eval().requires_grad_(False)
assert model.num_layers == action["representation_layer"]
assert model.embed_dim == action["input_dim"]
started = time.perf_counter()
embeddings, stats = embed_proteins(
    model,
    alphabet,
    sequences,
    action["representation_layer"],
    action["sequence_chunk_residues"],
    action["token_budget"],
)
runtime = time.perf_counter() - started
targets = sorted([*manifest["targets"], *manifest["unknown_targets"]])
known = torch.tensor([target in manifest["targets"] for target in targets])
matrix = torch.zeros(len(targets), action["input_dim"])
accessions = []
for index, target in enumerate(targets):
    accession = manifest["targets"].get(target, {}).get("accession", "")
    accessions.append(accession)
    if accession:
        matrix[index] = embeddings[accession]
assert len(targets) == action["expected_targets"] and int(known.sum()) == action["expected_known"]
assert torch.isfinite(matrix).all() and torch.count_nonzero(matrix[~known]) == 0
output = Path(action["output_path"])
assert not output.exists()
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
torch.save(
    {
        "format_version": 1,
        "targets": targets,
        "accessions": accessions,
        "known": known,
        "embedding": matrix,
        "input_dim": action["input_dim"],
        "projection_dim": action["projection_dim"],
        "mechanism": action["mechanism"],
        "unknown_policy": action["unknown_policy"],
        "provenance": {
            "action_manifest_sha256": declared_hash,
            "config_sha256": file_sha256(config_path),
            "model_sha256": esm_config["model_sha256"],
            "uniprot_release": uniprot["release"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "fair_esm": version("fair-esm"),
            "git": _git_state(),
        },
    },
    temporary,
)
temporary.replace(output)
print(
    json.dumps(
        {
            "output": str(output),
            "sha256": file_sha256(output),
            "targets": len(targets),
            "known": int(known.sum()),
            "unique_proteins": stats["proteins"],
            "chunks": stats["chunks"],
            "runtime_seconds": runtime,
            "embedding_min": float(matrix[known].min()),
            "embedding_max": float(matrix[known].max()),
        },
        indent=2,
    )
)
