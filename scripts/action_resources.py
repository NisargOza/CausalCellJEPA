# Download frozen action resources and build the auditable target-to-protein manifest.
# This script performs no ESM inference; model execution lives in scripts/actions.py.
import csv
import gzip
import json
import re
import time
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import anndata as ad
import yaml

from causalcelljepa.actions import map_targets
from causalcelljepa.resources import ensure_download, file_sha256

config_path = Path("configs/action.yaml")
config = yaml.safe_load(config_path.read_text())
resources, action = config["resources"], config["action"]
esm, uniprot = resources["esm2"], resources["uniprot"]
for kind in ("model", "regression"):
    ensure_download(
        esm[f"{kind}_url"],
        esm[f"{kind}_path"],
        esm[f"{kind}_bytes"],
        esm[f"{kind}_sha256"],
    )

proteome_path = Path(uniprot["proteome_path"])
if not proteome_path.exists():
    parameters = {
        "query": uniprot["query"],
        "format": "tsv",
        "fields": "accession,gene_primary,gene_synonym,sequence,length",
        "size": 500,
    }
    url = "https://rest.uniprot.org/uniprotkb/search?" + urlencode(parameters)
    header, lines, releases, release_dates, total = None, [], set(), set(), None
    while url:
        with urlopen(Request(url, headers={"User-Agent": "CausalCellJEPA/0.1"})) as response:
            page = response.read().decode().splitlines()
            total = int(response.headers["X-Total-Results"]) if total is None else total
            assert int(response.headers["X-Total-Results"]) == total
            releases.add(response.headers["X-UniProt-Release"])
            release_dates.add(response.headers["X-UniProt-Release-Date"])
            header = page[0] if header is None else header
            assert page[0] == header
            lines.extend(page[1:])
            match = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
            url = match.group(1) if match else None
    assert total == len(lines) == uniprot["proteome_entries"]
    assert releases == {uniprot["release"]} and release_dates == {uniprot["release_date"]}
    proteome_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = proteome_path.with_suffix(proteome_path.suffix + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
    ) as compressed:
        compressed.write(("\n".join([header, *lines]) + "\n").encode())
    temporary.replace(proteome_path)
assert (proteome_path.stat().st_size, file_sha256(proteome_path)) == (
    uniprot["proteome_bytes"],
    uniprot["proteome_sha256"],
)

replogle = json.loads(Path("manifests/replogle_v1.json").read_text())
targets = set().union(*replogle["targets"]["split"]["targets"].values())
target_gene_ids = {}
for source in yaml.safe_load(Path("configs/replogle.yaml").read_text())["data"]["files"].values():
    data = ad.read_h5ad(Path("data/raw") / source["filename"], backed="r")
    for target, gene_id in data.obs.loc[data.obs["gene"].isin(targets), ["gene", "gene_id"]].itertuples(
        index=False, name=None
    ):
        assert target not in target_gene_ids or target_gene_ids[target] == gene_id
        target_gene_ids[target] = gene_id
    data.file.close()
assert len(target_gene_ids) == action["expected_targets"]

mapping_path = Path(uniprot["id_mapping_path"])
if not mapping_path.exists():
    request = Request(
        "https://rest.uniprot.org/idmapping/run",
        data=urlencode(
            {
                "from": "Ensembl",
                "to": "UniProtKB-Swiss-Prot",
                "ids": ",".join(sorted(target_gene_ids.values())),
            }
        ).encode(),
    )
    job_id = json.loads(urlopen(request).read())["jobId"]
    for _ in range(600):
        status = json.loads(urlopen(f"https://rest.uniprot.org/idmapping/status/{job_id}").read())
        if status.get("jobStatus") == "FINISHED" or "results" in status:
            break
        time.sleep(1)
    assert status.get("jobStatus") == "FINISHED" or "results" in status
    redirect = json.loads(
        urlopen(f"https://rest.uniprot.org/idmapping/details/{job_id}").read()
    )["redirectURL"]
    url = redirect + "?" + urlencode(
        {"format": "tsv", "fields": "accession,gene_primary,gene_synonym,length", "size": 500}
    )
    header, lines, releases, release_dates = None, [], set(), set()
    while url:
        with urlopen(Request(url, headers={"User-Agent": "CausalCellJEPA/0.1"})) as response:
            page = response.read().decode().splitlines()
            releases.add(response.headers["X-UniProt-Release"])
            release_dates.add(response.headers["X-UniProt-Release-Date"])
            header = page[0] if header is None else header
            assert page[0] == header
            lines.extend(page[1:])
            match = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
            url = match.group(1) if match else None
    assert releases == {uniprot["release"]} and release_dates == {uniprot["release_date"]}
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text("\n".join([header, *sorted(lines)]) + "\n")
assert (mapping_path.stat().st_size, file_sha256(mapping_path)) == (
    uniprot["id_mapping_bytes"],
    uniprot["id_mapping_sha256"],
)

with gzip.open(proteome_path, "rt") as handle:
    proteins = {row["Entry"]: row for row in csv.DictReader(handle, delimiter="\t")}
with mapping_path.open() as handle:
    id_mapping_rows = list(csv.DictReader(handle, delimiter="\t"))
assert len(proteins) == uniprot["proteome_entries"]
assert len(id_mapping_rows) == uniprot["id_mapping_rows"]
mapped, unknown = map_targets(target_gene_ids, proteins, id_mapping_rows)
assert len(mapped) == action["expected_known"] and len(mapped) + len(unknown) == len(targets)
entries = {
    target: {
        **mapping,
        "primary_gene": proteins[mapping["accession"]]["Gene Names (primary)"],
        "sequence_length": int(proteins[mapping["accession"]]["Length"]),
        "sequence_sha256": sha256(proteins[mapping["accession"]]["Sequence"].encode()).hexdigest(),
    }
    for target, mapping in mapped.items()
}
split = replogle["targets"]["split"]["targets"]
report = {
    "format_version": 1,
    "replogle_manifest_sha256": replogle["manifest_sha256"],
    "target_split_sha256": replogle["targets"]["split"]["sha256"],
    "resources": {
        "esm2": {key: value for key, value in esm.items() if not key.endswith("_path")},
        "uniprot": {key: value for key, value in uniprot.items() if not key.endswith("_path")},
    },
    "protocol": action,
    "mapping_counts": dict(sorted(Counter(row["method"] for row in mapped.values()).items())),
    "split_counts": {
        name: {
            "known": sum(target in mapped for target in members),
            "unknown": sum(target in unknown for target in members),
        }
        for name, members in split.items()
    },
    "targets": entries,
    "unknown_targets": unknown,
    "runtime": {
        "config_sha256": file_sha256(config_path),
        "actions_source_sha256": file_sha256("src/causalcelljepa/actions.py"),
        "fair_esm": version("fair-esm"),
    },
}
report["manifest_sha256"] = sha256(
    json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(action["manifest_path"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(
    json.dumps(
        {
            "manifest_sha256": report["manifest_sha256"],
            "mapped": len(mapped),
            "unknown": unknown,
            "mapping_counts": report["mapping_counts"],
            "split_counts": report["split_counts"],
        },
        indent=2,
    )
)
