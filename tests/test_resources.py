import gzip

import torch

from causalcelljepa.actions import (
    contextual_multiteacher_action_payload,
    learned_target_id_payload,
    map_targets,
    multiteacher_action_payload,
    protein_symbols,
)
from causalcelljepa.resources import (
    derive_hvg_programs,
    load_gmt_gene_indices,
    parse_go_basic,
    parse_human_bp_annotations,
    write_gmt,
)


def test_action_mapping_rejects_stale_and_ambiguous_identifiers():
    proteins = {
        "P1": {"Gene Names (primary)": "GENEA", "Gene Names (synonym)": "OLD_A"},
        "P2": {"Gene Names (primary)": "UNRELATED", "Gene Names (synonym)": ""},
        "P3": {"Gene Names (primary)": "GENEB", "Gene Names (synonym)": ""},
        "P4": {"Gene Names (primary)": "CURRENT_C", "Gene Names (synonym)": "GENEC"},
        "P5": {"Gene Names (primary)": "GENED; GENED", "Gene Names (synonym)": ""},
        "P6": {"Gene Names (primary)": "GENED", "Gene Names (synonym)": ""},
    }
    target_ids = {"GENEA": "E1", "GENEB": "E2", "GENEC": "E3", "GENED": "E4"}
    rows = [
        {"From": "E1", "Entry": "P1"},
        {"From": "E2", "Entry": "P2"},
        {"From": "E4", "Entry": "P5"},
        {"From": "E4", "Entry": "P6"},
    ]
    mapped, unknown = map_targets(target_ids, proteins, rows)
    assert protein_symbols("A; B C") == {"A", "B", "C"}
    assert {target: row["accession"] for target, row in mapped.items()} == {
        "GENEA": "P1",
        "GENEB": "P3",
        "GENEC": "P4",
    }
    assert mapped["GENEB"]["method"] == "primary_symbol_fallback"
    assert mapped["GENEC"]["method"] == "synonym_symbol_fallback"
    assert unknown == {
        "GENED": {
            "gene_id": "E4",
            "candidate_accessions": ["P5", "P6"],
            "reason": "no_unique_reviewed_canonical_protein",
        }
    }


def test_learned_target_ids_encode_only_training_vocabulary():
    known, embedding = learned_target_id_payload(
        ["heldout", "train_b", "train_a", "test"], ["train_b", "train_a"]
    )
    assert known.tolist() == [False, True, True, False]
    assert embedding.tolist() == [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]


def test_multiteacher_action_is_deterministic_and_preserves_unknown_policy():
    action = {
        "targets": ["A", "B", "C", "D"],
        "embedding": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "known": torch.tensor([True, True, False, True]),
    }
    programs = {"GO:1": ("A", "B"), "GO:2": ("B", "C"), "GO:3": ("A", "D")}
    first, report = multiteacher_action_payload(action, programs, rank=2)
    second, _ = multiteacher_action_payload(action, programs, rank=2)
    assert torch.equal(first["embedding"], second["embedding"])
    assert first["modality_dims"] == [3, 2]
    assert torch.equal(first["known"], action["known"])
    assert report["targets_with_go"] == 4 and report["target_coverage"] == 1


def test_contextual_multiteacher_action_appends_availability_and_uses_any_teacher():
    action = {
        "targets": ["A", "B", "C"],
        "embedding": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        "known": torch.tensor([True, False, False]),
    }
    payload, report = contextual_multiteacher_action_payload(
        action, {"GO:1": ("B", "C")}, rank=1
    )
    assert payload["embedding"].shape == (3, 6)
    assert payload["embedding"][:, -2:].tolist() == [[1, 0], [0, 1], [0, 1]]
    assert payload["known"].tolist() == [True, True, True]
    assert payload["modality_dims"] == [3, 1]
    assert payload["modality_availability"] is True
    assert report["targets_known_from_any_modality"] == 3


def test_go_parsing_propagation_and_gmt_are_deterministic(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text(
        """format-version: 1.2
data-version: test/2026-01-01

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000001
name: parent process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0000002
name: child process
namespace: biological_process
alt_id: GO:9999999
relationship: part_of GO:0000001 ! parent process

[Term]
id: GO:0000003
name: obsolete process
namespace: biological_process
is_obsolete: true
"""
    )
    gaf = tmp_path / "human.gaf.gz"
    rows = [
        "!gaf-version: 2.2\n",
        "UniProtKB\tP1\tGENE1\tinvolved_in\tGO:9999999\tR\tEXP\t\tP\tN\t\tprotein\ttaxon:9606\t20260101\tGO\t\tP1\n",
        "UniProtKB\tP2\tGENE2\tinvolved_in\tGO:0000002\tR\tIEA\t\tP\tN\t\tprotein\ttaxon:9606\t20260101\tGO\t\tP2\n",
        "UniProtKB\tP3\tGENE3\tNOT|involved_in\tGO:0000002\tR\tEXP\t\tP\tN\t\tprotein\ttaxon:9606\t20260101\tGO\t\tP3\n",
        "UniProtKB\tP4\tGENE4\tinvolved_in\tGO:0000002\tR\tEXP\t\tF\tN\t\tprotein\ttaxon:9606\t20260101\tGO\t\tP4\n",
        "UniProtKB\tP5\tGENE5\tinvolved_in\tGO:0000002\tR\tND\t\tP\tN\t\tprotein\ttaxon:9606\t20260101\tGO\t\tP5\n",
    ]
    with gzip.open(gaf, "wt") as handle:
        handle.writelines(rows)

    header, terms, aliases = parse_go_basic(obo)
    gaf_header, annotations, stats = parse_human_bp_annotations(gaf, terms, aliases)
    programs = derive_hvg_programs(terms, annotations, ["GENE2", "GENE1", "GENE3"], 2, 5)
    assert header["data-version"] == "test/2026-01-01"
    assert gaf_header["gaf-version"] == "2.2"
    assert stats["accepted_rows"] == 2
    assert programs == {"GO:0000001": ("GENE1", "GENE2"), "GO:0000002": ("GENE1", "GENE2")}

    gmt = write_gmt(programs, terms, tmp_path / "programs.gmt")
    loaded = load_gmt_gene_indices(gmt, ["GENE2", "GENE1", "GENE3"])
    assert [program.tolist() for program in loaded] == [[0, 1], [0, 1]]
