import gzip

from causalcelljepa.resources import (
    derive_hvg_programs,
    load_gmt_gene_indices,
    parse_go_basic,
    parse_human_bp_annotations,
    write_gmt,
)


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
