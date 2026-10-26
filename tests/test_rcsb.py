"""Check reference query constraints, pair metadata, and selection flags offline."""

from __future__ import annotations

import pytest
from analysis import rcsb


def _terms(body: dict) -> dict[str, dict]:
    return {node["parameters"]["attribute"]: node["parameters"] for node in body["query"]["nodes"]}


def test_the_query_is_pinned_to_a_release_date() -> None:
    """Check that the live query excludes entries released after the snapshot."""
    terms = _terms(rcsb.search_body(grouped=True))
    pin = terms["rcsb_accession_info.initial_release_date"]
    assert pin["operator"] == "less_or_equal"
    assert pin["value"] == rcsb.SNAPSHOT_RELEASE_DATE


def test_the_query_asks_for_what_the_methodology_claims() -> None:
    terms = _terms(rcsb.search_body(grouped=True))
    assert terms["exptl.method"]["value"] == "X-RAY DIFFRACTION"
    assert terms["rcsb_entry_info.resolution_combined"]["value"] == 2.5
    assert terms["rcsb_entry_info.polymer_entity_count"]["value"] == 2
    assert terms["rcsb_entry_info.polymer_composition"]["value"] == "heteromeric protein"
    length = terms["entity_poly.rcsb_sample_sequence_length"]["value"]
    assert (length["from"], length["to"]) == (60, 200)
    organism = terms["rcsb_entity_source_organism.ncbi_scientific_name"]
    assert organism.get("negation") is True


def test_only_the_grouped_query_clusters() -> None:
    grouped = rcsb.search_body(grouped=True)["request_options"]
    ungrouped = rcsb.search_body(grouped=False)["request_options"]
    assert grouped["group_by"]["similarity_cutoff"] == rcsb.IDENTITY_CUTOFF_PERCENT
    assert grouped["group_by_return_type"] == "representatives"
    assert "group_by" not in ungrouped
    assert grouped["return_all_hits"] is True


def test_both_queries_return_entities_not_entries() -> None:
    """Entity-level return is what lets one entry contribute two binders."""
    for grouped in (True, False):
        assert rcsb.search_body(grouped)["return_type"] == "polymer_entity"


def an_entity(entity_id: str, length: int, description: str = "") -> rcsb.EntityMeta:
    return rcsb.EntityMeta(
        entity_id=entity_id,
        pdb_id=entity_id.split("_")[0],
        auth_chain="A" if entity_id.endswith("1") else "B",
        n_instances=1,
        sample_length_residues=length,
        description=description,
        organism="Homo sapiens",
        taxonomy_id=9606,
        cluster_30="c1",
    )


def an_entry(*entities: rcsb.EntityMeta, title: str = "a complex") -> rcsb.EntryMeta:
    return rcsb.EntryMeta(
        pdb_id=entities[0].pdb_id,
        title=title,
        resolution_angstrom=2.0,
        deposit_date="2020-01-01",
        release_date="2020-06-01",
        deposited_instances=2,
        entities=entities,
    )


def test_each_in_range_entity_becomes_a_binder_in_turn() -> None:
    """Both chains in range means two rows, with the roles swapped."""
    small, also_small = an_entity("1ABC_1", 120), an_entity("1ABC_2", 90)
    entries = {"1ABC": an_entry(small, also_small)}
    frame = rcsb.pair_rows(["1ABC_1", "1ABC_2"], {"1ABC_1", "1ABC_2"}, entries)
    assert len(frame) == 2
    assert set(frame["binder_entity_id"]) == {"1ABC_1", "1ABC_2"}
    assert list(frame["partner_in_range"]) == [1, 1]


def test_an_out_of_range_partner_is_still_the_target() -> None:
    binder, big = an_entity("1ABC_1", 120), an_entity("1ABC_2", 900)
    entries = {"1ABC": an_entry(binder, big)}
    frame = rcsb.pair_rows(["1ABC_1"], {"1ABC_1"}, entries)
    assert len(frame) == 1
    assert frame.loc[0, "target_sample_length_residues"] == 900
    assert frame.loc[0, "partner_in_range"] == 0


def test_design_like_and_scaffold_flags_read_the_right_text() -> None:
    """design_like reads the title too; scaffold flags read only the chain."""
    binder = an_entity("1ABC_1", 120, description="nanobody against a kinase")
    target = an_entity("1ABC_2", 300, description="kinase domain")
    entries = {"1ABC": an_entry(binder, target, title="de novo designed complex")}
    row = rcsb.pair_rows(["1ABC_1"], {"1ABC_1"}, entries).loc[0]
    assert row["binder_design_like"] == 1
    assert row["binder_engineered_scaffold_like"] == 1
    assert row["target_engineered_scaffold_like"] == 0


def test_a_plain_natural_pair_trips_no_flags() -> None:
    binder = an_entity("1ABC_1", 120, description="ribonuclease P protein component")
    target = an_entity("1ABC_2", 300, description="ribonuclease P subunit")
    entries = {"1ABC": an_entry(binder, target, title="crystal structure of RNase P")}
    row = rcsb.pair_rows(["1ABC_1"], {"1ABC_1"}, entries).loc[0]
    assert row["binder_design_like"] == 0
    assert row["binder_engineered_scaffold_like"] == 0


def test_cluster_membership_picks_the_right_cutoff() -> None:
    def membership(group_id: str, method: str, cutoff: int) -> dict[str, object]:
        return {
            "group_id": group_id,
            "aggregation_method": method,
            "similarity_cutoff": cutoff,
        }

    memberships = [
        membership("wrong", "sequence_identity", 50),
        membership("right", "sequence_identity", 30),
        membership("other", "matching_uniprot_accession", 30),
    ]
    assert rcsb.cluster_30(memberships) == "right"
    assert rcsb.cluster_30([]) == ""
    assert rcsb.cluster_30(None) == ""


@pytest.mark.parametrize(
    ("text", "keywords", "expected"),
    [
        ("a Nanobody construct", rcsb.ENGINEERED_SCAFFOLD_KEYWORDS, True),
        ("De Novo design", rcsb.DESIGN_KEYWORDS, True),
        ("ribonuclease", rcsb.ENGINEERED_SCAFFOLD_KEYWORDS, False),
    ],
)
def test_keyword_matching_is_case_insensitive(
    text: str, keywords: tuple[str, ...], expected: bool
) -> None:
    assert rcsb.keyword_hit(text, keywords) is expected
