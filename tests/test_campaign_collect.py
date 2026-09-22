"""Routing a returned design payload into the per-design table.

The bug this guards against cost a collected campaign. A design payload carried
the whole weight schedule, and the vanilla schedule's waypoint holds an empty
weights mapping, which parquet cannot store: a struct needs at least one field.
The failure is nastier than it sounds because of where it hides. Only an arm
that sets no weights produces an empty mapping, so vanilla breaks and every
steered arm writes fine, which is the opposite of the order anyone would test
in.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest
from analysis import campaign


def a_payload() -> dict[str, object]:
    """The shape analysis.modal_app.design returns, trimmed to what matters."""
    return {
        "steps": 150,
        "target": "pdl1_igv",
        "arm": "vanilla",
        "arm_label": "Vanilla",
        "critic_iptm": 0.94,
        "critic_plddt_binder": 87.4,
        "critic_iptm_by_model": {"ESMFold2-Experimental-Fast": 0.94},
        "schedule": {"key": "vanilla", "waypoints": [{"step": 0, "weights": {}}]},
        "step_trace": [{"step (index)": 0}],
        "complex_pdb": "ATOM ...",
    }


def test_routed_fields_stay_out_of_the_row() -> None:
    record = campaign.design_record(
        7, {key: value for key, value in a_payload().items() if key != "step_trace"}
    )
    assert record["seed"] == 7
    for key in campaign.ROUTED_KEYS:
        assert key not in record


def test_a_populated_nested_field_is_allowed() -> None:
    """The per-critic dicts are nested and fine; only empty ones are not."""
    record = campaign.design_record(
        0,
        {
            "critic_iptm_by_model": {"fast": 0.9, "cutoff": 0.8},
        },
    )
    assert record["critic_iptm_by_model"] == {"fast": 0.9, "cutoff": 0.8}


def test_the_resulting_row_is_parquet_writable(tmp_path) -> None:
    payload = {k: v for k, v in a_payload().items() if k != "step_trace"}
    frame = pd.DataFrame([campaign.design_record(0, payload)])
    path = tmp_path / "all_results.parquet"
    frame.to_parquet(path, index=False)
    assert pd.read_parquet(path).loc[0, "critic_plddt_binder"] == pytest.approx(87.4)


@pytest.mark.parametrize(
    ("field", "value", "where"),
    [
        ("weights", {}, "weights"),
        ("schedule_copy", {"waypoints": [{"weights": {}}]}, "schedule_copy.waypoints[0].weights"),
        ("critic_iptm_by_model", {}, "critic_iptm_by_model"),
        ("seeds_used", [], "seeds_used"),
    ],
)
def test_an_empty_container_is_named_rather_than_written(
    field: str, value: object, where: str
) -> None:
    with pytest.raises(ValueError, match="empty container"):
        campaign.design_record(0, {"critic_iptm": 0.9, field: value})
    assert campaign._empty_container_paths({field: value}) == [where]


def test_the_vanilla_schedule_is_exactly_what_broke_it() -> None:
    """Regression: the real payload, unrouted, has to be rejected."""
    from analysis import targets

    unrouted = {"schedule": targets.arm("vanilla").schedule.to_dict()}
    assert campaign._empty_container_paths(unrouted) == ["schedule.waypoints[0].weights"]


def test_force_rerun_reads_the_documented_variable(monkeypatch) -> None:
    from analysis import deposit

    monkeypatch.delenv(deposit.FORCE_RERUN_ENV, raising=False)
    assert deposit.force_rerun() is False
    monkeypatch.setenv(deposit.FORCE_RERUN_ENV, "1")
    assert deposit.force_rerun() is True
    monkeypatch.setenv(deposit.FORCE_RERUN_ENV, "true")
    assert deposit.force_rerun() is False, "only the literal 1 should arm a destructive path"


def test_the_archive_is_its_own_marker(tmp_path) -> None:
    """A half-finished download must not look like a complete one."""
    from analysis import deposit

    artifact = deposit.DataArtifact("thing.zip", tmp_path, "a thing")
    assert artifact.is_fetched() is False
    (tmp_path / "thing.partial").write_bytes(b"incomplete")
    assert artifact.is_fetched() is False, "a .partial must not satisfy the gate"
    artifact.archive_path.write_bytes(b"complete")
    assert artifact.is_fetched() is True


def test_packaging_round_trips_through_extraction(tmp_path) -> None:
    from analysis import deposit

    source = tmp_path / "nested"
    source.mkdir()
    (source / "table.parquet").write_bytes(b"pretend parquet")
    (tmp_path / "top.json").write_text("{}")

    artifact = deposit.DataArtifact("bundle.zip", tmp_path, "a bundle")
    artifact.package([source / "table.parquet", tmp_path / "top.json"])

    import zipfile

    with zipfile.ZipFile(artifact.archive_path) as archive:
        assert sorted(archive.namelist()) == ["nested/table.parquet", "top.json"]


def _generated_tree(tmp_path, monkeypatch) -> tuple[Path, Path]:
    from analysis import deposit, paths

    reference, campaigns = tmp_path / "reference", tmp_path / "campaigns"
    for directory in (reference, campaigns):
        directory.mkdir()
        (directory / ".gitkeep").touch()
    (reference / deposit.REFERENCE_ARCHIVE).write_bytes(b"archive")
    (reference / "natural_binder_composition.csv").write_text("derived")
    (campaigns / deposit.DESIGN_ARCHIVE).write_bytes(b"archive")
    (campaigns / "egfr_d3__vanilla").mkdir()

    monkeypatch.setattr(paths, "REFERENCE", reference)
    monkeypatch.setattr(paths, "CAMPAIGNS", campaigns)
    return reference, campaigns


def test_clear_generated_refuses_while_the_deposit_is_unminted(tmp_path, monkeypatch) -> None:
    """The tables on disk are the only copies until a record exists to restore from."""
    from analysis import deposit

    reference, campaigns = _generated_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(deposit, "ZENODO_RECORD", "PENDING")

    with pytest.raises(RuntimeError, match=deposit.ZENODO_RECORD_NAME):
        deposit.clear_generated()

    assert (reference / "natural_binder_composition.csv").is_file()
    assert (campaigns / "egfr_d3__vanilla").is_dir()


def test_clear_generated_spares_the_archives(tmp_path, monkeypatch) -> None:
    """FORCE_RERUN skips the fetch, so deleting a download would be gratuitous."""
    from analysis import deposit

    reference, campaigns = _generated_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(deposit, "ZENODO_RECORD", "17000000")
    removed = deposit.clear_generated()

    assert (reference / deposit.REFERENCE_ARCHIVE).is_file()
    assert (campaigns / deposit.DESIGN_ARCHIVE).is_file()
    assert (reference / ".gitkeep").is_file()
    assert not (reference / "natural_binder_composition.csv").exists()
    assert not (campaigns / "egfr_d3__vanilla").exists()
    assert len(removed) == 2
