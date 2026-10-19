"""Preserved model inputs must be sufficient, complete and separate from measurements."""

import json
from dataclasses import asdict

import pandas as pd
import pytest
from analysis import composition, deposit, paths, rcsb, reference, targets


def test_composition_ignores_stale_measurements_and_remeasures(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path)
    folder = paths.campaign_dir("test", "vanilla")
    structures = paths.campaign_complexes("test", "vanilla")
    structures.mkdir(parents=True)
    (structures / "seed00.pdb").write_text("saved model output")
    pd.DataFrame({"seed": [0], "helix (fraction 0-1)": [999]}).to_parquet(
        folder / "composition.parquet"
    )
    pd.DataFrame({"seed": [0], "critic_iptm": [0.9], "critic_plddt_binder": [90]}).to_parquet(
        paths.campaign_results("test", "vanilla")
    )
    from types import SimpleNamespace

    measured = SimpleNamespace(
        n_residues=100, helix_fraction=0.2, sheet_fraction=0.5, coil_fraction=0.3
    )
    monkeypatch.setattr(composition.secondary_structure, "binder_composition", lambda _: measured)
    result = composition.load("test", "vanilla")
    assert result.loc[0, "helix (fraction 0-1)"] == 0.2
    measured.helix_fraction = 0.1
    assert composition.load("test", "vanilla").loc[0, "helix (fraction 0-1)"] == 0.1
    (structures / "seed01.pdb").write_text("unexpected seed")
    with pytest.raises(ValueError, match="Structures and score rows"):
        composition.load("test", "vanilla")


def test_design_archive_uses_publication_grid_and_preserves_structures(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path)
    monkeypatch.setattr(targets, "publication_campaigns", lambda: [("test", "vanilla")])
    monkeypatch.setattr(targets, "N_SEEDS", 1)
    folder = paths.campaign_dir("test", "vanilla")
    structures = paths.campaign_complexes("test", "vanilla")
    structures.mkdir(parents=True)
    for name in (
        "all_results.parquet",
        "step_trace.parquet",
        "protocol.json",
        "composition.parquet",
        "summary.json",
    ):
        (folder / name).write_text("input or obsolete measurement")
    (folder / "protocol.json").write_text("{}")
    ignored = tmp_path / "test__exploratory"
    ignored.mkdir()
    (ignored / "all_results.parquet").write_text("not published")
    with pytest.raises(FileNotFoundError, match="Incomplete campaign"):
        deposit.design_members()
    model = structures / "seed00.pdb"
    model.write_text("model coordinates")
    assert set(deposit.design_members()) == {
        model,
        folder / "all_results.parquet",
        folder / "step_trace.parquet",
        folder / "protocol.json",
    }


def test_reference_recalculates_instead_of_reading_old_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REFERENCE", tmp_path)
    (tmp_path / rcsb.SEARCH_HITS).write_text("{}")
    (tmp_path / reference.COMPOSITION_TABLE).write_text("broken,stale\n")
    calls = []

    def measure(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"measurement": [len(calls)]})

    monkeypatch.setattr(rcsb, "build", measure)
    assert reference.composition().measurement.iloc[0] == 1
    assert reference.composition().measurement.iloc[0] == 2
    reference.pairs()
    assert calls[-1] == {"include_excluded": True, "workers": rcsb.DEFAULT_WORKERS}


def test_frozen_entry_metadata_needs_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REFERENCE", tmp_path)
    entity = rcsb.EntityMeta("TEST_1", "TEST", "A", 1, 100, "protein", "species", 1, "cluster")
    entry = rcsb.EntryMeta("TEST", "title", 2.0, "2020-01-01", "2020-01-01", 2, (entity,))
    (tmp_path / rcsb.ENTRY_METADATA).write_text(json.dumps({"TEST": asdict(entry)}))

    def no_network(*args, **kwargs):
        raise AssertionError("Frozen metadata must not query the live API")

    monkeypatch.setattr(rcsb, "post_json", no_network)
    assert rcsb.fetch_entries(["TEST"]) == {"TEST": entry}


def test_unpublished_archive_reports_missing_local_inputs_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(deposit, "ZENODO_RECORD", "PENDING")
    artifact = deposit.DataArtifact("inputs.zip", tmp_path, "model outputs")
    with pytest.raises(FileNotFoundError, match="not been published"):
        artifact.fetch()


def test_existing_archive_can_restore_missing_inputs(tmp_path):
    original = tmp_path / "model.pdb"
    original.write_text("preserved prediction")
    artifact = deposit.DataArtifact("inputs.zip", tmp_path, "model outputs")
    artifact.package([original])
    original.unlink()
    artifact.fetch()
    assert original.read_text() == "preserved prediction"


def test_incomplete_repack_keeps_previous_archive(tmp_path):
    original = tmp_path / "model.pdb"
    original.write_text("preserved prediction")
    artifact = deposit.DataArtifact("inputs.zip", tmp_path, "model outputs")
    artifact.package([original])
    before = artifact.archive_path.read_bytes()
    with pytest.raises(FileNotFoundError, match="Missing archive inputs"):
        artifact.package([original, tmp_path / "missing.pdb"])
    assert artifact.archive_path.read_bytes() == before


def test_initialization_archive_keeps_inputs_and_audits_without_report_copies(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        targets,
        "INITIALIZATION_SPECS",
        {
            "softness": {"inputs": 1},
            "short": {"pairs": 1},
            "full": {"n_seeds": 1, "temperatures": [1]},
        },
    )
    preserved = []
    for experiment, run in (("softness", "input00"),):
        folder = tmp_path / experiment
        work = folder / run
        work.mkdir(parents=True)
        for file in (
            folder / "protocol.json",
            folder / "manifest.csv",
            work / "audit.json",
            work / "initialization.pt",
            work / "model.pdb",
            work / "step_trace.csv",
        ):
            file.write_text("original model input or output")
            preserved.append(file)
        (work / "summary.json").write_text("derived")
        (folder / "summary.csv").write_text("derived")
        pilot = folder / "pilot"
        pilot.mkdir()
        (pilot / "model.pdb").write_text("unreported pilot")
    assert set(deposit.initialization_members(tmp_path)) == set(preserved)
    (tmp_path / "softness/input00/audit.json").unlink()
    with pytest.raises(FileNotFoundError, match="provenance"):
        deposit.initialization_members(tmp_path)
