"""Matched trajectory outputs use the same campaign paths and analysis units."""

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from analysis import (
    artifacts,
    campaign,
    composition,
    deposit,
    initialization,
    loss_trace,
    paths,
    publication,
    targets,
)


@pytest.fixture
def collected(tmp_path, monkeypatch, trajectory_archive):
    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path / "campaigns")
    directory = tmp_path / "short"
    protocol = {
        **targets.INITIALIZATION_SPECS["short"],
        "pairs": 2,
        "updates": 2,
        "initialization_temperature": 1.0,
        "target": "pdl1_igv",
        "target_sequence": "TEST",
        "protocol_id": "shared-test-inputs",
    }
    initialization.save(directory / "protocol.json", protocol)
    initialization.save(directory / "inputs.json", [{"pair": i} for i in range(2)])
    pd.DataFrame({"run": ["runs/pair00", "runs/pair01"], "call_id": ["call-0", "call-1"]}).to_csv(
        directory / "manifest.csv", index=False
    )
    for seed in range(2):
        folder = directory / f"runs/pair{seed:02d}"
        with zipfile.ZipFile(io.BytesIO(trajectory_archive(protocol, seed))) as archive:
            archive.extractall(folder)
    return directory


def test_import_preserves_inputs_and_uses_shared_loaders(collected, monkeypatch):
    folders = campaign.import_short_trajectories(collected)
    all_traces = []
    for arm, _folder in zip(targets.SHORT_TRAJECTORY_ARMS.values(), folders, strict=True):
        state = artifacts.status("pdl1_igv", arm)
        assert state.is_analysable and state.trajectory_only and state.n_complexes == 0
        trace = loss_trace.load_campaign("pdl1_igv", arm)
        assert trace.arm.eq(arm).all() and trace.seed.tolist() == [0, 0, 1, 1]
        assert trace["optimization seed"].tolist() == [81000, 81000, 81001, 81001]
        all_traces.append(trace)
        results = artifacts.load_campaign("pdl1_igv", arm)
        assert "critic_iptm" not in results and "critic_plddt_binder" not in results
        with pytest.raises(KeyError, match="cannot judge viability"):
            composition.viable(results)
        prior = targets.arm(arm).initialization_prior
        for seed in range(2):
            run = paths.campaign_run("pdl1_igv", arm, seed)
            original = collected / f"runs/pair{seed:02d}"
            assert (run / "initialization.pt").read_bytes() == (
                original / "initialization.pt"
            ).read_bytes()
            assert (run / "step_trace.csv").read_bytes() == (
                original / prior / "step_trace.csv"
            ).read_bytes()
    loss_trace.validate_matching(pd.concat(all_traces))
    assert campaign.import_short_trajectories(collected) == folders
    monkeypatch.setattr(targets, "publication_campaigns", lambda: [])
    members = deposit.design_members()
    assert len([p for p in members if p.name.startswith("state")]) == 12
    assert len([p for p in members if p.suffix == ".pdb"]) == 12
    assert len([p for p in members if p.name == "audit.json"]) == 6
    assert all("all_complexes" not in str(p) for p in members)
    missing = paths.campaign_step_complexes("pdl1_igv", arm, 1) / "step001.pdb"
    missing.unlink()
    assert not artifacts.status("pdl1_igv", arm).is_analysable
    with pytest.raises(FileNotFoundError, match="Incomplete campaign"):
        deposit.design_members()


@pytest.mark.parametrize(
    "change", ["features", "sequence", "step", "model", "temperature", "arm", "other_arm"]
)
def test_mismatched_or_incomplete_runs_are_rejected(collected, change):
    folder = collected / "runs/pair00/sheet"
    if change == "features":
        state = torch.load(folder / "state000.pt", weights_only=True)
        state["prepared"]["inputs"]["ref_pos"][0, 0] = 999
        torch.save(state, folder / "state000.pt")
    else:
        file = folder / "step_trace.csv"
        frame = pd.read_csv(file)
        column, value = {
            "sequence": ("binder sequence", "DIFFERENT"),
            "arm": ("arm", "vanilla"),
            "other_arm": ("arm", targets.SHORT_TRAJECTORY_ARMS["helix"]),
            "step": ("step (index)", 99),
            "model": ("inversion model", "different"),
            "temperature": ("temperature (0-1)", 0.8),
        }[change]
        frame.loc[0, column] = value
        frame.to_csv(file, index=False)
    with pytest.raises(
        ValueError, match="Unmatched|Missing or duplicate|Unexpected trajectory arm"
    ):
        campaign.import_short_trajectories(collected)


def test_conflicting_outputs_are_not_overwritten(collected):
    campaign.import_short_trajectories(collected)
    file = collected / "runs/pair00/helix/step001.pdb"
    file.write_text("changed coordinates")
    with pytest.raises(ValueError, match="Different model output"):
        campaign.import_short_trajectories(collected)


def test_step_measurement_keeps_geometry_flags_and_optimizer_metrics(collected, monkeypatch):
    campaign.import_short_trajectories(collected)
    arm = targets.SHORT_TRAJECTORY_ARMS["uniform"]
    monkeypatch.setattr(composition.secondary_structure, "checked_backbone", lambda *args: object())
    monkeypatch.setattr(
        composition.secondary_structure,
        "composition",
        lambda _: SimpleNamespace(
            n_residues=2,
            helix_fraction=0.2,
            sheet_fraction=0.3,
            coil_fraction=0.5,
        ),
    )
    monkeypatch.setattr(composition.secondary_structure, "geometry_pass", lambda _: False)
    measured = composition.measure_steps("pdl1_igv", arm)
    assert len(measured) == 4 and not measured["geometry pass (bool)"].any()
    assert np.isfinite(measured["helix (fraction 0-1)"]).all()
    assert measured["sequence entropy (bits)"].eq(4).all()
    trace = loss_trace.load_campaign("pdl1_igv", arm)
    pd.concat([trace, trace.iloc[:1]]).to_parquet(paths.campaign_trace("pdl1_igv", arm))
    assert not artifacts.status("pdl1_igv", arm).is_analysable
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        composition.measure_steps("pdl1_igv", arm)


def test_partial_collection_can_resume(collected, monkeypatch, trajectory_archive):
    protocol = json.loads((collected / "protocol.json").read_text())
    manifest = pd.read_csv(collected / "manifest.csv")
    arms = campaign._register_short_campaigns(protocol, manifest)
    arm = arms["uniform"]

    def completed(manifest):
        for row in manifest.itertuples(index=False):
            if row.seed == 1:
                yield row, None, "pending"
            else:
                yield row, trajectory_archive(protocol, row.seed), ""

    monkeypatch.setattr(campaign, "completed_jobs", completed)
    result = campaign.collect_campaign("pdl1_igv", arm)
    assert result.n_pending == 1 and result.n_designs == 1 and result.n_viable is None
    assert not artifacts.status("pdl1_igv", arm).is_analysable
    monkeypatch.setattr(
        campaign,
        "completed_jobs",
        lambda manifest: (
            (row, trajectory_archive(protocol, row.seed), "")
            for row in manifest.itertuples(index=False)
        ),
    )
    campaign.collect_initialization(collected.parent, "short", 1)
    assert artifacts.status("pdl1_igv", arm).is_analysable


def test_short_report_excludes_bad_geometry_but_keeps_entropy(collected, monkeypatch):
    campaign.import_short_trajectories(collected)
    monkeypatch.setitem(targets.INITIALIZATION_SPECS["short"], "updates", 2)

    def measured(target, arm):
        trace = loss_trace.load_campaign(target, arm)
        trace["helix (fraction 0-1)"] = np.where(trace.seed == 0, 0.2, 0.8)
        trace["sheet (fraction 0-1)"] = 0.1
        trace["coil (fraction 0-1)"] = 0.9 - trace["helix (fraction 0-1)"]
        trace["geometry pass (bool)"] = trace.seed == 0
        trace["sequence entropy (bits)"] = np.where(trace.seed == 0, 4.0, 3.0)
        return trace

    def unrelated(*args):
        pytest.fail("Short report tried to load full or softness inputs")

    monkeypatch.setattr(composition, "measure_steps", measured)
    monkeypatch.setattr(composition, "load", unrelated)
    monkeypatch.setattr(initialization, "measure_softness", unrelated)
    report = publication.short_initialization_report()["short_step1_means"]
    assert set(report) == set(targets.SHORT_TRAJECTORY_ARMS.values())
    for outcome in report.values():
        assert outcome["helix (fraction 0-1)"] == pytest.approx(0.2)
        assert outcome["sequence entropy (bits)"] == 3.5
