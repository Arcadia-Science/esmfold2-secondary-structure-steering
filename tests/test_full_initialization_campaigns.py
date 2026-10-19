"""Full prior runs use campaign collection, readers, and archives without a GPU."""

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from analysis import (
    artifacts,
    campaign,
    composition,
    deposit,
    initialization,
    loss_trace,
    paths,
    targets,
)
from analysis.priors import AA
from typer.testing import CliRunner


@pytest.fixture
def full_campaign(tmp_path, monkeypatch, full_design_archive):
    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path / "campaigns")
    prior = {name: [1 / len(AA)] * len(AA) for name in ("uniform", "helix", "sheet")}
    monkeypatch.setattr(initialization, "from_reference", lambda: {"amino_acid_order": AA, **prior})
    monkeypatch.setitem(
        targets.INITIALIZATION_SPECS,
        "full",
        {
            **targets.INITIALIZATION_SPECS["full"],
            "n_seeds": 2,
            "steps": 3,
        },
    )
    baseline = paths.campaign_trace("pdl1_igv", "vanilla")
    baseline.parent.mkdir(parents=True)
    sequence = "".join(AA)
    pd.DataFrame(
        [
            {
                "target": "pdl1_igv",
                "arm": "vanilla",
                "seed": seed,
                "step (index)": step,
                "binder sequence": sequence,
                "inversion model": "model1",
                "temperature (0-1)": 0.999,
            }
            for seed in range(2)
            for step in range(3)
        ]
    ).to_parquet(baseline)
    root = tmp_path / "preparation"
    directory = initialization.prepare_inputs(root, "full", 1)
    protocol, _ = initialization.prepared_jobs(root, "full", 1)
    arm = campaign._register_full_campaign(directory)
    pd.DataFrame({"seed": [0, 1], "call_id": ["call-0", "call-1"]}).to_csv(
        paths.campaign_manifest("pdl1_igv", arm), index=False
    )
    payloads = {seed: full_design_archive(protocol, seed, sequence) for seed in range(2)}
    monkeypatch.setattr(
        campaign,
        "completed_jobs",
        lambda manifest: (
            (row, payloads[row.seed], "") for row in manifest.itertuples(index=False)
        ),
    )
    return SimpleNamespace(arm=arm, payloads=payloads, root=root, directory=directory)


def test_full_outputs_round_trip_through_shared_readers_and_archive(full_campaign, monkeypatch):
    arm = full_campaign.arm
    result = campaign.collect_campaign("pdl1_igv", arm)
    assert result.n_designs == 2 and result.n_trace_rows == 6 and result.n_viable == 0
    assert artifacts.status("pdl1_igv", arm).is_analysable
    measured = composition.measure_campaign("pdl1_igv", arm)
    assert measured.arm.eq(arm).all()
    assert measured["iptm (0-1)"].eq(0.9).all()
    assert measured["binder pLDDT (0-100)"].eq(85).all()
    assert np.allclose(
        measured[[f"{s} (fraction 0-1)" for s in ("helix", "sheet", "coil")]].sum(axis=1), 1
    )
    trace = loss_trace.load_campaign("pdl1_igv", arm)
    loss_trace.validate_steps(trace, n_seeds=2, steps=3)
    scores = artifacts.load_campaign("pdl1_igv", arm)
    assert set(scores.loc[0, "critic_iptm_by_model"]) == set(targets.CRITIC_MODELS)
    for seed, payload in full_campaign.payloads.items():
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                assert (
                    paths.campaign_run("pdl1_igv", arm, seed) / name
                ).read_bytes() == archive.read(name)
            assert (
                paths.campaign_complexes("pdl1_igv", arm) / f"seed{seed:02d}.pdb"
            ).read_bytes() == archive.read(f"{targets.PRIMARY_CRITIC}.pdb")
    assert not (full_campaign.directory / "seed00").exists()

    monkeypatch.setattr(targets, "publication_campaigns", lambda: [])
    members = deposit.design_members()
    artifact = deposit.DataArtifact("inputs.zip", paths.CAMPAIGNS, "test campaigns")
    with zipfile.ZipFile(artifact.package(members)) as archive:
        for file in members:
            assert archive.read(str(file.relative_to(paths.CAMPAIGNS))) == file.read_bytes()
    assert len([file for file in members if file.name == "sequence-states.pt"]) == 2
    assert len([file for file in members if file.suffix == ".pdb"]) == 10
    (paths.campaign_run("pdl1_igv", arm, 0) / "audit.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete campaign"):
        deposit.design_members()


def test_partial_full_collection_resumes_from_local_runs(full_campaign, monkeypatch):
    polled = []

    def pending(manifest):
        for row in manifest.itertuples(index=False):
            polled.append(row.seed)
            yield (
                row,
                full_campaign.payloads[row.seed] if row.seed == 0 else None,
                "pending" if row.seed else "",
            )

    monkeypatch.setattr(campaign, "completed_jobs", pending)
    result = campaign.collect_campaign("pdl1_igv", full_campaign.arm)
    assert (result.n_designs, result.n_pending) == (1, 1)

    def finished(manifest):
        for row in manifest.itertuples(index=False):
            polled.append(row.seed)
            yield row, full_campaign.payloads[row.seed], ""

    monkeypatch.setattr(campaign, "completed_jobs", finished)
    args = ["collect", "--target", "pdl1_igv", "--arm", full_campaign.arm]
    for _ in range(2):
        result = CliRunner().invoke(campaign.app, args)
        assert result.exit_code == 0, (result.output, result.exception)
    assert polled == [0, 1, 1]
    assert len(artifacts.load_campaign("pdl1_igv", full_campaign.arm)) == 2


@pytest.mark.parametrize(
    "change, message",
    [
        ("arm", "Unexpected full-design trace arm"),
        ("sequence", "Unmatched initial sequence"),
        ("schedule", "Unmatched initial sequence"),
        ("critic", "Missing or unexpected critics"),
        ("confidence", "Expected ipTM from every campaign critic"),
        ("selected", "Critic sequence differs"),
        ("provenance", "Incomplete full design"),
    ],
)
def test_invalid_full_results_are_rejected_without_relabeling(full_campaign, change, message):
    with zipfile.ZipFile(io.BytesIO(full_campaign.payloads[0])) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    if change in {"arm", "sequence", "schedule"}:
        trace = pd.read_csv(io.BytesIO(files["step_trace.csv"]))
        column, value = {
            "arm": ("arm", "vanilla"),
            "sequence": ("binder sequence", "I" * len(AA)),
            "schedule": ("inversion model", "different-model"),
        }[change]
        trace.loc[0, column] = value
        files["step_trace.csv"] = trace.to_csv(index=False).encode()
        files["step_trace.jsonl"] = trace.to_json(orient="records", lines=True).encode()
    elif change in {"critic", "selected", "confidence"}:
        critics = json.loads(files["critics.json"])
        if change == "critic":
            critics.pop()
        elif change == "confidence":
            critics[-1]["iptm"] = None
        else:
            critics[0]["designed_sequence"] = "A|II"
        files["critics.json"] = json.dumps(critics).encode()
    else:
        files.pop("sequence-states.pt")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    full_campaign.payloads[0] = payload.getvalue()
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        campaign.collect_campaign("pdl1_igv", full_campaign.arm)
    assert not paths.campaign_results("pdl1_igv", full_campaign.arm).exists()
    assert not paths.campaign_run("pdl1_igv", full_campaign.arm, 0).exists()


def test_full_submission_resumes_using_the_campaign_manifest(full_campaign, monkeypatch):
    import modal

    manifest = paths.campaign_manifest("pdl1_igv", full_campaign.arm)
    manifest.unlink()
    submitted = []
    interrupt = True

    def submit(**kwargs):
        seed = kwargs["bundle"]["seed"]
        if interrupt and seed == 1:
            raise ConnectionError("interrupted submission")
        submitted.append(seed)
        return SimpleNamespace(object_id=f"call-{seed}")

    remote = SimpleNamespace(design_prior=SimpleNamespace(spawn=submit))
    remote.with_options = lambda **kwargs: lambda **kwargs: remote
    monkeypatch.setattr(modal.Cls, "from_name", lambda *args: remote)
    with pytest.raises(ConnectionError, match="interrupted submission"):
        campaign.spawn_initialization(full_campaign.root, "full", 1)
    assert pd.read_csv(manifest).seed.tolist() == [0]
    interrupt = False
    campaign.spawn_initialization(full_campaign.root, "full", 1)
    assert submitted == [0, 1]
    assert pd.read_csv(manifest).seed.tolist() == [0, 1]
    assert not (full_campaign.directory / "manifest.csv").exists()
