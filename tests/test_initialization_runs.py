"""The maintained workflow must prepare matched inputs without private EDA files."""

import io

import numpy as np
import pandas as pd
import pytest
import torch
from analysis import initialization as workflow
from analysis import paths, targets
from analysis.priors import (
    AA,
    matched_probabilities,
    paired_inputs,
    tempered_sheet,
)


@pytest.fixture
def prior():
    # Synthetic normalized frequencies exercise the algorithm without model data.
    helix = np.linspace(1, 3, 19)
    sheet = np.linspace(3, 1, 19)
    return {
        "amino_acid_order": AA,
        "helix": (helix / helix.sum()).tolist(),
        "sheet": (sheet / sheet.sum()).tolist(),
    }


def test_temperature_preserves_argmax_and_increases_softness(prior):
    entropy, boosts = [], []
    for temperature in (1, 2, 4):
        q, boost = tempered_sheet(prior, temperature)
        p = matched_probabilities(q, "".join(AA), boost)
        np.testing.assert_array_equal(p.argmax(-1), np.arange(19))
        np.testing.assert_allclose(p.sum(-1), 1)
        assert (p > 0).all()
        entropy.append(float(-(p * np.log2(p)).sum(-1).mean()))
        boosts.append(boost)
    assert entropy[0] < entropy[1] < entropy[2]
    assert boosts[0] == 0.12 and boosts[0] > boosts[1] > boosts[2]


def test_paired_priors_have_identical_discrete_inputs(prior):
    bundles = paired_inputs()
    assert len(bundles) == 5
    for bundle in bundles:
        h, e = (
            matched_probabilities(prior[a], bundle["sequence"], 0.12) for a in ("helix", "sheet")
        )
        np.testing.assert_array_equal(h.argmax(-1), e.argmax(-1))
        assert h.shape == e.shape == (120, 19)
        assert "".join(AA[i] for i in h.argmax(-1)) == bundle["sequence"]
        np.testing.assert_allclose(
            h - e,
            np.broadcast_to(np.asarray(prior["helix"]) - prior["sheet"], h.shape),
            atol=1e-15,
        )


def test_prepare_full_temperatures_in_checkout_without_exploratory(tmp_path, monkeypatch, prior):
    monkeypatch.setattr(workflow, "from_reference", lambda: prior)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path / "data/campaigns")
    source = tmp_path / "src/analysis/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Source snapshot for the isolated preparation test\n")
    initial = pd.DataFrame(
        {"seed": range(30), "step (index)": 0, "binder sequence": ["ADEFGHIKLMNPQRSTVWY" * 4] * 30}
    )
    for arm in targets.INITIALIZATION_BASELINES:
        folder = paths.campaign_dir("pdl1_igv", arm)
        folder.mkdir(parents=True)
        for name in ("step_trace.parquet", "composition.parquet", "all_results.parquet"):
            initial.to_parquet(folder / name)
    root = tmp_path / "runs"
    short = workflow.prepare_inputs(root, "short", 1)
    assert len(workflow.read(short / "inputs.json")) == 5
    t1 = workflow.prepare_inputs(root, "full", 1)
    inputs = workflow.read(t1 / "inputs.json")
    for item in inputs:
        folder = t1 / f"seed{item['seed']:02d}"
        folder.mkdir()
        torch.save(
            {"bundle": item, "features": {"test": torch.tensor([item["seed"]])}},
            folder / "initialization.pt",
        )
    for temperature in (2, 4):
        out = workflow.prepare_inputs(root, "full", temperature)
        protocol = workflow.read(out / "protocol.json")
        assert protocol["initialization_temperature"] == temperature
        assert protocol["seeds"] == list(range(30))
        for item in workflow.read(out / "inputs.json"):
            blob = (out / "inputs" / f"seed{item['seed']:02d}-features.pt").read_bytes()
            features = torch.load(io.BytesIO(blob), weights_only=True)
            assert features["test"].item() == item["seed"]
            assert item["initial_sequence"] == inputs[item["seed"]]["initial_sequence"]
    assert not (tmp_path / "exploratory").exists()
    assert not (tmp_path / "data/exploratory").exists()
    before = (t1 / "protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        workflow.prepare_inputs(root, "full", 1)
    assert (t1 / "protocol.json").read_bytes() == before


def test_stale_source_prevents_launch_before_any_remote_call(tmp_path, monkeypatch):
    directory = tmp_path / "short"
    workflow.save(directory / "protocol.json", {"source_sha256": {"old": "digest"}})
    monkeypatch.setattr(workflow, "source_hashes", lambda: {"new": "digest"})
    with pytest.raises(RuntimeError, match="Source changed"):
        workflow.prepared_jobs(tmp_path, "short", 1)


@pytest.mark.parametrize("experiment", ["softness", "short", "full"])
def test_shared_cli_prepares_submits_and_collects_initialization(
    tmp_path, monkeypatch, prior, experiment
):
    import json
    import zipfile
    from types import SimpleNamespace

    import modal
    from analysis import campaign, modal_app
    from typer.testing import CliRunner

    monkeypatch.setattr(workflow, "from_reference", lambda: prior)
    monkeypatch.setattr(paths, "CAMPAIGNS", tmp_path / "campaigns")
    trace = paths.campaign_trace("pdl1_igv", "vanilla")
    trace.parent.mkdir(parents=True)
    pd.DataFrame(
        {"seed": range(30), "step (index)": 0, "binder sequence": ["ADEFGHIKLMNPQRSTVWY"] * 30}
    ).to_parquet(trace)
    screen_inputs = {f"input{i:02d}.pt": f"tensor-{i}".encode() for i in range(10)}
    monkeypatch.setattr(
        modal.Function, "from_name", lambda app, name: SimpleNamespace(remote=lambda: screen_inputs)
    )
    submitted, requested, options, instances = [], [], [], []

    class RemoteClass:
        def with_options(self, **kwargs):
            options.append(kwargs)
            return self

        def __call__(self, **kwargs):
            instances.append(kwargs)
            return self

        def __getattr__(self, name):
            def submit(**kwargs):
                submitted.append((name, kwargs))
                return SimpleNamespace(object_id=f"call-{len(submitted)}")

            return SimpleNamespace(spawn=submit)

    def from_name(app, cls):
        requested.append((app, cls))
        return RemoteClass()

    monkeypatch.setattr(modal.Cls, "from_name", from_name)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("model.pdb", "model output")
        archive.writestr("initialization.pt", "original tensor")
    monkeypatch.setattr(
        modal.FunctionCall,
        "from_id",
        lambda _: SimpleNamespace(get=lambda **kwargs: payload.getvalue()),
    )
    runner = CliRunner()
    args = ["--experiment", experiment, "--root", str(tmp_path / "runs")]
    for verb in ["prepare", "spawn", "spawn", "collect", "collect"]:
        result = runner.invoke(campaign.app, [verb, *args])
        assert result.exit_code == 0, (result.output, result.exception)
    expected = {"softness": 11, "short": 5, "full": 30}[experiment]
    assert len(submitted) == expected
    expected_class = "FixedInputModal" if experiment == "softness" else campaign.REMOTE_CLASS
    assert all(item == (modal_app.APP_NAME, expected_class) for item in requested)
    if experiment != "softness":
        assert all(item == {"load_critics": experiment == "full"} for item in instances)
        assert all(
            item == {"timeout": 7200, "max_containers": 10 if experiment == "full" else 5}
            for item in options
        )
    directory = workflow.experiment_dir(tmp_path / "runs", experiment, 1)
    frame = pd.read_csv(directory / "manifest.csv")
    assert len(frame) == expected
    assert not (directory / "calls.json").exists()
    for name in frame.run:
        assert (directory / name / "model.pdb").read_text() == "model output"
    protocol = json.loads((directory / "protocol.json").read_text())
    assert protocol["app"] == modal_app.APP_NAME
    if experiment == "short":
        for method, kwargs in submitted:
            assert method == "design_pair"
            h, e = (np.asarray(kwargs["bundle"][arm]) for arm in ("helix", "sheet"))
            np.testing.assert_array_equal(h.argmax(-1), e.argmax(-1))
    elif experiment == "full":
        assert all(method == "design_prior" for method, _ in submitted)
        assert all(kwargs["protocol"]["steps"] == 150 for _, kwargs in submitted)
    else:
        assert [method for method, _ in submitted] == ["screen"] * 10 + ["controls"]


def test_prepared_prior_change_is_rejected_before_submission(tmp_path, monkeypatch, prior):
    monkeypatch.setattr(workflow, "from_reference", lambda: prior)
    workflow.prepare_inputs(tmp_path, "short", 1)
    prior["sheet"] = list(reversed(prior["sheet"]))
    with pytest.raises(ValueError, match="Reference prior changed"):
        workflow.prepared_jobs(tmp_path, "short", 1)


def test_shared_worker_methods_delegate_without_changing_inputs(monkeypatch):
    import sys
    from types import ModuleType

    from analysis import modal_app

    # The CPU environment deliberately has no GPU-only esm dependency.
    design = ModuleType("analysis.design")
    gpu = ModuleType("analysis.design.initialization")
    design.initialization = gpu
    monkeypatch.setitem(sys.modules, "analysis.design", design)
    monkeypatch.setitem(sys.modules, "analysis.design.initialization", gpu)

    cls = modal_app.ESMFold2DesignModal._get_user_cls()
    worker = cls()
    worker.load_critics = True
    worker.driver, worker.audit, worker.sources = object(), {}, {}
    calls = []

    def record(*args):
        calls.append(args)
        return b"model outputs"

    gpu.design_prior = record
    gpu.design_pair = record
    bundle, protocol = {"seed": 0}, {"steps": 150}
    # Modal wraps methods; the underlying callable runs without a deployment or GPU.
    for name in ("design_prior", "design_pair"):
        method = getattr(cls, name)
        assert method._get_raw_f()(worker, bundle, protocol) == b"model outputs"
    assert all(
        args == (worker.driver, worker.audit, worker.sources, bundle, protocol, modal_app.study)
        for args in calls
    )
