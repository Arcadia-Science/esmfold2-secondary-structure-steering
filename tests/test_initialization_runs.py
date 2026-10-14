"""The maintained workflow must prepare matched inputs without private EDA files."""

import io
import math
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from analysis import initialization as workflow
from analysis import paths, targets
from analysis.priors import (
    AA,
    matched_probabilities,
    minimum_boosts,
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
        q = tempered_sheet(prior, temperature)
        p = matched_probabilities(q, "".join(AA))
        np.testing.assert_array_equal(p.argmax(-1), np.arange(19))
        np.testing.assert_allclose(p.sum(-1), 1)
        assert (p > 0).all()
        entropy.append(float(-(p * np.log2(p)).sum(-1).mean()))
        boosts.append(minimum_boosts(q, "".join(AA)).mean())
    assert entropy[0] < entropy[1] < entropy[2]
    assert boosts[0] > boosts[1] > boosts[2]


def test_paired_priors_have_identical_discrete_inputs(prior):
    bundles = paired_inputs()
    assert len(bundles) == 5
    for bundle in bundles:
        h, e = (matched_probabilities(prior[a], bundle["sequence"]) for a in ("helix", "sheet"))
        np.testing.assert_array_equal(h.argmax(-1), e.argmax(-1))
        assert h.shape == e.shape == (120, 19)
        assert "".join(AA[i] for i in h.argmax(-1)) == bundle["sequence"]
        # Each prior gets its own adjustment, even with identical hard identities.
        assert not np.allclose(
            minimum_boosts(prior["helix"], bundle["sequence"]),
            minimum_boosts(prior["sheet"], bundle["sequence"]),
        )


@pytest.mark.parametrize("arm", ["helix", "sheet"])
def test_boost_is_minimal_and_already_favored_identity_is_unchanged(prior, arm):
    q = np.asarray(prior[arm])
    margin = targets.INITIALIZATION_MARGIN
    sequence = "".join(AA)
    original = q.copy()
    boosts = minimum_boosts(q, sequence)
    p = matched_probabilities(q, sequence)
    np.testing.assert_array_equal(q, original)
    np.testing.assert_allclose(p.sum(-1), 1)
    for i, boost in enumerate(boosts):
        gap = p[i, i] - np.delete(p[i], i).max()
        assert gap >= margin - 1e-14
        if boost == 0:
            np.testing.assert_array_equal(p[i], q)
        else:
            assert gap == pytest.approx(margin)
            smaller = q + boost / 2 * (np.eye(19)[i] - 1 / 19)
            assert smaller[i] - np.delete(smaller, i).max() < margin
    assert boosts[q.argmax()] == 0


def test_uniform_prior_and_tied_maximum():
    q = np.full(19, 1 / 19)
    p = matched_probabilities(q, "AI")
    assert p[0, AA.index("A")] - p[0, AA.index("I")] == pytest.approx(0.001)
    assert p[1, AA.index("I")] - p[1, AA.index("A")] == pytest.approx(0.001)
    q[[0, 1]] += 0.01
    q[2:] -= 0.02 / 17
    p = matched_probabilities(q, "A")
    assert p[0, 0] - p[0, 1] == pytest.approx(0.001)


@pytest.mark.parametrize("margin", [0, -0.001, float("nan"), float("inf")])
def test_invalid_margin_is_rejected(prior, margin):
    with pytest.raises(ValueError, match="margin"):
        matched_probabilities(prior["helix"], "I", margin=margin)


@pytest.mark.parametrize("sequence", ["", "C", "X", "a"])
def test_invalid_sequence_is_rejected(prior, sequence):
    with pytest.raises(ValueError, match="sequence"):
        matched_probabilities(prior["helix"], sequence)


@pytest.mark.parametrize(
    "q",
    [
        np.ones(18) / 18,
        np.ones(19),
        np.zeros(19),
        np.full(19, np.nan),
        np.full(19, np.inf),
        np.array([-1] + [2 / 18] * 18),
    ],
)
def test_invalid_prior_is_rejected(q):
    with pytest.raises(ValueError):
        matched_probabilities(q, "I")


def test_infeasible_boost_is_rejected_without_clipping():
    q = np.array([0.98] + [0.02 / 18] * 18)
    with pytest.raises(ValueError, match="nonpositive"):
        matched_probabilities(q, "I")


def test_too_small_margin_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        matched_probabilities(np.full(19, 1 / 19), "I", margin=1e-30)


@pytest.mark.parametrize("arm", ["helix", "sheet"])
@pytest.mark.parametrize("temperature", [1, 2, 4])
def test_worker_logit_conversion_preserves_probabilities_and_identities(prior, arm, temperature):
    module = Path(__file__).resolve().parents[1] / "src/analysis/design/initialization.py"
    prior_logits = runpy.run_path(str(module))["prior_logits"]
    # Model order differs from the prior order; cysteine is excluded from the prior.
    order = list(reversed("ACDEFGHIKLMNPQRSTVWY"))
    sequence = "".join(AA)
    q = tempered_sheet({**prior, "sheet": prior[arm]}, temperature)
    t0 = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi / 150))
    logits, p = prior_logits(q, sequence, order, t0, margin=targets.INITIALIZATION_MARGIN)
    assert logits.dtype == p.dtype == torch.float32
    recovered = (logits / t0).softmax(-1)
    np.testing.assert_allclose(
        recovered[0, :, [order.index(aa) for aa in AA]],
        matched_probabilities(q, sequence),
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.equal(recovered.argmax(-1)[0], torch.tensor([order.index(aa) for aa in sequence]))
    assert torch.equal(recovered.to(torch.bfloat16).argmax(-1), recovered.argmax(-1))
    assert recovered[..., order.index("C")].eq(0).all()


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
    assert workflow.protocol(short, "short")["probability_margin"] == 0.001
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
        assert protocol["probability_margin"] == 0.001
        assert "boost" not in protocol and "epsilon" not in protocol
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
            for arm in ("helix", "sheet"):
                np.testing.assert_allclose(
                    kwargs["bundle"][arm],
                    matched_probabilities(prior[arm], kwargs["bundle"]["sequence"]),
                )
                assert kwargs["protocol"]["priors"][arm] == prior[arm]
    elif experiment == "full":
        assert all(method == "design_prior" for method, _ in submitted)
        assert all(kwargs["protocol"]["steps"] == 150 for _, kwargs in submitted)
        assert all(kwargs["protocol"]["probability_margin"] == 0.001 for _, kwargs in submitted)
    else:
        assert [method for method, _ in submitted] == ["screen"] * 10 + ["controls"]


def test_prepared_prior_change_is_rejected_before_submission(tmp_path, monkeypatch, prior):
    monkeypatch.setattr(workflow, "from_reference", lambda: prior)
    workflow.prepare_inputs(tmp_path, "short", 1)
    prior["sheet"] = list(reversed(prior["sheet"]))
    with pytest.raises(ValueError, match="Reference prior changed"):
        workflow.prepared_jobs(tmp_path, "short", 1)


def test_prepared_margin_change_is_rejected_before_submission(tmp_path, monkeypatch, prior):
    monkeypatch.setattr(workflow, "from_reference", lambda: prior)
    directory = workflow.prepare_inputs(tmp_path, "short", 1)
    protocol = workflow.read(directory / "protocol.json")
    protocol["probability_margin"] *= 2
    workflow.save(directory / "protocol.json", protocol)
    with pytest.raises(ValueError, match="Prepared protocol changed"):
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
