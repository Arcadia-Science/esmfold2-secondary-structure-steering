"""Run the logging path on CPU, with model dependencies replaced by small functions."""

import __future__
import ast
import io
import json
import logging
import math
import random
import runpy
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from analysis import loss_trace, schedule, targets
from analysis.priors import AA, matched_probabilities


@pytest.fixture
def design_loop():
    file = Path(__file__).resolve().parents[1] / "src/analysis/design/loop.py"
    functions = [
        node
        for node in ast.parse(file.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"design_binder", "normalized_gradient_tensor"}
    ]
    cpu_torch = SimpleNamespace(**vars(torch))
    cpu_torch.device = lambda _: torch.device("cpu")
    cpu_torch.cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
    )
    namespace = {
        "torch": cpu_torch,
        "F": F,
        "optim": torch.optim,
        "math": math,
        "time": time,
        "random": random,
        "logger": logging.getLogger(__name__),
        "loss_trace": loss_trace,
        "VANILLA": schedule.VANILLA,
        "STEPS": 150,
        "SAMPLING_STEPS": 1,
        "CONFIDENCE_SAMPLING_STEPS": 50,
        "TARGET_SEQUENCES": {},
        "BINDER_PROMPT_FACTORIES": {},
        "seed_context": lambda seed: nullcontext(),
        "sequence_to_one_hot": lambda *args, **kwargs: torch.zeros(1),
        "build_initial_soft_sequence_logits": lambda seq, batch_size: (
            torch.arange(20, dtype=torch.float32).repeat(batch_size, len(seq), 1).requires_grad_()
        ),
        "build_gradient_mask": lambda seq, batch_size: torch.ones(batch_size, len(seq), 20),
        "get_mid_points": lambda: None,
        "fold_and_get_distogram": lambda model, target, hot, design, **kwargs: {
            "seq_list": ["A|II"],
            "distogram_logits": design,
        },
        "compute_structure_losses": lambda design, length: {"total_loss": design.square().sum()},
        "compute_steering_losses": lambda design, *args: SimpleNamespace(
            total=design.sum() * 0, terms={}
        ),
        "compute_esmc_pseudoperplexity_nll": lambda **kwargs: kwargs["binder_design"]
        .square()
        .sum(),
        "compute_distogram_iptm_proxy": lambda *args: {"distogram_iptm_proxy": 0.5},
        "LEARNING_RATE": 0.1,
        "LM_LOSS_BATCH_SIZE": 1,
        "LM_MASK_PASSES": 1,
        "LOG_INTERVAL": 10,
        "BYTES_PER_GIB": 1024**3,
        "TEMPERATURE_MIN": 0.01,
        "TIMING_KEYS": ("time", "peak_allocated_gib", "peak_reserved_gib"),
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(file),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["design_binder"]


@pytest.mark.parametrize("temperature", [1, 1.5, 2, 4])
def test_arm_labels_reach_live_and_final_logs_without_changing_optimization(
    design_loop, tmp_path, temperature
):
    arms = targets.short_trajectory_arms(temperature)
    records = []
    for label in [None, *arms.values(), targets.full_initialization_arm(temperature)]:
        file = tmp_path / f"{label}.jsonl"
        result = design_loop(
            inversion_models={"model": object()},
            hf_critic_models={},
            esmc_model=None,
            target_name="test",
            target_sequence="A",
            binder_name="test",
            binder_sequence="##",
            is_antibody=False,
            seed=0,
            steps=150,
            stop_after_steps=2,
            trace_path=file,
            **({} if label is None else {"arm": label}),
        )
        trace = pd.DataFrame([record.row() for record in result[3]])
        trace.to_csv(file.with_suffix(".csv"), index=False)
        expected = "vanilla" if label is None else label
        assert {json.loads(line)["arm"] for line in file.read_text().splitlines()} == {expected}
        assert pd.read_csv(file.with_suffix(".csv")).arm.eq(expected).all()
        records.append(trace.drop(columns=["arm", "step time (s)"]))
    for trace in records[1:]:
        pd.testing.assert_frame_equal(trace, records[0])


@pytest.mark.parametrize(
    "experiment,pair,prior",
    [("short", 0, "uniform"), ("short", 1, "helix"), ("short", 2, "sheet"), ("full", 0, "sheet")],
)
@pytest.mark.parametrize("temperature", [1, 1.5, 2, 4])
def test_initialization_worker_passes_registered_arm_to_design_loop(
    tmp_path, monkeypatch, experiment, pair, prior, temperature
):
    names = [
        "esm",
        "esm.models",
        "esm.models.esmfold2",
        "esm.models.esmfold2.constants",
        "analysis.design",
        *[
            f"analysis.design.{name}"
            for name in ("constants", "folding", "loop", "losses", "sequence", "structure")
        ],
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    order = list("ACDEFGHIKLMNPQRSTVWY")
    modules["esm.models.esmfold2.constants"].PROTEIN_3TO1 = {aa: aa for aa in order}
    modules["analysis.design.constants"].TOKENS = ["pad", "gap", *order]
    modules["analysis.design.folding"].prepare_folding_inputs = lambda *args, **kwargs: {
        "inputs": {"feature": torch.zeros(1)},
        "seq_list": ["A|II"],
    }
    modules["analysis.design.folding"].fold_and_get_distogram = None
    modules["analysis.design.losses"].get_mid_points = None
    modules["analysis.design.sequence"].sequence_to_one_hot = None
    modules["analysis.design.structure"].build_complex = None
    received = {}

    class ReachedDesignLoop(Exception):
        pass

    def capture(**kwargs):
        received.update(kwargs)
        raise ReachedDesignLoop

    modules["analysis.design.loop"].design_binder = capture
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    file = Path(__file__).resolve().parents[1] / "src/analysis/design/initialization.py"
    worker = runpy.run_path(str(file))["design_pair" if experiment == "short" else "design_prior"]
    monkeypatch.setitem(worker.__globals__, "Path", lambda _: tmp_path)
    arms = ["uniform", "helix", "sheet"]
    q = [1 / len(AA)] * len(AA)
    bundle = {
        "pair": pair,
        "sequence": "II",
        "aa_order": AA,
        **{arm: matched_probabilities(q, "II", temperature=temperature) for arm in arms},
    }
    protocol = {
        **targets.INITIALIZATION_SPECS[experiment],
        "binder_length": 2,
        "target": "test",
        "target_sequence": "A",
        "protocol_id": "test",
        "initialization_temperature": temperature,
        "priors": {arm: q for arm in arms},
    }
    expected_arm = targets.short_trajectory_arms(temperature)[prior]
    if experiment == "full":
        features = io.BytesIO()
        torch.save({"feature": torch.zeros(1)}, features)
        bundle = {"seed": pair, "initial_sequence": "II", "features_blob": features.getvalue()}
        expected_arm = targets.full_initialization_arm(temperature)
        protocol.update(arm=expected_arm, aa_order=AA, sheet_pdf=q)
    with pytest.raises(ReachedDesignLoop):
        worker(
            SimpleNamespace(inversion_models={}, hf_critic_models={}, esmc_model=None),
            {},
            {},
            bundle,
            protocol,
            SimpleNamespace(commit=lambda: None),
        )
    assert received["arm"] == expected_arm
