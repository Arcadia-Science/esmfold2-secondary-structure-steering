"""Run the logging path on CPU, with model dependencies replaced by small functions."""

import __future__
import ast
import json
import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from analysis import loss_trace, schedule, targets


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


def test_arm_labels_reach_live_and_final_logs_without_changing_optimization(design_loop, tmp_path):
    records = []
    for label in [None, *targets.CAMPAIGN_ARMS]:
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
