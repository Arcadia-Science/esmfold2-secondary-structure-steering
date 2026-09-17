"""The gradient-guided binder design loop, and the driver that loads its models.

Verbatim from ``cookbook/tutorials/binder_design.py`` at upstream commit
``ba4d712``, lines 60-63 (module setup), 815-825
(``normalized_gradient_tensor``), 828-1056 (``design_binder``) and 1098-1176
(``ESMFold2Design``).

``design_binder`` runs in four phases, which upstream marked with inline
comments: setup, the optimization loop over ``STEPS`` on a cosine temperature
schedule, and scoring of the best sequence by every loaded critic.

``steps`` is a parameter defaulting to ``STEPS``. Upstream reads the module
global directly and this port used to as well, with callers rebinding
``analysis.design.loop.STEPS`` before calling. That worked only because Modal
runs one input per container and because the caller always passed a value, so
every call reassigned it; it also left ``analysis.design.constants.STEPS`` and
the re-export on :mod:`analysis.design` holding the original number while the
loop used another. A parameter has none of those hazards.

``design_binder`` is the one ported definition this repository extends rather
than reproduces. Four upstream statements change: the signature gains two
keyword parameters whose defaults reproduce upstream behaviour, the return
gains the loss log, the inversion-model draw is split in two so the chosen
replicate can be named in that log, and the structural loss gains the steering
total. Everything else is inserted. ``tests/test_design_port.py`` enumerates
that divergence statement by statement and fails if it grows.

Five things are added. Four sit inside the optimization step:

1. The secondary-structure steering terms from :mod:`analysis.steering`, at the
   weights :mod:`analysis.schedule` sets for that step, added to the structural
   loss before the gradient is taken. With the default ``VANILLA`` schedule
   every weight is zero and the objective is exactly the cookbook's.
2. The distogram ipTM proxy, computed every step rather than only on the
   confidence steps, so the interface cost of a steering weight is visible while
   it is being paid.
3. A per-step loss log, :class:`analysis.loss_trace.StepRecord`, which records
   every loss, every weight in force, the binding readout and the sequence
   entropy. Every registered steering term is measured every step, including
   the ones carrying no weight, because the claim that the capped barrier runs
   out of gradient early is a claim about unsteered runs.

A fifth is optional and off by default: when ``log_step_structures`` is set,
every step's predicted complex is built and kept as PDB text, so a run can be
watched as a trajectory rather than read as a curve. The coordinates are already
computed on every step, since the structure module has to run for the distogram
to exist, and the cookbook discards them until the critics. Pair it with
``sampling_steps`` raised to ``confidence_sampling_steps``, or the trajectory is
sampled one way for most of the run and another way at the end.

The next is in the critic scoring that follows the loop: per-chain pLDDT,
via :func:`analysis.confidence.chain_plddt`. The cookbook builds the complex
with per-atom pLDDT in its B-factor column but reports no per-chain number, and
the viability rule this publication uses needs binder pLDDT specifically,
because a disordered binder on a confident target still scores well on the
complex mean.

The log describes batch element 0 throughout, losses included, rather than
mixing per-element sequences with batch-mean losses. This publication runs
``batch_size=1``, where the distinction does not arise; at larger batch sizes
the log follows one design rather than summarising all of them. The in-memory
``trajectory`` is unaffected and still holds the full batch.
"""

from __future__ import annotations
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)

from analysis import confidence, loss_trace
from analysis.design.constants import (
    BYTES_PER_GIB,
    COMPILE,
    LEARNING_RATE,
    LM_LOSS_BATCH_SIZE,
    LM_MASK_PASSES,
    LOG_INTERVAL,
    REUSE_ESMC,
    STEPS,
    TEMPERATURE_MIN,
    Trajectory,
    TrajectoryStep,
)
from analysis.design.folding import fold_and_get_distogram
from analysis.design.losses import compute_structure_losses, get_mid_points
from analysis.design.metrics import _is_antibody_sequence, compute_distogram_iptm_proxy
from analysis.design.models import _apply_torch_compile, _load_hf_model
from analysis.design.plm import compute_esmc_pseudoperplexity_nll
from analysis.design.prompts import BINDER_PROMPT_FACTORIES, TARGET_SEQUENCES
from analysis.design.sequence import (
    build_gradient_mask,
    build_initial_soft_sequence_logits,
    sequence_to_one_hot,
)
from analysis.design.structure import build_complex
from analysis.schedule import VANILLA, LossSchedule
from analysis.steering import compute_steering_losses

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SAMPLING_STEPS = 1
"""Solver steps the structure module takes on an ordinary optimization step.

The cookbook takes one. A single step is enough to give the distogram the
gradient is read from, and it is the cheapest thing that does, which is what
matters when the step runs 150 times. Raising it buys a better sampled set of
coordinates rather than a better distogram, so it is only worth paying for when
the coordinates themselves are wanted.
"""

CONFIDENCE_SAMPLING_STEPS = 50
"""Solver steps on a confidence step, where a real ipTM is computed.

Confidence needs coordinates rather than a distogram, so the cookbook spends
fifty steps on the tail of the run where the temperature has collapsed and the
sequence has stopped moving. A trajectory logged across the whole run therefore
changes sampling budget partway through unless this and ``SAMPLING_STEPS`` are
set to the same value.
"""

TIMING_KEYS = ("time", "peak_allocated_gib", "peak_reserved_gib")


def normalized_gradient_tensor(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    """Rescale a masked gradient to unit norm per effective position.

    ``index_has_nonzero_grad`` is ``(B, L)``; ``eff_L`` and ``grad_norm`` are
    ``(B,)``.
    """
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0
    eff_L = index_has_nonzero_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_L[:, None, None]
    )
    return normalized_grad * gradient_mask


def design_binder(
    inversion_models: dict[str, ESMFold2ExperimentalModel],
    hf_critic_models: dict[str, ESMFold2ExperimentalModel],
    esmc_model: ESMCForMaskedLM,
    target_name: str,
    target_sequence: str | None,
    binder_name: str,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
    schedule: LossSchedule = VANILLA,
    trace_path: Path | None = None,
    steps: int = STEPS,
    sampling_steps: int = SAMPLING_STEPS,
    confidence_sampling_steps: int = CONFIDENCE_SAMPLING_STEPS,
    log_step_structures: bool = False,
) -> tuple[
    list[str],
    Trajectory,
    list[dict],
    tuple[loss_trace.StepRecord, ...],
    tuple[str, ...],
]:
    """
    Algorithm 11 Gradient-Guided Binder Sequence Optimization.

    Run the full optimization loop.
    Returns dict with designed_sequence, complex, and trajectory.

    Every critic is folded once on the best designed sequence via HF ESMFold2.
    Hero critics expose iPTM; scaling critics contribute distogram scores only.
    ``distogram_binding_confidence`` / ``cdr_distogram_binding_confidence`` come
    from the distogram in all cases.

    ``trajectory`` maps a step index to ``{loss_name: [B] tensor on CPU,
    metric_name: float}``.

    ``schedule`` sets the steering weight each term carries at each step, and
    defaults to no steering at all. ``trace_path`` is where the per-step loss
    log is appended as JSON lines; the log is returned either way, so a path is
    only needed when a run has to survive its own process.

    Only single binder and target chains are supported by default. To support
    the multi-chain case, remove the two ``"|" not in`` asserts below and check
    that the losses and the selection metrics are appropriate for it.
    """
    device = "cuda"
    if target_name in TARGET_SEQUENCES:
        if target_sequence is not None:
            raise ValueError(f"{target_name!r} is a preset target; omit target_sequence.")
        target_sequence = TARGET_SEQUENCES[target_name]
    elif target_sequence is None:
        raise ValueError(f"{target_name!r} is not a preset target; provide target_sequence.")
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    if binder_name in BINDER_PROMPT_FACTORIES:
        if binder_sequence is not None:
            raise ValueError(f"{binder_name!r} is a preset binder; omit binder_sequence.")
        binder_prompt_factor = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert binder_prompt_factor.is_antibody == is_antibody, (
                "Conflict in is_antibody settings."
            )
        is_antibody = binder_prompt_factor.is_antibody
        binder_sequence = binder_prompt_factor.sample(seed=seed)
    elif binder_sequence is None:
        raise ValueError(f"{binder_name!r} is not a preset binder; provide binder_sequence.")
    elif is_antibody is None:
        is_antibody = _is_antibody_sequence(binder_sequence)

    binder_length = len(binder_sequence)
    target_length = len(target_sequence.replace("|", ""))

    assert "|" not in target_sequence
    assert "|" not in binder_sequence

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(binder_sequence, batch_size=batch_size)
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)

    trajectory: Trajectory = {}
    trace: list[loss_trace.StepRecord] = []
    step_structures: list[str] = []
    bin_distance = get_mid_points()
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        optimizer.zero_grad()

        random.seed(seed + global_step)
        replicate_choice = random.randint(0, len(inversion_models) - 1)
        inversion_name = list(inversion_models)[replicate_choice]
        inversion_model = inversion_models[inversion_name]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=(
                confidence_sampling_steps if calculate_confidence else sampling_steps
            ),
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(fold_result["distogram_logits"], binder_length)
        weights = schedule.weights_at(global_step)
        steering_losses = compute_steering_losses(
            fold_result["distogram_logits"], binder_length, bin_distance, weights
        )
        structure_loss = losses["total_loss"] + steering_losses.total
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + global_step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=LM_LOSS_BATCH_SIZE,
                n_passes=LM_MASK_PASSES,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask) + (
            0.05 if is_antibody else 0.15
        ) * normalized_gradient_tensor(plm_grad, gradient_mask)

        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE * temperature

        optimizer.step()

        step = global_step
        step_losses: TrajectoryStep = {k: v.detach().cpu() for k, v in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        step_losses["structure_loss"] = structure_loss.detach().cpu()
        step_losses["steering_loss"] = steering_losses.total.detach().cpu()
        step_losses["time"] = time.time() - start
        step_losses["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / BYTES_PER_GIB
        step_losses["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / BYTES_PER_GIB
        trajectory[step] = step_losses
        loss_str = "  ".join(
            f"{k}={v.mean().item() if torch.is_tensor(v) else v:.4f}"
            for k, v in step_losses.items()
        )
        if step % LOG_INTERVAL == 0:
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}")

        binder_seq = sequences[0].split("|")[-1]
        with torch.no_grad():
            proxy = compute_distogram_iptm_proxy(
                fold_result["distogram_logits"], target_length, binder_seq, is_antibody
            )
        step_iptm = fold_result.get("iptm", None)
        logged_losses = {
            key: float(value.flatten()[0]) if torch.is_tensor(value) else float(value)
            for key, value in step_losses.items()
            if key not in TIMING_KEYS
        }
        logged_losses.update(
            {
                name: float(value.detach().flatten()[0])
                for name, value in steering_losses.terms.items()
            }
        )
        record = loss_trace.StepRecord(
            target=target_name,
            arm=schedule.key,
            seed=seed,
            step=step,
            steps=steps,
            temperature=temperature,
            confidence_step=calculate_confidence,
            inversion_model=inversion_name,
            binder_sequence=binder_seq,
            step_seconds=float(step_losses["time"]),
            sequence_entropy_bits=loss_trace.sequence_entropy_bits(design),
            distogram_iptm_proxy=proxy["distogram_iptm_proxy"],
            iptm=None if step_iptm is None else float(step_iptm[0]),
            losses=logged_losses,
            weights=weights,
        )
        trace.append(record)
        if trace_path is not None:
            loss_trace.append(record, trace_path)
        if log_step_structures:
            step_structures.append(
                build_complex(fold_result["inputs"], fold_result["output"]).to_pdb_string()
            )

        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(steps):
        t = (step + 1) / steps
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits,
            optimizer,
            temperature=temperature,
            calculate_confidence=temperature < 0.05,
        )
        if iptm is not None:
            for b in range(batch_size):
                if iptm[b] is not None and iptm[b] > best_iptm[b]:
                    best_iptm[b] = iptm[b]
                    best_sequences[b] = sequences[b]
    assert all(seq != "" for seq in best_sequences)

    critic_results: list[dict] = []
    final_total_loss = trajectory[global_step - 1]["total_loss"]
    assert isinstance(final_total_loss, torch.Tensor)
    for batch_idx in range(batch_size):
        best_seq = best_sequences[batch_idx]
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq)[..., 2:22]
        for critic_name, critic_model in hf_critic_models.items():
            is_scaling_critic = "ESMFold2-Experimental-Fast-base" in critic_name
            if is_scaling_critic:
                critic_model.cuda()
            final_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=seed,
            )
            if is_scaling_critic:
                critic_model.cpu()
            pred_complex = build_complex(final_fold["inputs"], final_fold["output"])
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"], target_length, binder_seq, is_antibody
            )
            iptm = final_fold["iptm"].item() if final_fold["iptm"] is not None else None
            chain_plddt = confidence.chain_plddt(
                final_fold["plddt"], target_length, len(binder_seq), batch_idx
            )
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "critic_name": critic_name,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_seq,
                    "complex": pred_complex,
                    "final_loss": final_total_loss[batch_idx].item(),
                    "iptm": iptm,
                    "plddt_complex": chain_plddt.complex_plddt,
                    "plddt_target": chain_plddt.target_plddt,
                    "plddt_binder": chain_plddt.binder_plddt,
                    "logits": logits[batch_idx].detach().cpu(),
                    **iptm_proxy_scores,
                }
            )

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_sequences[batch_idx],
                    "final_loss": final_total_loss[batch_idx].item(),
                    "logits": logits[batch_idx].detach().cpu(),
                }
            )

    return (
        best_sequences,
        trajectory,
        critic_results,
        tuple(trace),
        tuple(step_structures),
    )


class ESMFold2Design:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]
    scaling_critic_hf_paths: list[str] = []

    def load(self, use_scaling_critics: bool):
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            model_name: _load_hf_model(model_name, lm_dropout=0.5, cache_esmc=True, device="cuda")
            for model_name in self.inversion_model_names
        }
        if COMPILE:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        self.hf_critic_models: dict[str, Any] = {}
        for name in self.hero_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=True, device="cuda"
            )
        for name in self.scaling_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=False, device="cpu"
            )

        self.esmc_model = ESMCForMaskedLM.from_pretrained(self.lm_name, torch_dtype=torch.float32)
        if REUSE_ESMC:
            reusable_esmc_model = self.inversion_models["ESMFold2-Experimental-Fast"]
            assert reusable_esmc_model.config.esmc_id == self.lm_name, (
                f"Cannot reuse ESMC trunk from {reusable_esmc_model.config.esmc_id!r} "
                f"with LM head from {self.lm_name!r}."
            )
            assert reusable_esmc_model._esmc is not None
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = reusable_esmc_model._esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)

    def design(
        self,
        target_name: str,
        binder_name: str,
        target_sequence: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        schedule: LossSchedule = VANILLA,
        trace_path: Path | None = None,
        steps: int = STEPS,
        sampling_steps: int = SAMPLING_STEPS,
        confidence_sampling_steps: int = CONFIDENCE_SAMPLING_STEPS,
        log_step_structures: bool = False,
    ) -> tuple[
        list[str],
        Trajectory,
        list[dict],
        tuple[loss_trace.StepRecord, ...],
        tuple[str, ...],
    ]:
        return design_binder(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            schedule=schedule,
            trace_path=trace_path,
            steps=steps,
            sampling_steps=sampling_steps,
            confidence_sampling_steps=confidence_sampling_steps,
            log_step_structures=log_step_structures,
        )
