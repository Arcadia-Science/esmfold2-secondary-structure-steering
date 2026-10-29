"""Gradient-guided binder design with steering, critic scoring, and diagnostics."""

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
"""Default solver steps for optimization folds without confidence scoring."""

CONFIDENCE_SAMPLING_STEPS = 50
"""Default solver steps for optimization folds with confidence scoring."""

TIMING_KEYS = ("time", "peak_allocated_gib", "peak_reserved_gib")


def normalized_gradient_tensor(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    """Scale each masked batch gradient to approximately sqrt(effective positions) norm.

    An effective position has a nonzero masked gradient.
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
    stop_after_steps: int | None = None,
    snapshot_sampling_steps: int | None = None,
) -> tuple[
    list[str],
    Trajectory,
    list[dict],
    tuple[loss_trace.StepRecord, ...],
    tuple[str, ...],
]:
    """Optimize one binder chain against one target chain on CUDA.

    Return sequences, the batch loss trajectory, critic records, the loss trace,
    and optional per-step PDB strings, in that order. The loss trace and PDB
    snapshots describe batch element 0; the trajectory retains the full batch.

    ``steps`` sets the annealing horizon. A shorter ``stop_after_steps`` returns
    the last folded sequences and skips critic evaluation. Otherwise, select
    sequences by the highest optimization-step ipTM and evaluate each critic.

    ``schedule`` controls steering and base structural loss timing. Append trace
    rows to ``trace_path`` when supplied. ``log_step_structures`` enables PDB
    snapshots; ``snapshot_sampling_steps`` resamples saved soft states after
    optimization. Otherwise snapshots use the optimization fold's coordinates.
    Ordinary and confidence folds use their respective sampling-step budgets.
    """
    # Truncate execution without compressing the annealing schedule.
    run_steps = steps if stop_after_steps is None else stop_after_steps
    if snapshot_sampling_steps is not None and not log_step_structures:
        raise ValueError("snapshot_sampling_steps requires structure logging")
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
    snapshot_states: list[tuple[str, int, torch.Tensor]] = []
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
        structure_loss = schedule.structural_objective(
            losses["total_loss"], steering_losses.total, global_step
        )
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
        step_losses["base_structure_loss"] = losses["total_loss"].detach().cpu()
        step_losses["base_structure_weight"] = schedule.base_structure_weight_at(global_step)
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
            if snapshot_sampling_steps is not None:
                snapshot_states.append((inversion_name, seed + step, design.detach().cpu()))
            else:
                step_structures.append(
                    build_complex(fold_result["inputs"], fold_result["output"]).to_pdb_string()
                )

        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(run_steps):
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
    # Read out saved soft states only after every optimization update is done.
    # Sampling these coordinates cannot feed back into the trajectory.
    for inversion_name, snapshot_seed, soft_design in snapshot_states:
        with torch.no_grad():
            sampled = fold_and_get_distogram(
                inversion_models[inversion_name],
                target_sequence,
                target_one_hot,
                soft_design.to(device),
                num_loops=1,
                num_sampling_steps=snapshot_sampling_steps,
                calculate_confidence=False,
                seed=snapshot_seed,
            )
            step_structures.append(
                build_complex(sampled["inputs"], sampled["output"]).to_pdb_string()
            )
        del sampled
    if run_steps < steps:
        # Early trajectories have no confidence-based sequence selection yet.
        # Return their optimization snapshots without final critic evaluation.
        return sequences, trajectory, [], tuple(trace), tuple(step_structures)
    assert all(seq != "" for seq in best_sequences)

    critic_results: list[dict] = []
    final_total_loss = trajectory[global_step - 1]["total_loss"]
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
        stop_after_steps: int | None = None,
        snapshot_sampling_steps: int | None = None,
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
            stop_after_steps=stop_after_steps,
            snapshot_sampling_steps=snapshot_sampling_steps,
        )
