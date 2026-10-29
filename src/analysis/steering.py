"""Differentiable secondary-structure proxies from binder distogram logits.

Each term returns one scalar per batch element. Schedules set their weights;
DSSP composition is measured separately from saved coordinates.
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass

import torch

HELIX_PENALTY_OFFSET = 4
HELIX_PENALTY_CUTOFF_ANGSTROM = 7.0


HELIX_FAMILY = "helix penalty"


def contact_probability_matrix(
    distogram_logits: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """P(Ca-Ca distance < cutoff) per residue pair, summed over the near bins."""
    contact_bins = bin_distance < cutoff
    if not bool(contact_bins.any()):
        raise ValueError(f"cutoff {cutoff} A excludes every distogram bin")
    probs = torch.softmax(distogram_logits, dim=-1)
    return probs[..., contact_bins].sum(dim=-1)


def helix_mean_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = HELIX_PENALTY_CUTOFF_ANGSTROM,
    offset: int = HELIX_PENALTY_OFFSET,
) -> torch.Tensor:
    """Mean contact probability between binder residues separated by ``offset``.

    Use bins with midpoints below ``cutoff``. At the defaults, this is an i to i+4
    helix proxy on a 0–1 scale. Exclude trailing residues without a partner.
    """
    contact_prob = contact_probability_matrix(distogram_logits, bin_distance, cutoff)
    binder = contact_prob[:, -binder_length:, -binder_length:]
    diagonal = torch.diagonal(binder, offset=offset, dim1=-2, dim2=-1)
    return diagonal.mean(dim=-1)


SteeringCompute = Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SteeringTermSpec:
    """One steering term: its name, family, unit, what it reads, and how."""

    key: str
    label: str
    family: str
    unit: str
    measures: str
    compute: SteeringCompute


TERM_SPECS: dict[str, SteeringTermSpec] = {
    "helix_mean": SteeringTermSpec(
        key="helix_mean",
        label="Mean helix rung fraction",
        family=HELIX_FAMILY,
        unit="0-1",
        measures=(
            "mean binder i to i+4 contact probability inside 7 A over every "
            "scored residue, so the loss is the expected helical-rung fraction "
            "and stays monotone in helix content where the top-12 pooling "
            "saturates"
        ),
        compute=helix_mean_loss,
    ),
}


def term(key: str) -> SteeringTermSpec:
    """Look up one steering term."""
    return TERM_SPECS[key]


@dataclass(frozen=True)
class SteeringLosses:
    """All registered terms, their configured weights, and the weighted total."""

    terms: dict[str, torch.Tensor]
    weights: dict[str, float]
    total: torch.Tensor


def compute_steering_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    weights: dict[str, float],
) -> SteeringLosses:
    """Every steering term, plus the weighted total of those with a weight.

    A term with zero weight is still measured, under ``no_grad`` so it costs no
    backward pass and adds nothing to the graph.
    """
    unknown = set(weights) - set(TERM_SPECS)
    if unknown:
        raise KeyError(
            f"unknown steering term(s) {sorted(unknown)}; registered: {sorted(TERM_SPECS)}"
        )
    if distogram_logits.size(1) < binder_length:
        raise ValueError("Distogram covers fewer than the requested binder residues")

    bin_distance = bin_distance.to(device=distogram_logits.device, dtype=distogram_logits.dtype)
    terms: dict[str, torch.Tensor] = {}
    total = torch.zeros(distogram_logits.size(0), device=distogram_logits.device)
    for key, spec in TERM_SPECS.items():
        weight = weights.get(key, 0.0)
        if weight == 0.0:
            with torch.no_grad():
                terms[key] = spec.compute(distogram_logits, binder_length, bin_distance)
            continue
        value = spec.compute(distogram_logits, binder_length, bin_distance)
        terms[key] = value
        total = total + weight * value
    return SteeringLosses(terms=terms, weights=dict(weights), total=total)
