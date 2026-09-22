"""The secondary-structure steering loss terms.

This is the publication's own contribution, layered on the ported cookbook
losses in :mod:`analysis.design.losses`. The cookbook's three structural terms
are intra-binder contact, binder-target contact, and globularity. None of them
says anything about secondary structure, which is why unsteered designs come
out helix-rich: nothing in the objective asks for anything else.

Every term reads the same object, the ESMFold2 distogram over Ca-Ca distances
as 128 binned logits per residue pair, and returns one scalar per batch element
that is differentiable back to the sequence logits. Every term is scoped to the
binder block of the distogram, so no term can reshape the target.

Two geometric facts do most of the work. In an alpha helix the i to i+4 Ca
distance is about 6.2 A and the i to i+2 distance about 5.4 A. In a beta strand
the i to i+2 distance is about 6.7 A, and the pairing partner sits 4.8 to 5.5 A
away at a sequence separation that is long rather than local. So a helix reads
as a short-range contact ladder, and a sheet as a long-range one with wider
local spacing. The sheet terms use the i to i+2 spacing as a gate, and that
gate is what stops a helix from scoring as a sheet.

This module needs torch but not ``esm`` and not a GPU, which is deliberate:
the shape of a loss term is a claim the notebook can check on its own, without
the design stack behind it. ``bin_distance`` is passed in rather than derived
for the same reason. The design loop supplies
:func:`analysis.design.losses.get_mid_points`, the cookbook's own bin layout.

When each term applies is a separate question, answered by
:mod:`analysis.schedule`.

Provenance: these terms were developed in a separate private pipeline and are
transcribed here so this repository holds one implementation of the whole
method. ``tests/test_steering_port.py`` checks that transcription numerically
against that source when it is available.
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass

import torch

MIN_BINDER_LENGTH = 16

HELIX_PENALTY_OFFSET = 4
HELIX_PENALTY_CUTOFF_ANGSTROM = 7.0
HELIX_PENALTY_TOPK = 12


HELIX_FAMILY = "helix penalty"


def contact_probability_matrix(
    distogram_logits: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """P(Ca-Ca distance < cutoff) per residue pair, summed over the near bins."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
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
    """Mean i to i+4 contact probability over every scored binder residue.

    The same per-residue quantity ``helix_penalty`` pools, but pooled over the
    whole binder instead of its twelve largest values. The loss is then the
    expected fraction of binder residues sitting on a helical rung, which is
    the quantity this publication reports, so the objective and the readout
    describe the same thing.

    ``helix_penalty`` cannot rank helical designs. Its top-12 mean reaches one
    once roughly two dozen rungs saturate, and measured across the design grid
    it takes the value 1.000 on every design above 20% helix regardless of
    whether that design is a fifth helical or entirely so. Composed with the
    clamp in ``helix_logcap`` this leaves no gradient at all on 97% of designs
    carrying at least twelve helical residues. Averaging over every residue
    removes both ceilings: the score is monotone in helix content across the
    whole range, and every residue contributes the same constant derivative
    rather than only the twelve that happen to be largest this step.

    No log barrier and no clamp. The barrier existed to keep pushing as the
    top-12 mean saturated near one, and this statistic does not: it tops out
    near 0.85 on a fully helical binder. The cost of that is a smaller gradient
    norm than the barrier supplied while it was live, which is a weight
    question rather than a shape question. ``analysis.schedule`` sets the
    weight.

    The trailing ``offset`` residues have no i to i+4 partner and are excluded
    from the mean rather than counted as non-helical, so the score does not
    drift with binder length.
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
    """Look up one steering term, naming the valid keys on failure."""
    if key not in TERM_SPECS:
        raise KeyError(f"unknown steering term {key!r}; registered: {sorted(TERM_SPECS)}")
    return TERM_SPECS[key]


@dataclass(frozen=True)
class SteeringLosses:
    """Every steering term at one step, and the weighted sum of the active ones.

    ``terms`` holds every registered term, including those carrying zero
    weight. Those are computed under ``no_grad`` and exist to be logged: the
    claim that the capped barrier runs out of gradient early is a claim about
    unsteered runs, so the term has to be measured even where it is not applied.
    """

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
    if binder_length < MIN_BINDER_LENGTH:
        raise ValueError(
            f"binder_length {binder_length} is below MIN_BINDER_LENGTH "
            f"{MIN_BINDER_LENGTH}; the top-{HELIX_PENALTY_TOPK} selections need "
            f"at least that many scored residues"
        )
    if distogram_logits.size(1) < binder_length:
        raise ValueError(
            f"distogram covers {distogram_logits.size(1)} residues, fewer than "
            f"the binder_length {binder_length}"
        )

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
