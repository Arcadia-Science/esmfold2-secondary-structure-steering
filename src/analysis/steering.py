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
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import torch

MIN_BINDER_LENGTH = 16

HELIX_PENALTY_OFFSET = 4
HELIX_PENALTY_CUTOFF_ANGSTROM = 7.0
HELIX_PENALTY_TOPK = 12
HELIX_LOG_EPS = 1e-3
HELIX_LOG_CAP = 2.0
HELIX_I2_OFFSET = 2
HELIX_I2_CUTOFF_ANGSTROM = 6.0
HELIX_I2_TOPK = 12

SHEET_CONTACT_CUTOFF_ANGSTROM = 5.5
SHEET_CONTACT_MIN_SEP = 4
SHEET_CONTACT_TOPK = 12
SHEET_CONTACT_EXT_CENTER = 6.0
SHEET_CONTACT_EXT_ALPHA = 5.0

SHEET_LADDER_CUTOFF_ANGSTROM = 5.5
SHEET_LADDER_MIN_SEP = 4
SHEET_LADDER_TOPK = 12
SHEET_LADDER_MIN_RUN = 3
SHEET_LADDER_EXT_CENTER = 6.0
SHEET_LADDER_EXT_ALPHA = 5.0

HELIX_FAMILY = "helix penalty"
SHEET_FAMILY = "sheet reward"

CLAMP_CONTACT_PROBABILITY = 1.0 - math.exp(-HELIX_LOG_CAP)


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


def expected_distance_matrix(
    distogram_logits: torch.Tensor, bin_distance: torch.Tensor
) -> torch.Tensor:
    """E[Ca-Ca distance] per residue pair, in Angstrom."""
    probs = torch.softmax(distogram_logits, dim=-1)
    return torch.matmul(probs, bin_distance.to(dtype=probs.dtype))


def strand_spacing_gate(
    expected_dist: torch.Tensor, ext_center: float, ext_alpha: float
) -> torch.Tensor:
    """Per-residue sigmoid on E[d(i, i+2)], one at strand-like local spacing.

    Helix i to i+2 is about 5.4 A and strand i to i+2 about 6.7 A, so a 6 A
    centre separates them. The last two positions have no i+2 partner and are
    padded with zero, which excludes them rather than guessing at them.
    """
    local = torch.diagonal(expected_dist, offset=2, dim1=-2, dim2=-1)
    gate = (ext_alpha * (local - ext_center)).sigmoid()
    return torch.nn.functional.pad(gate, (0, 2))


def helix_contact_scores(contact_prob: torch.Tensor, offset: int) -> torch.Tensor:
    """Per-residue i to i+offset contact probability, zero-padded to length."""
    diagonal = torch.diagonal(contact_prob, offset=offset, dim1=-2, dim2=-1)
    return torch.nn.functional.pad(diagonal, (0, offset))


def helix_penalty_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = HELIX_PENALTY_CUTOFF_ANGSTROM,
    offset: int = HELIX_PENALTY_OFFSET,
    topk: int = HELIX_PENALTY_TOPK,
) -> torch.Tensor:
    """Mean of the top-k i to i+4 contact probabilities.

    High when the binder holds a helical stretch, low when its i to i+4 pairs
    sit outside 7 A. One contact rather than a product of several, so the
    gradient is live from the first step.
    """
    contact_prob = contact_probability_matrix(distogram_logits, bin_distance, cutoff)
    binder = contact_prob[:, -binder_length:, -binder_length:]
    scores = helix_contact_scores(binder, offset)
    return torch.topk(scores, k=topk, dim=-1).values.mean(dim=-1)


def helix_log_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = HELIX_PENALTY_CUTOFF_ANGSTROM,
    offset: int = HELIX_PENALTY_OFFSET,
    topk: int = HELIX_PENALTY_TOPK,
    eps: float = HELIX_LOG_EPS,
) -> torch.Tensor:
    """``-log(1 - helix_penalty)``, a barrier rather than a linear penalty.

    The linear penalty is bounded by one, so at a small weight it loses to the
    contact and globularity terms once a helix has formed. This one keeps
    growing as those contacts saturate.
    """
    base = helix_penalty_loss(
        distogram_logits, binder_length, bin_distance, cutoff, offset, topk
    )
    return -torch.log((1.0 - base).clamp(min=eps))


def helix_logcap_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = HELIX_PENALTY_CUTOFF_ANGSTROM,
    offset: int = HELIX_PENALTY_OFFSET,
    topk: int = HELIX_PENALTY_TOPK,
    eps: float = HELIX_LOG_EPS,
    cap: float = HELIX_LOG_CAP,
) -> torch.Tensor:
    """``helix_log`` clamped at ``cap``, the term this publication settles on.

    The uncapped barrier at weight 0.2 pulled harder late than the linear
    penalty did at 0.5, and interface confidence fell with it. The clamp keeps
    the mid-range shape and drops the late punch.

    The clamp has a consequence worth stating plainly. Above a top-k contact
    probability of ``1 - exp(-cap)``, which is 0.8647 at the default cap, the
    term is constant, so it contributes no gradient at all.
    """
    raw = helix_log_loss(
        distogram_logits, binder_length, bin_distance, cutoff, offset, topk, eps
    )
    return raw.clamp(max=cap)


def helix_i2_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = HELIX_I2_CUTOFF_ANGSTROM,
    offset: int = HELIX_I2_OFFSET,
    topk: int = HELIX_I2_TOPK,
) -> torch.Tensor:
    """Mean of the top-k i to i+2 contact probabilities at 6 A.

    A different handle from the i to i+4 rung: this one pushes on the local
    spacing that sheets already have, rather than only punishing the helix
    contact itself.
    """
    return helix_penalty_loss(
        distogram_logits, binder_length, bin_distance, cutoff, offset, topk
    )


def sheet_contact_residue_scores(
    contact_prob: torch.Tensor,
    expected_dist: torch.Tensor,
    min_sep: int = SHEET_CONTACT_MIN_SEP,
    ext_center: float = SHEET_CONTACT_EXT_CENTER,
    ext_alpha: float = SHEET_CONTACT_EXT_ALPHA,
) -> torch.Tensor:
    """Per-residue sheet score: its best long-range contact, spacing-gated."""
    n = int(contact_prob.size(-1))
    index = torch.arange(n, device=contact_prob.device)
    separated = (torch.abs(index[:, None] - index[None, :]) >= min_sep).to(
        dtype=contact_prob.dtype
    )
    gate = strand_spacing_gate(expected_dist, ext_center, ext_alpha)
    pair = contact_prob * separated * gate.unsqueeze(-1) * gate.unsqueeze(-2)
    return pair.amax(dim=-1)


def sheet_contact_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = SHEET_CONTACT_CUTOFF_ANGSTROM,
    min_sep: int = SHEET_CONTACT_MIN_SEP,
    topk: int = SHEET_CONTACT_TOPK,
    ext_center: float = SHEET_CONTACT_EXT_CENTER,
    ext_alpha: float = SHEET_CONTACT_EXT_ALPHA,
) -> torch.Tensor:
    """``1 - mean(top-k gated 5.5 A contact scores)``.

    One pair per residue rather than a registered run, so the gradient is live
    from the first step. The i to i+2 gates keep a helix from scoring.
    """
    contact_prob = contact_probability_matrix(distogram_logits, bin_distance, cutoff)
    expected = expected_distance_matrix(distogram_logits, bin_distance)
    binder = contact_prob[:, -binder_length:, -binder_length:]
    binder_expected = expected[:, -binder_length:, -binder_length:]
    scores = sheet_contact_residue_scores(
        binder, binder_expected, min_sep, ext_center, ext_alpha
    )
    return 1.0 - torch.topk(scores, k=topk, dim=-1).values.mean(dim=-1)


def sheet_mean_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = SHEET_CONTACT_CUTOFF_ANGSTROM,
    min_sep: int = SHEET_CONTACT_MIN_SEP,
    ext_center: float = SHEET_CONTACT_EXT_CENTER,
    ext_alpha: float = SHEET_CONTACT_EXT_ALPHA,
) -> torch.Tensor:
    """``1 - mean(all gated 5.5 A residue scores)``.

    The same score as ``sheet_contact``, but every residue counts. A top-12
    selection can reach zero on one short ladder and then stop pushing.
    """
    contact_prob = contact_probability_matrix(distogram_logits, bin_distance, cutoff)
    expected = expected_distance_matrix(distogram_logits, bin_distance)
    binder = contact_prob[:, -binder_length:, -binder_length:]
    binder_expected = expected[:, -binder_length:, -binder_length:]
    scores = sheet_contact_residue_scores(
        binder, binder_expected, min_sep, ext_center, ext_alpha
    )
    return 1.0 - scores.mean(dim=-1)


def sheet_ladder_residue_scores(
    contact_prob: torch.Tensor,
    expected_dist: torch.Tensor,
    min_sep: int = SHEET_LADDER_MIN_SEP,
    min_run: int = SHEET_LADDER_MIN_RUN,
    ext_center: float = SHEET_LADDER_EXT_CENTER,
    ext_alpha: float = SHEET_LADDER_EXT_ALPHA,
) -> torch.Tensor:
    """Per-residue ladder strength in ``[0, 1]``.

    A run is the product of ``min_run`` consecutive registered contacts, either
    parallel, where the partner advances with the residue, or antiparallel,
    where it retreats. Every pair in a run must clear ``min_sep``, and the
    starting pair is gated on strand-like i to i+2 spacing on both sides. Each
    residue takes the strongest run it sits on, in either register and at any
    position along that run.
    """
    n = int(contact_prob.size(-1))
    index = torch.arange(n, device=contact_prob.device)
    separated = (torch.abs(index[:, None] - index[None, :]) >= min_sep).to(
        dtype=contact_prob.dtype
    )
    width = n - min_run + 1
    parallel = contact_prob.new_ones(contact_prob.shape[:-2] + (width, width))
    antiparallel = parallel
    for step in range(min_run):
        rows = slice(step, step + width)
        parallel = parallel * contact_prob[:, rows, rows] * separated[rows, rows]
        columns = slice(min_run - 1 - step, min_run - 1 - step + width)
        antiparallel = (
            antiparallel * contact_prob[:, rows, columns] * separated[rows, columns]
        )

    gate = strand_spacing_gate(expected_dist, ext_center, ext_alpha)
    parallel = parallel * gate[:, :width].unsqueeze(-1) * gate[:, :width].unsqueeze(-2)
    antiparallel = (
        antiparallel
        * gate[:, :width].unsqueeze(-1)
        * gate[:, min_run - 1 :].unsqueeze(-2)
    )

    def placed(values: torch.Tensor, start: int) -> torch.Tensor:
        return torch.nn.functional.pad(values, (start, n - start - values.shape[-1]))

    runs = [placed(parallel.amax(dim=-1), offset) for offset in range(min_run)]
    runs.extend(placed(parallel.amax(dim=-2), offset) for offset in range(min_run))
    runs.extend(placed(antiparallel.amax(dim=-1), offset) for offset in range(min_run))
    runs.extend(placed(antiparallel.amax(dim=-2), offset) for offset in range(min_run))
    return torch.stack(runs, dim=0).amax(dim=0)


def sheet_ladder_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    cutoff: float = SHEET_LADDER_CUTOFF_ANGSTROM,
    min_sep: int = SHEET_LADDER_MIN_SEP,
    topk: int = SHEET_LADDER_TOPK,
    min_run: int = SHEET_LADDER_MIN_RUN,
    ext_center: float = SHEET_LADDER_EXT_CENTER,
    ext_alpha: float = SHEET_LADDER_EXT_ALPHA,
) -> torch.Tensor:
    """``1 - mean(top-k residue ladder scores)``.

    High when the binder holds no registered strand pair with strand-like local
    spacing. The three-step product is the strictest of the sheet terms and in
    practice the least trainable: early in a run every factor is small, so the
    product and its gradient are both near zero.
    """
    contact_prob = contact_probability_matrix(distogram_logits, bin_distance, cutoff)
    expected = expected_distance_matrix(distogram_logits, bin_distance)
    binder = contact_prob[:, -binder_length:, -binder_length:]
    binder_expected = expected[:, -binder_length:, -binder_length:]
    scores = sheet_ladder_residue_scores(
        binder, binder_expected, min_sep, min_run, ext_center, ext_alpha
    )
    return 1.0 - torch.topk(scores, k=topk, dim=-1).values.mean(dim=-1)


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
    "helix_penalty": SteeringTermSpec(
        key="helix_penalty",
        label="Helix penalty",
        family=HELIX_FAMILY,
        unit="0-1",
        measures=(
            "mean of the top-12 binder i to i+4 contact probabilities inside 7 A, "
            "the rung spacing of an alpha helix"
        ),
        compute=helix_penalty_loss,
    ),
    "helix_log": SteeringTermSpec(
        key="helix_log",
        label="Helix log barrier",
        family=HELIX_FAMILY,
        unit="nats",
        measures=(
            "-log(1 - helix_penalty), which keeps growing as those contacts "
            "saturate, so a small weight can still fight a formed helix"
        ),
        compute=helix_log_loss,
    ),
    "helix_logcap": SteeringTermSpec(
        key="helix_logcap",
        label="Capped helix log barrier",
        family=HELIX_FAMILY,
        unit="nats",
        measures=(
            "helix_log clamped at 2.0, which keeps the mid-range shape without "
            "the late-stage punch that cost interface confidence"
        ),
        compute=helix_logcap_loss,
    ),
    "helix_i2": SteeringTermSpec(
        key="helix_i2",
        label="Helix i to i+2 penalty",
        family=HELIX_FAMILY,
        unit="0-1",
        measures=(
            "mean of the top-12 binder i to i+2 contact probabilities inside 6 A, "
            "a handle on local spacing rather than on the helix rung"
        ),
        compute=helix_i2_loss,
    ),
    "sheet_contact": SteeringTermSpec(
        key="sheet_contact",
        label="Sheet contact reward",
        family=SHEET_FAMILY,
        unit="0-1",
        measures=(
            "one minus the mean of the top-12 gated long-range 5.5 A contact "
            "scores, one pair per residue rather than a registered run"
        ),
        compute=sheet_contact_loss,
    ),
    "sheet_mean": SteeringTermSpec(
        key="sheet_mean",
        label="Sheet contact reward, all residues",
        family=SHEET_FAMILY,
        unit="0-1",
        measures=(
            "the same gated contact score averaged over every binder residue, so "
            "a short ladder cannot finish the term and stop pushing"
        ),
        compute=sheet_mean_loss,
    ),
    "sheet_ladder": SteeringTermSpec(
        key="sheet_ladder",
        label="Sheet ladder reward",
        family=SHEET_FAMILY,
        unit="0-1",
        measures=(
            "one minus the mean of the top-12 three-step registered ladder "
            "scores, the strictest of the sheet terms and the hardest to train"
        ),
        compute=sheet_ladder_loss,
    ),
}

TERM_KEYS: tuple[str, ...] = tuple(TERM_SPECS)


def term(key: str) -> SteeringTermSpec:
    """Look up one steering term, naming the valid keys on failure."""
    if key not in TERM_SPECS:
        raise KeyError(
            f"unknown steering term {key!r}; registered: {sorted(TERM_SPECS)}"
        )
    return TERM_SPECS[key]


def term_table() -> pd.DataFrame:
    """Every registered steering term, one row each."""
    return pd.DataFrame(
        [
            {
                "term": spec.key,
                "family": spec.family,
                "unit": spec.unit,
                "what it measures": spec.measures,
            }
            for spec in TERM_SPECS.values()
        ]
    )


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
            f"unknown steering term(s) {sorted(unknown)}; "
            f"registered: {sorted(TERM_SPECS)}"
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

    bin_distance = bin_distance.to(
        device=distogram_logits.device, dtype=distogram_logits.dtype
    )
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
