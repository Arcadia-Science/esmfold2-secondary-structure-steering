"""The design targets, and the loss arms run against them.

All five targets are the constructs used in the ESMFold2 preprint, taken
verbatim from ``TARGET_SEQUENCES`` in the ESM cookbook binder-design tutorial.
Every one is an exact substring of its UniProt entry, and the
``construct_uniprot`` range below records where.

This module imports nothing from Modal, so the notebook can read target
metadata without a Modal account.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from analysis import schedule
from analysis.paths import TARGETS
from analysis.schedule import LossSchedule


@dataclass(frozen=True)
class TargetSpec:
    """One design target.

    ``sequence_fasta`` is the cropped construct handed to sequence-only design.
    ``structure_cif`` plus ``chain`` and ``auth_crop`` locate the same residues
    in deposited coordinates, which is what the distogram-conditioned mode and
    the secondary-structure ground truth both need. ``auth_crop`` is in author
    numbering, which differs from UniProt numbering for 1YY9 and 3OSK.
    """

    key: str
    protein: str
    uniprot: str
    construct_uniprot: str
    construct_length: int
    sequence_fasta: str
    structure_cif: str
    chain: str
    register_fasta: str
    auth_crop: str
    resolution_angstrom: float
    fold: str

    @property
    def sequence_path(self) -> Path:
        return TARGETS / self.sequence_fasta

    @property
    def structure_path(self) -> Path:
        return TARGETS / self.structure_cif

    @property
    def register_path(self) -> Path:
        return TARGETS / self.register_fasta

    @property
    def crop_start(self) -> int:
        return int(self.auth_crop.split("-")[0])

    @property
    def crop_end(self) -> int:
        return int(self.auth_crop.split("-")[1])

    def sequence(self) -> str:
        lines = self.sequence_path.read_text().splitlines()
        return "".join(line.strip() for line in lines if not line.startswith(">"))


TARGET_SPECS: dict[str, TargetSpec] = {
    "egfr_d3": TargetSpec(
        key="egfr_d3",
        protein="Human EGFR domain III",
        uniprot="P00533",
        construct_uniprot="334-524",
        construct_length=191,
        sequence_fasta="egfr_d3.fasta",
        structure_cif="1yy9.cif",
        chain="A",
        register_fasta="1yy9_chainA_auth2-614.fasta",
        auth_crop="310-473",
        resolution_angstrom=2.61,
        fold="leucine-rich, loop-dominated",
    ),
    "pdgfrb_d2d3": TargetSpec(
        key="pdgfrb_d2d3",
        protein="Human PDGFRB Ig-C2 domains 2-3",
        uniprot="P09619",
        construct_uniprot="126-312",
        construct_length=187,
        sequence_fasta="pdgfrb_d2d3.fasta",
        structure_cif="3mjg.cif",
        chain="X",
        register_fasta="3mjg_chainX_auth33-312.fasta",
        auth_crop="126-312",
        resolution_angstrom=2.30,
        fold="Ig-like C2 tandem, all-beta",
    ),
    "pdl1_igv": TargetSpec(
        key="pdl1_igv",
        protein="Human PD-L1 IgV domain",
        uniprot="Q9NZQ7",
        construct_uniprot="18-132",
        construct_length=115,
        sequence_fasta="pdl1_igv.fasta",
        structure_cif="4zqk.cif",
        chain="A",
        register_fasta="4zqk_chainA_auth18-132.fasta",
        auth_crop="18-132",
        resolution_angstrom=2.45,
        fold="IgV, all-beta",
    ),
    "ctla4_igv": TargetSpec(
        key="ctla4_igv",
        protein="Human CTLA-4 IgV domain",
        uniprot="P16410",
        construct_uniprot="38-155",
        construct_length=118,
        sequence_fasta="ctla4_igv.fasta",
        structure_cif="3osk.cif",
        chain="A",
        register_fasta="3osk_chainA_auth3-123.fasta",
        auth_crop="3-120",
        resolution_angstrom=1.80,
        fold="IgV, all-beta",
    ),
    "cd45_d3d4": TargetSpec(
        key="cd45_d3d4",
        protein="Human CD45 FN3 domains d3-d4",
        uniprot="P08575",
        construct_uniprot="390-574",
        construct_length=185,
        sequence_fasta="cd45_d3d4.fasta",
        structure_cif="5fmv.cif",
        chain="A",
        register_fasta="5fmv_chainA_auth224-574.fasta",
        auth_crop="388-571",
        resolution_angstrom=2.90,
        fold="fibronectin type-III tandem, all-beta",
    ),
}

DEFAULT_TARGET = "pdl1_igv"
"""The target a bare smoke test runs against.

The shortest construct of the five at 115 residues, so it is the cheapest way
to check that a deployment works end to end. It carries no other significance:
every arm runs against every target.
"""


@dataclass(frozen=True)
class ArmSpec:
    """One loss configuration, applied identically across targets.

    An arm is a weight schedule over the steering terms in
    :mod:`analysis.steering`, and nothing else: the same design loop, the same
    step count, the same seeds, the same targets. That is what makes the
    comparison between arms a comparison of objectives.
    """

    key: str
    label: str
    schedule: LossSchedule
    note: str


ARM_SPECS: dict[str, ArmSpec] = {
    "vanilla": ArmSpec(
        key="vanilla",
        label="Vanilla",
        schedule=schedule.VANILLA,
        note=(
            "The cookbook objective untouched. Every steering term registered "
            "when the campaign ran is still measured and logged, which is how "
            "the reference curve for a steered arm is built. Terms added to "
            "the registry afterwards are absent rather than zero: these "
            "campaigns predate helix_mean and carry no column for it, so a "
            "per-step panel of that term shows the steered arms diverging from "
            "their shared step-zero value instead of from an unsteered curve."
        ),
    ),
    "helix_logcap_w02": ArmSpec(
        key="helix_logcap_w02",
        label="Capped helix barrier at 0.2",
        schedule=schedule.constant("helix_logcap_w02", "helix_logcap", 0.2),
        note=(
            "One weight, held for the whole run. The simplest thing to try, and "
            "the arm every other steering result is measured against."
        ),
    ),
    "helix_logcap_w05": ArmSpec(
        key="helix_logcap_w05",
        label="Capped helix barrier at 0.5",
        schedule=schedule.constant("helix_logcap_w05", "helix_logcap", 0.5),
        note=(
            "The control that separates timing from magnitude. The burst arm "
            "raises the weight early and lowers it late, so on its own it "
            "cannot say which of the two did the work. This arm holds the "
            "burst's opening weight for the whole run."
        ),
    ),
    "helix_mean_w02": ArmSpec(
        key="helix_mean_w02",
        label="Mean helix rung fraction at 0.2",
        schedule=schedule.constant("helix_mean_w02", "helix_mean", 0.2),
        note=(
            "The averaged term at the capped barrier's own weight, so the only "
            "thing that differs from helix_logcap_w02 is how the per-residue "
            "scores are pooled."
        ),
    ),
    "helix_mean_w05": ArmSpec(
        key="helix_mean_w05",
        label="Mean helix rung fraction at 0.5",
        schedule=schedule.constant("helix_mean_w05", "helix_mean", 0.5),
        note=(
            "The averaged term at the higher of the two weights the capped "
            "barrier was run at, again matched so that pooling is the only "
            "difference."
        ),
    ),
    "helix_mean_w10": ArmSpec(
        key="helix_mean_w10",
        label="Mean helix rung fraction at 1.0",
        schedule=schedule.constant("helix_mean_w10", "helix_mean", 1.0),
        note=(
            "Fills the gap below the first rung that worked. On PD-L1 the "
            "averaged term left a median helix fraction of 0.779 at weight 0.5 "
            "and 0.017 at 2.0, so the whole transition falls between them and "
            "neither bound locates it."
        ),
    ),
    "helix_mean_w20": ArmSpec(
        key="helix_mean_w20",
        label="Mean helix rung fraction at 2.0",
        schedule=schedule.constant("helix_mean_w20", "helix_mean", 2.0),
        note=(
            "The averaged term at a weight chosen to match the capped "
            "barrier's integrated push rather than its nominal weight. Where "
            "the barrier still has a gradient it is 2 to 11 times the averaged "
            "term's, and it carried that gradient on 58% of steps at weight 0.2 "
            "and 95% at 0.5, so a matched weight is several times higher. This "
            "arm brackets the two matched-weight arms so that a weak result "
            "there can be read as a weight rather than as the pooling."
        ),
    ),
    "helix_mean_w30": ArmSpec(
        key="helix_mean_w30",
        label="Mean helix rung fraction at 3.0",
        schedule=schedule.constant("helix_mean_w30", "helix_mean", 3.0),
        note=(
            "Between the two rungs that each matched the capped barrier on one "
            "axis and missed on the other. At 2.0 the averaged term matched the "
            "barrier's confidence cost but left residual helix in 11 of 30 "
            "designs; at 5.0 it matched the helix distribution and cost 0.061 "
            "of ipTM. This weight asks whether a single one does both."
        ),
    ),
    "helix_mean_w50": ArmSpec(
        key="helix_mean_w50",
        label="Mean helix rung fraction at 5.0",
        schedule=schedule.constant("helix_mean_w50", "helix_mean", 5.0),
        note=(
            "Second rung of the weight ladder. At 0.5 the averaged term left a "
            "median helix fraction of 0.779 on PD-L1 against the capped "
            "barrier's 0.000, and no design reached any sheet, so the ladder "
            "asks how far the weight has to rise before the averaged term "
            "reproduces that collapse."
        ),
    ),
    "helix_mean_w100": ArmSpec(
        key="helix_mean_w100",
        label="Mean helix rung fraction at 10.0",
        schedule=schedule.constant("helix_mean_w100", "helix_mean", 10.0),
        note=(
            "Top rung of the weight ladder, an order of magnitude above the "
            "barrier's own weights. The averaged term spreads a constant "
            "per-rung derivative of one over the number of scored residues "
            "across every rung, where the barrier concentrates a larger one on "
            "the twelve most formed. If helix survives even here the pooling "
            "is doing work that no weight recovers."
        ),
    ),
    "helix_logcap_burst20_hold01": ArmSpec(
        key="helix_logcap_burst20_hold01",
        label="Capped helix barrier, burst then hold",
        schedule=schedule.burst(
            "helix_logcap_burst20_hold01",
            "helix_logcap",
            burst_weight=0.5,
            hold_weight=0.1,
            burst_steps=20,
        ),
        note=(
            "0.5 while secondary structure is still forming, then 0.1 to hold "
            "it. The capped barrier stops producing gradient above a top-12 "
            "contact probability of 0.8647, which unsteered runs reach within a "
            "few steps, so this arm asks whether the weight has to arrive early "
            "to do anything at all."
        ),
    ),
}

CAMPAIGN_ARMS: tuple[str, ...] = (
    "vanilla",
    "helix_mean_w05",
    "helix_mean_w10",
    "helix_mean_w20",
    "helix_mean_w50",
)
"""The arms run across every target: unsteered, then the weight scan.

The ``helix_logcap`` arms are registered and their campaigns are still on disk,
but they sit outside this grid. That term pools the twelve highest i to i+4
contact probabilities, passes the pooled value through a log barrier and then
clamps it, which needs a paragraph to state and stops separating designs once
a dozen rungs have formed. The averaged term is a plain mean over the same
probabilities, reaches the same fold at a higher weight, and unlike the barrier
its weight sets how much helix survives rather than only whether any does.

``helix_logcap_burst20_hold01`` is outside the grid for a second reason: it was
built to test whether a steering weight has to arrive early, and the constant
0.5 control showed that it does not.
"""

N_SEEDS = 30
SGD_STEPS = 150


def target(key: str) -> TargetSpec:
    """Look up one target, with a message naming the valid keys on failure."""
    if key not in TARGET_SPECS:
        raise KeyError(f"unknown target {key!r}; registered: {sorted(TARGET_SPECS)}")
    return TARGET_SPECS[key]


def arm(key: str) -> ArmSpec:
    """Look up one loss arm, with a message naming the valid keys on failure."""
    if key not in ARM_SPECS:
        raise KeyError(f"unknown arm {key!r}; registered: {sorted(ARM_SPECS)}")
    return ARM_SPECS[key]
