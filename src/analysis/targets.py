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
    "helix_mean_w20_from50": ArmSpec(
        key="helix_mean_w20_from50",
        label="Mean helix rung fraction at 2.0 from step 50",
        schedule=schedule.delayed("helix_mean_w20_from50", "helix_mean", 2.0, 50),
        note=(
            "The same weight as helix_mean_w20, withheld until step 50. Its "
            "first fifty steps are the unsteered objective and are identical "
            "to the unsteered arm on the same seed, so the arm asks whether a "
            "design the unsteered run has already carried into the alpha-rich "
            "basin can still be steered out of it, or whether the weight only "
            "works on a sequence that has not yet committed. Outside the grid: "
            "one target, one seed, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until1": ArmSpec(
        key="helix_mean_w20_until1",
        label="Mean helix rung fraction at 2.0 until step 1",
        schedule=schedule.released("helix_mean_w20_until1", "helix_mean", 2.0, 1),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until2": ArmSpec(
        key="helix_mean_w20_until2",
        label="Mean helix rung fraction at 2.0 until step 2",
        schedule=schedule.released("helix_mean_w20_until2", "helix_mean", 2.0, 2),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until4": ArmSpec(
        key="helix_mean_w20_until4",
        label="Mean helix rung fraction at 2.0 until step 4",
        schedule=schedule.released("helix_mean_w20_until4", "helix_mean", 2.0, 4),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until7": ArmSpec(
        key="helix_mean_w20_until7",
        label="Mean helix rung fraction at 2.0 until step 7",
        schedule=schedule.released("helix_mean_w20_until7", "helix_mean", 2.0, 7),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until10": ArmSpec(
        key="helix_mean_w20_until10",
        label="Mean helix rung fraction at 2.0 until step 10",
        schedule=schedule.released("helix_mean_w20_until10", "helix_mean", 2.0, 10),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until20": ArmSpec(
        key="helix_mean_w20_until20",
        label="Mean helix rung fraction at 2.0 until step 20",
        schedule=schedule.released("helix_mean_w20_until20", "helix_mean", 2.0, 20),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until30": ArmSpec(
        key="helix_mean_w20_until30",
        label="Mean helix rung fraction at 2.0 until step 30",
        schedule=schedule.released("helix_mean_w20_until30", "helix_mean", 2.0, 30),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until40": ArmSpec(
        key="helix_mean_w20_until40",
        label="Mean helix rung fraction at 2.0 until step 40",
        schedule=schedule.released("helix_mean_w20_until40", "helix_mean", 2.0, 40),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_until50": ArmSpec(
        key="helix_mean_w20_until50",
        label="Mean helix rung fraction at 2.0 until step 50",
        schedule=schedule.released("helix_mean_w20_until50", "helix_mean", 2.0, 50),
        note=(
            "One release in the scan that holds the weight at 2.0 from step 0 "
            "and moves only the step it is withdrawn. The delayed arms show "
            "the fold committing inside the first few steps; these ask whether "
            "the commitment the weight makes in those same steps is equally "
            "durable once the weight is gone. Released at step 0 is the "
            "unsteered arm and never released is helix_mean_w20, so both ends "
            "of this scan are campaigns that already exist. Outside the grid: "
            "one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_from1": ArmSpec(
        key="helix_mean_w20_from1",
        label="Mean helix rung fraction at 2.0 from step 1",
        schedule=schedule.delayed("helix_mean_w20_from1", "helix_mean", 2.0, 1),
        note=(
            "One onset in the fine half of the scan. At onset 0 the weight "
            "takes the fold to sheet and at onset 10 it no longer does, while "
            "the unsteered run is already at three quarters of its final helix "
            "by step 10, so whatever decides the fold happens in between. "
            "Outside the grid: one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_from2": ArmSpec(
        key="helix_mean_w20_from2",
        label="Mean helix rung fraction at 2.0 from step 2",
        schedule=schedule.delayed("helix_mean_w20_from2", "helix_mean", 2.0, 2),
        note=(
            "One onset in the fine half of the scan. At onset 0 the weight "
            "takes the fold to sheet and at onset 10 it no longer does, while "
            "the unsteered run is already at three quarters of its final helix "
            "by step 10, so whatever decides the fold happens in between. "
            "Outside the grid: one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_from4": ArmSpec(
        key="helix_mean_w20_from4",
        label="Mean helix rung fraction at 2.0 from step 4",
        schedule=schedule.delayed("helix_mean_w20_from4", "helix_mean", 2.0, 4),
        note=(
            "One onset in the fine half of the scan. At onset 0 the weight "
            "takes the fold to sheet and at onset 10 it no longer does, while "
            "the unsteered run is already at three quarters of its final helix "
            "by step 10, so whatever decides the fold happens in between. "
            "Outside the grid: one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_from7": ArmSpec(
        key="helix_mean_w20_from7",
        label="Mean helix rung fraction at 2.0 from step 7",
        schedule=schedule.delayed("helix_mean_w20_from7", "helix_mean", 2.0, 7),
        note=(
            "One onset in the fine half of the scan. At onset 0 the weight "
            "takes the fold to sheet and at onset 10 it no longer does, while "
            "the unsteered run is already at three quarters of its final helix "
            "by step 10, so whatever decides the fold happens in between. "
            "Outside the grid: one target, run as a diagnostic."
        ),
    ),
    "helix_mean_w20_from10": ArmSpec(
        key="helix_mean_w20_from10",
        label="Mean helix rung fraction at 2.0 from step 10",
        schedule=schedule.delayed("helix_mean_w20_from10", "helix_mean", 2.0, 10),
        note=(
            "One onset in the scan that holds the weight at 2.0 and moves only "
            "the step it arrives. From step 0 the weight takes the fold to "
            "sheet; withheld to step 50 it leaves the fold where the unsteered "
            "objective put it. The scan asks where between those the design "
            "stops being steerable. Outside the grid: one target, run as a "
            "diagnostic."
        ),
    ),
    "helix_mean_w20_from20": ArmSpec(
        key="helix_mean_w20_from20",
        label="Mean helix rung fraction at 2.0 from step 20",
        schedule=schedule.delayed("helix_mean_w20_from20", "helix_mean", 2.0, 20),
        note=(
            "One onset in the scan that holds the weight at 2.0 and moves only "
            "the step it arrives. From step 0 the weight takes the fold to "
            "sheet; withheld to step 50 it leaves the fold where the unsteered "
            "objective put it. The scan asks where between those the design "
            "stops being steerable. Outside the grid: one target, run as a "
            "diagnostic."
        ),
    ),
    "helix_mean_w20_from30": ArmSpec(
        key="helix_mean_w20_from30",
        label="Mean helix rung fraction at 2.0 from step 30",
        schedule=schedule.delayed("helix_mean_w20_from30", "helix_mean", 2.0, 30),
        note=(
            "One onset in the scan that holds the weight at 2.0 and moves only "
            "the step it arrives. From step 0 the weight takes the fold to "
            "sheet; withheld to step 50 it leaves the fold where the unsteered "
            "objective put it. The scan asks where between those the design "
            "stops being steerable. Outside the grid: one target, run as a "
            "diagnostic."
        ),
    ),
    "helix_mean_w20_from40": ArmSpec(
        key="helix_mean_w20_from40",
        label="Mean helix rung fraction at 2.0 from step 40",
        schedule=schedule.delayed("helix_mean_w20_from40", "helix_mean", 2.0, 40),
        note=(
            "One onset in the scan that holds the weight at 2.0 and moves only "
            "the step it arrives. From step 0 the weight takes the fold to "
            "sheet; withheld to step 50 it leaves the fold where the unsteered "
            "objective put it. The scan asks where between those the design "
            "stops being steerable. Outside the grid: one target, run as a "
            "diagnostic."
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
}

CAMPAIGN_ARMS: tuple[str, ...] = (
    "vanilla",
    "helix_mean_w05",
    "helix_mean_w10",
    "helix_mean_w20",
    "helix_mean_w50",
)
"""The arms run across every target: unsteered, then the weight scan.

Every registered arm is either in this grid or in one of the schedule scans.
Arms that earlier versions of this work ran and no longer report are not kept
registered, so the registry and the publication describe the same experiment.
"""

N_SEEDS = 30
SGD_STEPS = 150

ONSET_SCAN: dict[int, str] = {
    0: "helix_mean_w20",
    1: "helix_mean_w20_from1",
    2: "helix_mean_w20_from2",
    4: "helix_mean_w20_from4",
    7: "helix_mean_w20_from7",
    10: "helix_mean_w20_from10",
    20: "helix_mean_w20_from20",
    30: "helix_mean_w20_from30",
    40: "helix_mean_w20_from40",
    50: "helix_mean_w20_from50",
    SGD_STEPS: "vanilla",
}
"""Weight 2.0 against the step it arrives, on one target.

Both ends of the scan are campaigns that exist for their own reasons. A weight
that arrives at step 0 is the constant arm of the grid, and one that arrives at
the step the run ends never acts, which is the unsteered arm. The steps between
are the scan itself.
"""

RELEASE_SCAN: dict[int, str] = {
    0: "vanilla",
    1: "helix_mean_w20_until1",
    2: "helix_mean_w20_until2",
    4: "helix_mean_w20_until4",
    7: "helix_mean_w20_until7",
    10: "helix_mean_w20_until10",
    20: "helix_mean_w20_until20",
    30: "helix_mean_w20_until30",
    40: "helix_mean_w20_until40",
    50: "helix_mean_w20_until50",
    SGD_STEPS: "helix_mean_w20",
}
"""Weight 2.0 from step 0 against the step it is withdrawn, on one target.

The same two campaigns anchor this scan from the other side: withdrawn at step
0 the weight never acts, and withdrawn at the step the run ends it never
leaves. Read together the two scans ask whether the fold commits to the
unsteered basin faster than it commits to the steered one.
"""


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
