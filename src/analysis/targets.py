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
            "The cookbook objective untouched. Every steering term is still "
            "measured and logged, which is how the reference curve for a "
            "steered arm is built."
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
    "helix_logcap_w02",
    "helix_logcap_w05",
)
"""The arms run across every target.

``helix_logcap_burst20_hold01`` is registered but deliberately outside this
grid. It was built to test whether a steering weight has to arrive early,
before the capped barrier saturates, and the constant 0.5 control showed that
it does not: the two arms are indistinguishable on every readout, so the
burst's advantage over constant 0.2 was its higher opening weight rather than
its timing. It stays registered because its data is what retired the question,
and it is not rerun because a schedule is a knob this work found no use for.
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
