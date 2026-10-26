"""Record per-step losses, steering weights, sequence state, and binding proxies.

Rows use unit-labelled columns. Optimization-step ipTM is available only on
confidence steps; the distogram proxy is a separate readout.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from analysis import paths
from analysis.steering import TERM_SPECS

STEP_TRACE_NAME = "step_trace.jsonl"

AGGREGATE_LOSS_UNITS: dict[str, str] = {
    "intra_contact_loss": "dimensionless",
    "inter_contact_loss": "dimensionless",
    "glob_loss": "dimensionless",
    "plm_loss": "nats/residue",
    "structure_loss": "dimensionless",
    "base_structure_loss": "dimensionless",
    "base_structure_weight": "dimensionless",
    "steering_loss": "dimensionless",
    "total_loss": "dimensionless",
}


def loss_unit(name: str) -> str:
    """The unit of one logged loss, for the column header."""
    if name in AGGREGATE_LOSS_UNITS:
        return AGGREGATE_LOSS_UNITS[name]
    return TERM_SPECS[name].unit


def loss_column(name: str) -> str:
    """Column header for one logged loss, unit included."""
    return f"{name} ({loss_unit(name)})"


def weight_column(name: str) -> str:
    """Column header for one steering weight, unit included."""
    return f"weight {name} (dimensionless)"


def sequence_entropy_bits(design_probs: torch.Tensor) -> float:
    """Return mean per-residue soft-sequence entropy in bits for batch element 0."""
    probs = design_probs[0].clamp(min=1e-8)
    return float(-(probs * probs.log2()).sum(dim=-1).mean().detach().cpu())


@dataclass(frozen=True)
class StepRecord:
    """One optimization step of one design.

    ``losses`` holds every loss the step computed, steered or not, keyed by the
    bare loss name. ``weights`` holds the steering weights in force at this
    step; a term the schedule does not mention is absent rather than zero, so
    the log distinguishes "held at zero" from "never configured".
    """

    target: str
    arm: str
    seed: int
    step: int
    steps: int
    temperature: float
    confidence_step: bool
    inversion_model: str
    binder_sequence: str
    step_seconds: float
    sequence_entropy_bits: float
    distogram_iptm_proxy: float
    iptm: float | None
    losses: dict[str, float]
    weights: dict[str, float]

    def row(self) -> dict[str, object]:
        """One flat, unit-labelled, JSON-safe row."""
        row: dict[str, object] = {
            "target": self.target,
            "arm": self.arm,
            "seed": self.seed,
            "step (index)": self.step,
            "steps (count)": self.steps,
            "temperature (0-1)": self.temperature,
            "confidence step (bool)": self.confidence_step,
            "inversion model": self.inversion_model,
            "binder sequence": self.binder_sequence,
            "binder length (residues)": len(self.binder_sequence),
            "step time (s)": self.step_seconds,
            "sequence entropy (bits)": self.sequence_entropy_bits,
            "distogram iptm proxy (0-1)": self.distogram_iptm_proxy,
            "iptm (0-1)": self.iptm,
        }
        for name, value in self.losses.items():
            row[loss_column(name)] = value
        for name in TERM_SPECS:
            row[weight_column(name)] = self.weights.get(name)
        return row


def rows(records: tuple[StepRecord, ...]) -> list[dict[str, object]]:
    """Flatten a whole trace, ready for JSON transport or a frame."""
    return [record.row() for record in records]


def append(record: StepRecord, path: Path) -> None:
    """Append one row as a JSON line and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.row(), separators=(",", ":")) + "\n")
        handle.flush()


def frame(flat_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a frame from already-flattened rows, sorted by seed and step."""
    return pd.DataFrame(flat_rows).sort_values(["seed", "step (index)"]).reset_index(drop=True)


def load(path: Path) -> pd.DataFrame:
    """Read a JSON-lines loss log into a frame."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No loss log at {path}. It is written by the design loop, one line "
            f"per optimization step, and collected with:\n"
            f"    python -m analysis.campaign collect --target <key> --arm <key>"
        )
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return frame([json.loads(line) for line in lines])


def load_campaign(target_key: str, arm: str) -> pd.DataFrame:
    """Read one campaign's collected per-step log, or raise saying how to make it."""
    path = paths.campaign_trace(target_key, arm)
    if not path.is_file():
        raise FileNotFoundError(
            f"No per-step log for {target_key!r} / {arm!r}.\n"
            f"Expected: {path}\n"
            f"It is written when a campaign is collected. To regenerate the "
            f"campaign:\n"
            f"    python -m analysis.campaign run --targets {target_key} --arms {arm}"
        )
    result = pd.read_parquet(path)
    if not result.target.eq(target_key).all() or not result.arm.eq(arm).all():
        raise ValueError(f"Trace identities do not match campaign: {path}")
    return result
