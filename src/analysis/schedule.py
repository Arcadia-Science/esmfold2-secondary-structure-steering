"""Piecewise-constant steering weights and optional base structural loss onset.

Each waypoint updates the named terms; omitted terms retain their prior weight.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Waypoint:
    """Weight updates that take effect at ``step``."""

    step: int
    weights: dict[str, float]


@dataclass(frozen=True)
class LossSchedule:
    """A named weight schedule over the optimization steps of one design."""

    key: str
    waypoints: tuple[Waypoint, ...]
    base_structure_onset: int = 0

    def __post_init__(self) -> None:
        steps = [waypoint.step for waypoint in self.waypoints]
        if steps != sorted(steps):
            raise ValueError(f"waypoints must be sorted by step, got {steps}")
        if len(set(steps)) != len(steps):
            raise ValueError(f"waypoints must have distinct steps, got {steps}")
        if steps[0] != 0:
            raise ValueError(f"the first waypoint must be at step 0, got {steps[0]}")

    def weights_at(self, step: int) -> dict[str, float]:
        """The weight each term carries at ``step``. Unset terms are omitted."""
        active: dict[str, float] = {}
        for waypoint in self.waypoints:
            if waypoint.step > step:
                break
            active.update(waypoint.weights)
        return active

    def base_structure_weight_at(self, step: int) -> float:
        """Gate the cookbook structural objective independently of steering."""
        return float(step >= self.base_structure_onset)

    def structural_objective(
        self, base_loss: torch.Tensor, steering_loss: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Combine active losses before the existing gradient normalization.

        Withholding the base term removes its gradient entirely. The default
        branch is the original sum, preserving existing campaign behavior.
        """
        if self.base_structure_weight_at(step) == 0:
            return steering_loss
        return base_loss + steering_loss

    def summary(self) -> str:
        """One line describing the schedule, for tables and protocol records."""
        if not any(waypoint.weights for waypoint in self.waypoints):
            return "no steering term"
        spans: list[str] = []
        for index, waypoint in enumerate(self.waypoints):
            named = (
                ", ".join(f"{name} {value:g}" for name, value in sorted(waypoint.weights.items()))
                or "no steering term"
            )
            if index + 1 < len(self.waypoints):
                until = self.waypoints[index + 1].step - 1
                spans.append(f"{named} for steps {waypoint.step}-{until}")
            else:
                spans.append(f"{named} from step {waypoint.step} on")
        if self.base_structure_onset:
            spans.append(f"base structural objective from step {self.base_structure_onset} on")
        return "; ".join(spans)

    def to_dict(self) -> dict[str, object]:
        """A JSON-friendly record of the schedule, stored with each campaign."""
        record = {
            "key": self.key,
            "summary": self.summary(),
            "waypoints": [
                {"step": waypoint.step, "weights": dict(waypoint.weights)}
                for waypoint in self.waypoints
            ],
        }
        if self.base_structure_onset:
            record["base_structure_onset"] = self.base_structure_onset
        return record


VANILLA = LossSchedule(key="vanilla", waypoints=(Waypoint(step=0, weights={}),))


def constant(key: str, term_key: str, weight: float) -> LossSchedule:
    """One term at one weight for the whole run."""
    return LossSchedule(key=key, waypoints=(Waypoint(step=0, weights={term_key: weight}),))


def released(key: str, term_key: str, weight: float, release_step: int) -> LossSchedule:
    """Apply one steering weight from step 0 until ``release_step``, exclusively.

    ``release_step`` must be positive.
    """
    return LossSchedule(
        key=key,
        waypoints=(
            Waypoint(step=0, weights={term_key: weight}),
            Waypoint(step=release_step, weights={term_key: 0.0}),
        ),
    )


def delayed(key: str, term_key: str, weight: float, onset_step: int) -> LossSchedule:
    """Apply one steering weight from ``onset_step`` onward.

    ``onset_step`` must be positive; earlier steps have no steering.
    """
    return LossSchedule(
        key=key,
        waypoints=(
            Waypoint(step=0, weights={}),
            Waypoint(step=onset_step, weights={term_key: weight}),
        ),
    )
