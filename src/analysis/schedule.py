"""When each steering loss term applies, and how hard.

A schedule is a list of waypoints. Each waypoint names the weights that take
effect at its step, and they hold until a later waypoint changes them; a name a
later waypoint omits keeps the value it already had. That is enough for both
shapes this publication uses: a constant weight for the whole run, and a burst
that starts strong and then drops to a maintenance level.

Timing is not cosmetic. ``helix_logcap`` is a log barrier clamped at 2.0, so it
stops producing gradient once the top-12 i to i+4 contact probability passes
1 - exp(-2), which is 0.8647. Unsteered runs cross that threshold within a
handful of steps, so a weight that arrives late arrives where the term has no
gradient left to give. The burst arm exists to ask whether front-loading the
weight buys anything the constant arm cannot.

The terms themselves, and the arithmetic behind them, are in
:mod:`analysis.steering`.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch

from analysis.steering import TERM_SPECS, term


@dataclass(frozen=True)
class Waypoint:
    """Weights that take effect at ``step`` and hold until the next waypoint."""

    step: int
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"waypoint step must be >= 0, got {self.step}")
        unknown = set(self.weights) - set(TERM_SPECS)
        if unknown:
            raise KeyError(
                f"unknown steering term(s) {sorted(unknown)}; registered: {sorted(TERM_SPECS)}"
            )
        negative = sorted(name for name, value in self.weights.items() if value < 0)
        if negative:
            raise ValueError(f"steering weights must be >= 0; negative: {negative}")


@dataclass(frozen=True)
class LossSchedule:
    """A named weight schedule over the optimization steps of one design."""

    key: str
    waypoints: tuple[Waypoint, ...]
    base_structure_onset: int = 0

    def __post_init__(self) -> None:
        if self.base_structure_onset < 0:
            raise ValueError("base_structure_onset must be >= 0")
        if not self.waypoints:
            raise ValueError("a schedule needs at least one waypoint")
        steps = [waypoint.step for waypoint in self.waypoints]
        if steps != sorted(steps):
            raise ValueError(f"waypoints must be sorted by step, got {steps}")
        if len(set(steps)) != len(steps):
            raise ValueError(f"waypoints must have distinct steps, got {steps}")
        if steps[0] != 0:
            raise ValueError(f"the first waypoint must be at step 0, got {steps[0]}")

    def weights_at(self, step: int) -> dict[str, float]:
        """The weight each term carries at ``step``. Unset terms are omitted."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        active: dict[str, float] = {}
        for waypoint in self.waypoints:
            if waypoint.step > step:
                break
            active.update(waypoint.weights)
        return active

    def base_structure_weight_at(self, step: int) -> float:
        """Gate the cookbook structural objective independently of steering."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
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
    return LossSchedule(
        key=key, waypoints=(Waypoint(step=0, weights={term(term_key).key: weight}),)
    )


def released(key: str, term_key: str, weight: float, release_step: int) -> LossSchedule:
    """One term at one weight until ``release_step``, then nothing.

    The complement of :func:`delayed`. A delayed arm asks whether a weight can
    still move a fold the unsteered objective has committed; a released arm
    asks whether a fold the weight has already produced survives once the
    weight is gone, and the rest of the objective is free to pull the design
    back. Released at step 0 is the unsteered arm and never released is the
    constant arm, so the two ends of the scan are campaigns that already exist.
    """
    if release_step < 1:
        raise ValueError(f"release_step must be >= 1, got {release_step}")
    name = term(term_key).key
    return LossSchedule(
        key=key,
        waypoints=(
            Waypoint(step=0, weights={name: weight}),
            Waypoint(step=release_step, weights={name: 0.0}),
        ),
    )


def delayed(key: str, term_key: str, weight: float, onset_step: int) -> LossSchedule:
    """No steering until ``onset_step``, then one term at one weight.

    Up to the onset this is the unsteered objective, step for step, which is
    what makes it a test rather than another weight: the design loop draws all
    of its randomness from the seed and the step index and none of it from the
    trajectory, so a delayed arm and an unsteered arm run on the same seed
    share their first ``onset_step`` steps exactly. What happens after the
    onset is therefore the weight acting on a design the unsteered objective
    has already committed, rather than on an initialization.
    """
    if onset_step < 1:
        raise ValueError(f"onset_step must be >= 1, got {onset_step}")
    name = term(term_key).key
    return LossSchedule(
        key=key,
        waypoints=(
            Waypoint(step=0, weights={}),
            Waypoint(step=onset_step, weights={name: weight}),
        ),
    )
