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
                f"unknown steering term(s) {sorted(unknown)}; "
                f"registered: {sorted(TERM_SPECS)}"
            )
        negative = sorted(name for name, value in self.weights.items() if value < 0)
        if negative:
            raise ValueError(f"steering weights must be >= 0; negative: {negative}")


@dataclass(frozen=True)
class LossSchedule:
    """A named weight schedule over the optimization steps of one design."""

    key: str
    waypoints: tuple[Waypoint, ...]

    def __post_init__(self) -> None:
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

    def summary(self) -> str:
        """One line describing the schedule, for tables and protocol records."""
        if not any(waypoint.weights for waypoint in self.waypoints):
            return "no steering term"
        spans: list[str] = []
        for index, waypoint in enumerate(self.waypoints):
            named = ", ".join(
                f"{name} {value:g}" for name, value in sorted(waypoint.weights.items())
            )
            if index + 1 < len(self.waypoints):
                until = self.waypoints[index + 1].step - 1
                spans.append(f"{named} for steps {waypoint.step}-{until}")
            else:
                spans.append(f"{named} from step {waypoint.step} on")
        return "; ".join(spans)

    def to_dict(self) -> dict[str, object]:
        """A JSON-friendly record of the schedule, stored with each campaign."""
        return {
            "key": self.key,
            "summary": self.summary(),
            "waypoints": [
                {"step": waypoint.step, "weights": dict(waypoint.weights)}
                for waypoint in self.waypoints
            ],
        }


VANILLA = LossSchedule(key="vanilla", waypoints=(Waypoint(step=0, weights={}),))


def constant(key: str, term_key: str, weight: float) -> LossSchedule:
    """One term at one weight for the whole run."""
    return LossSchedule(
        key=key, waypoints=(Waypoint(step=0, weights={term(term_key).key: weight}),)
    )


def burst(
    key: str, term_key: str, burst_weight: float, hold_weight: float, burst_steps: int
) -> LossSchedule:
    """One term strong for the first ``burst_steps``, then held lower.

    The burst covers the steps in which secondary structure is still forming,
    where a capped barrier still has gradient to give. The hold weight is what
    is left to keep the structure from drifting back.
    """
    if burst_steps < 1:
        raise ValueError(f"burst_steps must be >= 1, got {burst_steps}")
    name = term(term_key).key
    return LossSchedule(
        key=key,
        waypoints=(
            Waypoint(step=0, weights={name: burst_weight}),
            Waypoint(step=burst_steps, weights={name: hold_weight}),
        ),
    )
