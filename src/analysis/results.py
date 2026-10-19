"""Shared design records and their JSON/Parquet boundary formats.

Workers and archive readers construct the same typed result. CampaignOutput
holds its serialized row separately from the trace and structure artifacts.
"""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

from analysis.confidence import ChainPlddt
from analysis.targets import PRIMARY_CRITIC
from analysis.targets import arm as get_arm


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string; got {value!r}")


def _integer(value: object, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")


@dataclass(frozen=True)
class InitializationSettings:
    prior: str
    temperature: float
    probability_margin: float

    def __post_init__(self) -> None:
        if self.prior not in {"uniform", "helix", "sheet"}:
            raise ValueError(f"Unknown initialization prior: {self.prior!r}")
        for name in ("temperature", "probability_margin"):
            value = getattr(self, name)
            _finite(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")


@dataclass(frozen=True)
class RunSpec:
    target: str
    arm: str
    seed: int
    steps: int
    initialization: InitializationSettings | None = None
    primary_critic: str = PRIMARY_CRITIC

    def __post_init__(self) -> None:
        for name in ("target", "arm", "primary_critic"):
            _text(getattr(self, name), name)
        _integer(self.seed, "seed")
        _integer(self.steps, "steps", minimum=1)

    @classmethod
    def from_protocol(cls, protocol: Mapping, *, seed: int) -> RunSpec:
        original = protocol.get("source_protocol", protocol)
        initialization = None
        if "initialization_temperature" in original:
            initialization = InitializationSettings(
                prior=get_arm(protocol["arm"]).initialization_prior,
                temperature=original["initialization_temperature"],
                probability_margin=original["probability_margin"],
            )
        return cls(
            target=protocol["target"],
            arm=protocol["arm"],
            seed=seed,
            steps=protocol["sgd_steps"] if "sgd_steps" in protocol else original["steps"],
            initialization=initialization,
            primary_critic=original.get("primary_critic", PRIMARY_CRITIC),
        )

    def validate_against(self, expected: RunSpec) -> None:
        differences = [
            f"{field.name}: expected {getattr(expected, field.name)!r}, "
            f"observed {getattr(self, field.name)!r}"
            for field in fields(self)
            if getattr(self, field.name) != getattr(expected, field.name)
        ]
        if differences:
            raise ValueError("Run specification mismatch: " + "; ".join(differences))

    def row(self) -> dict[str, object]:
        row = {
            "seed": self.seed,
            "target": self.target,
            "arm": self.arm,
            "arm_label": get_arm(self.arm).label,
            "steps": self.steps,
        }
        if self.initialization is not None:
            row.update(
                initialization_prior=self.initialization.prior,
                initialization_temperature=self.initialization.temperature,
                probability_margin=self.initialization.probability_margin,
            )
        return row


@dataclass(frozen=True)
class CriticResult:
    name: str
    designed_sequence: str
    iptm: float | None
    plddt: ChainPlddt
    distogram_iptm_proxy: float
    final_loss: float

    def __post_init__(self) -> None:
        _text(self.name, "critic name")
        _text(self.designed_sequence, "designed sequence")
        if self.iptm is not None:
            _finite(self.iptm, "ipTM")
        for field in fields(self.plddt):
            _finite(getattr(self.plddt, field.name), field.name)
        _finite(self.distogram_iptm_proxy, "distogram ipTM proxy")
        _finite(self.final_loss, "final loss")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CriticResult:
        """Accept the same critic record from the model or its preserved JSON."""
        return cls(
            name=record["critic_name"],
            designed_sequence=record["designed_sequence"],
            iptm=record["iptm"],
            plddt=ChainPlddt(
                complex_plddt=record["plddt_complex"],
                target_plddt=record["plddt_target"],
                binder_plddt=record["plddt_binder"],
            ),
            distogram_iptm_proxy=record["distogram_iptm_proxy"],
            final_loss=record["final_loss"],
        )


@dataclass(frozen=True)
class DesignResult:
    run: RunSpec
    critics: tuple[CriticResult, ...]
    selected_step: int | None = None
    binder_name: str | None = None

    def __post_init__(self) -> None:
        names = [critic.name for critic in self.critics]
        if len(names) != len(set(names)) or self.run.primary_critic not in names:
            raise ValueError("Missing primary critic or duplicate critics")
        if any(c.designed_sequence != self.designed_sequence for c in self.critics):
            raise ValueError("Critic sequence differs between evaluations")
        parts = self.designed_sequence.split("|")
        if len(parts) != 2 or not all(parts):
            raise ValueError("Expected target|binder in the designed sequence")
        if self.primary.iptm is None:
            raise ValueError("Primary critic must report ipTM")
        if self.selected_step is not None:
            _integer(self.selected_step, "selected step")
            if self.selected_step >= self.run.steps:
                raise ValueError("Selected step is outside the optimization run")
        if self.binder_name is not None:
            _text(self.binder_name, "binder name")

    @classmethod
    def from_critics(
        cls,
        run: RunSpec,
        records: Sequence[Mapping],
        *,
        selected_step: int | None = None,
        binder_name: str | None = None,
    ) -> DesignResult:
        return cls(
            run=run,
            critics=tuple(CriticResult.from_record(r) for r in records),
            selected_step=selected_step,
            binder_name=binder_name,
        )

    @property
    def primary(self) -> CriticResult:
        return next(c for c in self.critics if c.name == self.run.primary_critic)

    @property
    def designed_sequence(self) -> str:
        return self.primary.designed_sequence

    @property
    def target_sequence(self) -> str:
        return self.designed_sequence.split("|")[0]

    @property
    def binder_sequence(self) -> str:
        return self.designed_sequence.split("|")[1]

    def validate_critics(self, expected: Sequence[str]) -> None:
        """Require a confidence evaluation from every expected campaign critic."""
        if {c.name for c in self.critics} != set(expected):
            raise ValueError("Missing or unexpected critics")
        if any(c.iptm is None for c in self.critics):
            raise ValueError("Expected ipTM from every campaign critic")

    def row(self) -> dict[str, object]:
        """The single table schema used by both workers and archive collection."""
        primary = self.primary
        row = {
            **self.run.row(),
            "target_length_residues": len(self.target_sequence),
            "designed_sequence": self.designed_sequence,
            "binder_sequence": self.binder_sequence,
            "binder_length_residues": len(self.binder_sequence),
            "final_loss": primary.final_loss,
            "critic_name": primary.name,
            "critic_iptm": primary.iptm,
            "critic_plddt_binder": primary.plddt.binder_plddt,
            "critic_plddt_target": primary.plddt.target_plddt,
            "critic_plddt_complex": primary.plddt.complex_plddt,
            "critic_distogram_iptm_proxy": primary.distogram_iptm_proxy,
            "critic_iptm_by_model": {c.name: c.iptm for c in self.critics},
            "critic_plddt_binder_by_model": {c.name: c.plddt.binder_plddt for c in self.critics},
            "critic_distogram_iptm_proxy_by_model": {
                c.name: c.distogram_iptm_proxy for c in self.critics
            },
            "design_steps_recorded": self.run.steps,
        }
        if self.selected_step is not None:
            row["selected_step"] = self.selected_step
        if self.binder_name is not None:
            row["binder_name"] = self.binder_name
        return row


@dataclass(frozen=True)
class DesignCompletion:
    """The full worker's completion record, with its existing disk format."""

    run: RunSpec
    initial_sequence: str
    binder_sequence: str
    selected_step: int
    iptm: float
    binder_plddt: float

    def __post_init__(self) -> None:
        _text(self.initial_sequence, "initial sequence")
        _text(self.binder_sequence, "binder sequence")
        _integer(self.selected_step, "selected step")
        if self.selected_step >= self.run.steps:
            raise ValueError("Selected step is outside the optimization run")
        _finite(self.iptm, "completion ipTM")
        _finite(self.binder_plddt, "completion binder pLDDT")
        if self.run.initialization is None:
            raise ValueError("Full initialization completion requires initialization settings")
        if len(self.initial_sequence) != len(self.binder_sequence):
            raise ValueError("Initial and selected sequence lengths differ")

    @classmethod
    def from_record(cls, record: Mapping, *, target: str, prior: str) -> DesignCompletion:
        run = RunSpec(
            target=target,
            arm=record["arm"],
            seed=record["seed"],
            steps=record["steps"],
            initialization=InitializationSettings(
                prior=prior,
                temperature=record["initialization_temperature"],
                probability_margin=record["probability_margin"],
            ),
            primary_critic=record["primary_critic"],
        )
        value = cls(
            run=run,
            initial_sequence=record["initial_sequence"],
            binder_sequence=record["binder_sequence"],
            selected_step=record["selected_step"],
            iptm=record["iptm"],
            binder_plddt=record["plddt"],
        )
        _integer(record["binder_length"], "binder length", minimum=1)
        if record["binder_length"] != len(value.binder_sequence):
            raise ValueError("Completion binder length differs from its sequence")
        return value

    @classmethod
    def from_result(cls, result: DesignResult, initial_sequence: str) -> DesignCompletion:
        if result.selected_step is None:
            raise ValueError("Completion requires a selected step")
        return cls(
            run=result.run,
            initial_sequence=initial_sequence,
            binder_sequence=result.binder_sequence,
            selected_step=result.selected_step,
            iptm=result.primary.iptm,
            binder_plddt=result.primary.plddt.binder_plddt,
        )

    def validate_against(self, result: DesignResult) -> None:
        self.run.validate_against(result.run)
        if (
            self.binder_sequence != result.binder_sequence
            or self.selected_step != result.selected_step
        ):
            raise ValueError("Completion record differs from selected sequence")
        if not all(
            math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8)
            for a, b in (
                (self.iptm, result.primary.iptm),
                (self.binder_plddt, result.primary.plddt.binder_plddt),
            )
        ):
            raise ValueError("Completion record differs from primary critic")

    def to_record(self) -> dict[str, object]:
        settings = self.run.initialization
        return {
            "seed": self.run.seed,
            "binder_length": len(self.binder_sequence),
            "steps": self.run.steps,
            "arm": self.run.arm,
            "initialization_temperature": settings.temperature,
            "probability_margin": settings.probability_margin,
            "initial_sequence": self.initial_sequence,
            "binder_sequence": self.binder_sequence,
            "primary_critic": self.run.primary_critic,
            "iptm": self.iptm,
            "plddt": self.binder_plddt,
            "selected_step": self.selected_step,
        }


def _empty_containers(value: object, path: str = "") -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [path or "<root>"]
        return [
            p
            for key, child in value.items()
            for p in _empty_containers(child, f"{path}.{key}" if path else key)
        ]
    if isinstance(value, list | tuple):
        if not value:
            return [path or "<root>"]
        return [
            p for i, child in enumerate(value) for p in _empty_containers(child, f"{path}[{i}]")
        ]
    return []


@dataclass(frozen=True)
class CampaignOutput:
    """Serialized table row and artifacts, kept separate until transport."""

    row: dict[str, object]
    trace: list[dict[str, object]]
    complex_pdb: str
    step_complexes_gz: bytes = b""

    def __post_init__(self) -> None:
        if empty := _empty_containers(self.row):
            raise ValueError(
                f"Design row has empty container(s) at {sorted(empty)}; Parquet cannot store them"
            )

    @classmethod
    def from_design(
        cls,
        result: DesignResult,
        trace: list[dict],
        complex_pdb: str,
        step_complexes_gz: bytes = b"",
    ) -> CampaignOutput:
        if len(trace) != result.run.steps:
            raise ValueError(f"Expected {result.run.steps} trace rows, observed {len(trace)}")
        return cls(result.row(), trace, complex_pdb, step_complexes_gz)

    @classmethod
    def from_payload(cls, payload: Mapping, *, seed: int) -> CampaignOutput:
        """Read the existing Modal wire format, including previously submitted calls."""
        row = {
            key: value
            for key, value in payload.items()
            if key not in {"step_trace", "complex_pdb", "step_complexes_gz", "schedule"}
        }
        if "seed" in row and row["seed"] != seed:
            raise ValueError(f"Result seed: expected {seed}, observed {row['seed']}")
        row["seed"] = seed
        return cls(
            row,
            payload["step_trace"],
            payload.get("complex_pdb", ""),
            payload.get("step_complexes_gz", b""),
        )

    def to_payload(self, *, schedule: dict) -> dict[str, object]:
        return {
            **self.row,
            "schedule": schedule,
            "step_trace": self.trace,
            "complex_pdb": self.complex_pdb,
            "step_complexes_gz": self.step_complexes_gz,
        }
