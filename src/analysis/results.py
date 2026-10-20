"""Shared design records and their JSON/Parquet boundary formats.

Workers construct typed results. CampaignOutput holds each serialized row
separately from the trace and structure artifacts.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from analysis.confidence import ChainPlddt
from analysis.targets import PRIMARY_CRITIC
from analysis.targets import arm as get_arm


@dataclass(frozen=True)
class RunSpec:
    target: str
    arm: str
    seed: int
    steps: int
    primary_critic: str = PRIMARY_CRITIC

    def row(self) -> dict[str, object]:
        row = {
            "seed": self.seed,
            "target": self.target,
            "arm": self.arm,
            "arm_label": get_arm(self.arm).label,
            "steps": self.steps,
        }
        return row


@dataclass(frozen=True)
class CriticResult:
    name: str
    designed_sequence: str
    iptm: float | None
    plddt: ChainPlddt
    distogram_iptm_proxy: float
    final_loss: float

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CriticResult:
        """Read a critic evaluation from the design worker."""
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
    binder_name: str | None = None

    def __post_init__(self) -> None:
        names = [critic.name for critic in self.critics]
        if len(names) != len(set(names)) or self.run.primary_critic not in names:
            raise ValueError("Missing primary critic or duplicate critics")
        if any(c.designed_sequence != self.designed_sequence for c in self.critics):
            raise ValueError("Critic sequence differs between evaluations")
        if self.primary.iptm is None:
            raise ValueError("Primary critic must report ipTM")

    @classmethod
    def from_critics(
        cls,
        run: RunSpec,
        records: Sequence[Mapping],
        *,
        binder_name: str | None = None,
    ) -> DesignResult:
        return cls(
            run=run,
            critics=tuple(CriticResult.from_record(r) for r in records),
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

    def row(self) -> dict[str, object]:
        """Per-design scores and metadata for the campaign table."""
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
        if self.binder_name is not None:
            row["binder_name"] = self.binder_name
        return row


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
