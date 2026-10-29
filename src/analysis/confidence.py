"""Split per-residue model pLDDT into target, binder, and complex means."""

from __future__ import annotations
from dataclasses import dataclass

import torch

PLDDT_SCALE = 100.0


@dataclass(frozen=True)
class ChainPlddt:
    """Mean pLDDT for the whole complex and for each chain, all 0 to 100."""

    complex_plddt: float
    target_plddt: float
    binder_plddt: float


def chain_plddt(
    plddt: torch.Tensor, target_length: int, binder_length: int, batch_index: int = 0
) -> ChainPlddt:
    """Return complex, target, and binder mean pLDDT on a 0–100 scale.

    Input shape is (batch, residues), scaled 0–1, with target then binder.
    Ignore padding beyond the supplied chain lengths.
    """
    if plddt.ndim != 2:
        raise ValueError("Expected one value per residue in a (batch, residues) tensor")
    total = target_length + binder_length
    if plddt.size(1) < total:
        raise ValueError(
            f"pLDDT covers {plddt.size(1)} residues, fewer than the two chains' {total}"
        )

    values = plddt[batch_index, :total].detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Model returned non-finite residue pLDDT")
    if float(values.max()) > 1.0:
        raise ValueError("Expected residue pLDDT on the 0 to 1 scale")

    return ChainPlddt(
        complex_plddt=float(values.mean()) * PLDDT_SCALE,
        target_plddt=float(values[:target_length].mean()) * PLDDT_SCALE,
        binder_plddt=float(values[target_length:].mean()) * PLDDT_SCALE,
    )
