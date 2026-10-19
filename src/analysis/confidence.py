"""Per-chain confidence, read from ESMFold2's own per-residue output.

The viability rule this publication uses needs binder pLDDT, and a complex-wide
mean will not do. A design whose binder is a disordered tail attached to a
confidently predicted target still scores well on the complex mean, which is
exactly the failure the rule is meant to catch. So pLDDT is split by chain.

The model returns ``plddt`` as one value per residue, shaped
``(batch, residues)`` on a 0 to 1 scale, alongside a separate
``plddt_per_atom`` that :func:`analysis.design.structure.build_complex` writes
into the B-factor column at 0 to 100. Both describe the same prediction, but a
per-atom mean and a per-residue mean are not the same number, because residues
differ in atom count. This module reads the per-residue tensor, which is what
produced the figures quoted in this publication, rather than parsing the
B-factors back out of the PDB.

The chain split is positional and needs no chain labels: the design loop folds
``target|binder`` in that order, so the binder is the suffix. Padding, when
there is any, sits past the end of both and is excluded by slicing to the
sequence lengths.

This module needs torch but not ``esm``, so the arithmetic is testable without
the design stack.
"""

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
    """Split a per-residue pLDDT tensor into complex, target and binder means.

    ``plddt`` is ``output["plddt"]`` from the fold, shaped
    ``(batch, residues)`` and scaled 0 to 1. The returned values are scaled to
    0 to 100, which is the scale the B-factor column and every published
    pLDDT threshold use.
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
