"""Three-state secondary structure, measured one way everywhere.

Every composition number in this publication comes through this module, so
designs, natural reference chains, and the design targets are all measured
identically: pydssp's Kabsch-Sander hydrogen-bond assignment, collapsed to
helix, sheet and coil, with proline excluded as a backbone amide donor.

The proline donor mask matters. Passing residue ids where residue names are
expected silently disables it, which shifts sheet fraction by up to about two
percentage points on beta-rich chains. :func:`backbone_residues` therefore
returns a dataclass rather than a tuple, so the arguments cannot be swapped.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pydssp
from biotite.sequence import ProteinSequence

BINDER_CHAIN = "B"
MIN_RESIDUES = 8


@dataclass(frozen=True)
class Backbone:
    """Complete N, CA, C, O backbone of one chain, residue-ordered."""

    coords: np.ndarray
    res_ids: np.ndarray
    res_names: np.ndarray

    def __len__(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True)
class Composition:
    """Three-state secondary-structure fractions over one chain."""

    helix_fraction: float
    sheet_fraction: float
    coil_fraction: float
    n_residues: int
    n_helix: int
    n_sheet: int
    n_coil: int
    max_residue_gap: int

    @property
    def helix_percent(self) -> float:
        return 100.0 * self.helix_fraction

    @property
    def sheet_percent(self) -> float:
        return 100.0 * self.sheet_fraction

    @property
    def coil_percent(self) -> float:
        return 100.0 * self.coil_fraction


def backbone_residues(chain: struc.AtomArray) -> Backbone:
    """Extract the complete backbone of one chain, dropping partial residues."""
    starts = struc.get_residue_starts(chain)
    coords: list[list[np.ndarray]] = []
    res_ids: list[int] = []
    res_names: list[str] = []
    n_res = len(starts)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < n_res else chain.array_length()
        res = chain[start:end]
        atoms: dict[str, np.ndarray] = {}
        complete = True
        for name in ("N", "CA", "C", "O"):
            sel = res[res.atom_name == name]
            if sel.array_length() == 0:
                complete = False
                break
            atoms[name] = np.asarray(sel.coord[0], dtype=float)
        if not complete:
            continue
        coords.append([atoms["N"], atoms["CA"], atoms["C"], atoms["O"]])
        res_ids.append(int(res.res_id[0]))
        res_names.append(str(res.res_name[0]))
    if len(coords) < MIN_RESIDUES:
        raise ValueError(f"only {len(coords)} complete backbone residues, need {MIN_RESIDUES}")
    return Backbone(
        coords=np.asarray(coords, dtype=float),
        res_ids=np.asarray(res_ids, dtype=int),
        res_names=np.asarray(res_names),
    )


def assign(backbone: Backbone) -> np.ndarray:
    """Per-residue three-state labels: ``H``, ``E`` or ``-``."""
    donor = np.array([0.0 if name == "PRO" else 1.0 for name in backbone.res_names], dtype=float)
    return np.asarray(pydssp.assign(backbone.coords, donor_mask=donor, out_type="c3"))


def checked_backbone(path: Path, sequence: str, chain_id: str = BINDER_CHAIN) -> Backbone:
    """Read a predicted backbone and verify its identity, order and coordinates."""
    backbone = backbone_residues(_chain(path, chain_id, None))
    observed = "".join(ProteinSequence.convert_letter_3to1(n) for n in backbone.res_names)
    if observed != sequence or not np.isfinite(backbone.coords).all():
        raise ValueError(f"Invalid coordinates or sequence mismatch: {path}")
    if not (np.diff(backbone.res_ids) == 1).all():
        raise ValueError(f"Noncontiguous backbone: {path}")
    return backbone


def geometry_pass(backbone: Backbone) -> bool:
    """Flag implausible adjacent CA distances or N–CA bond lengths."""
    adjacent = np.linalg.norm(np.diff(backbone.coords[:, 1], axis=0), axis=-1)
    nca = np.linalg.norm(backbone.coords[:, 0] - backbone.coords[:, 1], axis=-1)
    return bool(
        3 < np.median(adjacent) < 5
        and 1 < np.median(nca) < 2
        and ((adjacent > 3) & (adjacent < 5)).mean() > 0.9
    )


def composition(backbone: Backbone) -> Composition:
    """Collapse per-residue labels to fractions and counts.

    ``max_residue_gap`` is the largest jump in author residue numbering across
    the labelled residues. It is 1 for a chain with no break, and larger where
    the model is missing residues, which is how an experimental structure
    advertises disorder.
    """
    labels = assign(backbone)
    gaps = np.diff(backbone.res_ids)
    return Composition(
        helix_fraction=float(np.mean(labels == "H")),
        sheet_fraction=float(np.mean(labels == "E")),
        coil_fraction=float(np.mean(labels == "-")),
        n_residues=int(len(labels)),
        n_helix=int((labels == "H").sum()),
        n_sheet=int((labels == "E").sum()),
        n_coil=int((labels == "-").sum()),
        max_residue_gap=int(gaps.max()) if len(gaps) else 0,
    )


def _chain(path: Path, chain_id: str, crop: tuple[int, int] | None) -> struc.AtomArray:
    if path.suffix in (".cif", ".bcif"):
        reader = pdbx.BinaryCIFFile if path.suffix == ".bcif" else pdbx.CIFFile
        array = pdbx.get_structure(reader.read(path), model=1)
    else:
        import biotite.structure.io.pdb as pdb

        array = pdb.get_structure(pdb.PDBFile.read(path), model=1)
    array = array[struc.filter_amino_acids(array)]
    array = array[array.chain_id == chain_id]
    if array.array_length() == 0:
        raise ValueError(f"chain {chain_id!r} not found in {path}")
    if crop is not None:
        array = array[(array.res_id >= crop[0]) & (array.res_id <= crop[1])]
    return array


def structure_composition(
    path: Path,
    chain_id: str,
    crop: tuple[int, int] | None = None,
) -> Composition:
    """Composition of one chain of a structure file, optionally cropped."""
    return composition(backbone_residues(_chain(path, chain_id, crop)))


def binder_composition(complex_pdb: Path) -> Composition:
    """Composition of the designed binder in a predicted complex.

    The design pipeline writes the target as chain A and the binder as chain B.
    """
    return structure_composition(complex_pdb, BINDER_CHAIN)
