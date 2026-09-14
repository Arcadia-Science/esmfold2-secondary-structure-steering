"""Turning model output into a biotite AtomArray and an ESM ProteinComplex.

Verbatim from ``cookbook/tutorials/binder_design.py`` at upstream commit
``ba4d712``, lines 599-682.

This is not :mod:`analysis.secondary_structure`, which measures composition
from finished coordinates. This module only builds coordinates.
"""

from __future__ import annotations
import string
from typing import Any

import biotite.structure
import numpy as np
import torch
from esm.models.esmfold2.constants import MOL_TYPE_NONPOLYMER
from esm.utils.structure.mmcif_parsing import PLDDT_B_FACTOR_SCALE
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

from analysis.design.constants import ELEMENTS, TOKENS

_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords: np.ndarray,
    atom_to_token: np.ndarray,
    res_type: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    mol_type: np.ndarray,
    ref_atom_name_chars: np.ndarray,
    ref_element: np.ndarray,
    atom_attention_mask: np.ndarray,
    plddt_per_atom: np.ndarray | None = None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (
        atom_coord,
        token_idx,
        atom_name_chars,
        element_idx,
        is_not_pad,
    ) in enumerate(
        zip(
            coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask
        )
    ):
        if not is_not_pad:
            continue
        atoms.append(
            biotite.structure.Atom(
                coord=atom_coord,
                chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
                res_id=residue_index[token_idx] + 1,
                res_name=TOKENS[res_type[token_idx]],
                atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
                element=ELEMENTS[element_idx],
                ins_code=" ",
                hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
                b_factor=float(plddt_per_atom[atom_i])
                if plddt_per_atom is not None
                else 0.0,
            )
        )
    return biotite.structure.array(atoms)


def build_complex(
    inputs: dict[str, torch.Tensor], output: dict[str, Any]
) -> ProteinComplex:
    """Build ProteinComplex from model output."""
    plddt_per_atom = output.get("plddt_per_atom")
    if plddt_per_atom is not None:
        plddt_per_atom = plddt_per_atom[0].cpu().numpy() * PLDDT_B_FACTOR_SCALE
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
        plddt_per_atom=plddt_per_atom,
    )
    return ProteinComplex.from_chains(
        [
            ProteinChain.from_atomarray(a, is_predicted=True)
            for a in biotite.structure.chain_iter(atom_arr)
        ]
    )
