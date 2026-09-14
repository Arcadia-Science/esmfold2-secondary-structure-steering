"""Splitting ESMFold2's per-residue pLDDT by chain, and the viability rule.

The point of the split is the viability rule: a binder that is a disordered
tail on a confidently predicted target scores well on the complex mean and
badly on its own, and it is the second number the rule needs. So the tests
below are mostly about the split landing on the right residues and about the
scale being asserted rather than inferred.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from analysis import composition, confidence

TARGET_LENGTH = 40
BINDER_LENGTH = 10


def two_chain_plddt(target: float, binder: float, padding: int = 0) -> torch.Tensor:
    """A (1, residues) tensor: one value across the target, another the binder."""
    values = torch.cat(
        [
            torch.full((TARGET_LENGTH,), target),
            torch.full((BINDER_LENGTH,), binder),
            torch.full((padding,), float("nan")),
        ]
    )
    return values.unsqueeze(0)


def test_each_chain_gets_its_own_mean() -> None:
    result = confidence.chain_plddt(
        two_chain_plddt(0.9, 0.5), TARGET_LENGTH, BINDER_LENGTH
    )
    assert result.target_plddt == pytest.approx(90.0)
    assert result.binder_plddt == pytest.approx(50.0)


def test_the_complex_mean_is_residue_weighted_not_chain_weighted() -> None:
    """The whole reason to split: 82 hides a binder at 50."""
    result = confidence.chain_plddt(
        two_chain_plddt(0.9, 0.5), TARGET_LENGTH, BINDER_LENGTH
    )
    expected = (0.9 * TARGET_LENGTH + 0.5 * BINDER_LENGTH) / (
        TARGET_LENGTH + BINDER_LENGTH
    )
    assert result.complex_plddt == pytest.approx(expected * 100.0)
    assert result.complex_plddt == pytest.approx(82.0)
    assert result.binder_plddt < confidence.PLDDT_SCALE * 0.7


def test_the_binder_is_the_suffix_not_the_prefix() -> None:
    """Reversing the chains has to change the answer, or the slice is wrong."""
    forward = confidence.chain_plddt(
        two_chain_plddt(0.9, 0.5), TARGET_LENGTH, BINDER_LENGTH
    )
    assert forward.target_plddt > forward.binder_plddt


def test_trailing_padding_is_excluded() -> None:
    padded = confidence.chain_plddt(
        two_chain_plddt(0.9, 0.5, padding=16), TARGET_LENGTH, BINDER_LENGTH
    )
    unpadded = confidence.chain_plddt(
        two_chain_plddt(0.9, 0.5), TARGET_LENGTH, BINDER_LENGTH
    )
    assert padded == unpadded


def test_the_batch_element_is_selectable() -> None:
    batched = torch.cat(
        [two_chain_plddt(0.9, 0.5), two_chain_plddt(0.6, 0.8)], dim=0
    )
    first = confidence.chain_plddt(batched, TARGET_LENGTH, BINDER_LENGTH, 0)
    second = confidence.chain_plddt(batched, TARGET_LENGTH, BINDER_LENGTH, 1)
    assert first.binder_plddt == pytest.approx(50.0)
    assert second.binder_plddt == pytest.approx(80.0)


def test_a_per_atom_tensor_is_refused_rather_than_averaged() -> None:
    """plddt_per_atom is a separate output and is not interchangeable."""
    per_atom = torch.rand(1, TARGET_LENGTH + BINDER_LENGTH, 14)
    with pytest.raises(ValueError, match="one value per residue"):
        confidence.chain_plddt(per_atom, TARGET_LENGTH, BINDER_LENGTH)


def test_an_already_scaled_tensor_is_refused_rather_than_rescaled() -> None:
    """Silently rescaling by magnitude would hide a change in model output."""
    with pytest.raises(ValueError, match="0 to 1 scale"):
        confidence.chain_plddt(
            two_chain_plddt(90.0, 50.0), TARGET_LENGTH, BINDER_LENGTH
        )


def test_a_failed_fold_is_refused_rather_than_scored() -> None:
    broken = two_chain_plddt(0.9, 0.5)
    broken[0, TARGET_LENGTH + 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        confidence.chain_plddt(broken, TARGET_LENGTH, BINDER_LENGTH)


def test_a_tensor_shorter_than_the_two_chains_is_refused() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        confidence.chain_plddt(
            two_chain_plddt(0.9, 0.5), TARGET_LENGTH, BINDER_LENGTH + 1
        )


@pytest.mark.parametrize(
    ("target_length", "binder_length"), [(0, BINDER_LENGTH), (TARGET_LENGTH, 0)]
)
def test_an_empty_chain_is_refused(target_length: int, binder_length: int) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        confidence.chain_plddt(
            two_chain_plddt(0.9, 0.5), target_length, binder_length
        )


def a_design(
    helix: float, sheet: float, iptm: float, binder_plddt: float
) -> dict[str, float]:
    return {
        "helix (fraction 0-1)": helix,
        "sheet (fraction 0-1)": sheet,
        "iptm (0-1)": iptm,
        "binder pLDDT (0-100)": binder_plddt,
    }


def test_the_viability_rule_is_a_conjunction() -> None:
    frame = pd.DataFrame(
        [
            a_design(0.10, 0.30, 0.80, 85.0),
            a_design(0.25, 0.30, 0.80, 85.0),
            a_design(0.10, 0.05, 0.80, 85.0),
            a_design(0.10, 0.30, 0.60, 85.0),
            a_design(0.10, 0.30, 0.80, 65.0),
        ]
    )
    assert list(composition.viable(frame)) == [True, False, False, False, False]


def test_the_viability_thresholds_are_inclusive_where_stated() -> None:
    """Sheet, iptm and pLDDT are 'at least'; helix is 'below'."""
    frame = pd.DataFrame(
        [
            a_design(
                composition.MAX_HELIX_FRACTION,
                composition.MIN_SHEET_FRACTION,
                composition.MIN_IPTM,
                composition.MIN_BINDER_PLDDT,
            ),
            a_design(
                composition.MAX_HELIX_FRACTION - 0.01,
                composition.MIN_SHEET_FRACTION,
                composition.MIN_IPTM,
                composition.MIN_BINDER_PLDDT,
            ),
        ]
    )
    assert list(composition.viable(frame)) == [False, True]


def test_viability_refuses_a_frame_missing_binder_plddt() -> None:
    """The gap this closed: without binder pLDDT the rule cannot be applied."""
    frame = pd.DataFrame([a_design(0.10, 0.30, 0.80, 85.0)]).drop(
        columns=["binder pLDDT (0-100)"]
    )
    with pytest.raises(KeyError, match="binder pLDDT"):
        composition.viable(frame)
