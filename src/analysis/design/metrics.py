"""The distogram ipTM proxy and its antibody CDR variant.

Verbatim from ``cookbook/tutorials/binder_design.py`` at upstream commit
``ba4d712``, lines 369-473.

``_is_antibody_sequence`` and ``_cdr_indices`` import ``abnumber`` inside the
function body, as upstream does, so the rest of this module works without
ANARCI installed.
"""

from __future__ import annotations
import math

import torch

from analysis.design.constants import MUTABLE_TOKEN
from analysis.design.losses import get_mid_points


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Pair entropy within cutoff."""

    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    """Map mean pair entropy to [0, 1]; lower entropy → higher score."""
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _is_antibody_sequence(binder_sequence: str) -> bool:
    """Return True if ANARCI recognizes binder_sequence as antibody variable domain(s)."""
    from abnumber.common import _anarci_align

    sequence = binder_sequence.replace(MUTABLE_TOKEN, "A")
    result = _anarci_align(sequences=[sequence], scheme="chothia", allowed_species=None)[0]
    if not result:
        return False
    valid_chain_types = {"H", "K", "L"}
    return all(chain_type in valid_chain_types for _, chain_type, *_ in result)


def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs."""
    from abnumber import Chain
    from abnumber.common import _anarci_align

    result = _anarci_align(sequences=[binder_sequence], scheme="chothia", allowed_species=None)[0]
    chains = [Chain("".join(result[i][0].values()), scheme="chothia") for i in range(len(result))]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
) -> dict[str, float]:
    """Algorithm 15 Distogram ipTM Proxy.

    Distogram iptm proxy for a target|binder complex (binder at suffix).

    Returns distogram_iptm_proxy for all designs and
    cdr_distogram_iptm_proxy when the binder can be annotated as an
    antibody; otherwise the CDR score is NaN.
    """
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]

    binder_length = len(binder_sequence)
    assert distogram_logits.shape[0] == target_length + binder_length

    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        cdr_indices = _cdr_indices(binder_sequence)
        cdr_rows = [binder_start + i for i in cdr_indices]
        cdr_to_target_entropy = _binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
        )
        cdr_distogram_iptm_proxy = _entropy_to_confidence(
            _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }
