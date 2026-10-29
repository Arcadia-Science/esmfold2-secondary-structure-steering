"""Sequence encodings and mutation masks for binder design."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from esm.models.esmfold2.constants import PROTEIN_1TO3

from analysis.design.constants import AA_DIMS, CYS_IDX, MUTABLE_TOKEN, TOKEN_IDS, TOKENS


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    """Initialize trainable amino-acid logits for a batch of binder prompts.

    Fixed residues receive a logit of 10 for their amino acid. Mutable positions
    receive small random logits, with cysteine set to −1e6.
    """
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:
                assert aa in PROTEIN_1TO3, aa
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Build gradient mask [B, L, V]:
    - 0 for fixed (all amino acids)
    - 0 for cysteine at all positions
    - 1 for non-cysteine amino acids at mutable positions
    """
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot(sequence: str, device="cuda") -> torch.Tensor:
    """Convert target string to one-hot tensor [1, L_target, num_tokens]."""

    const_dict = {token: i for i, token in enumerate(TOKENS)}
    target_index = [const_dict[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()
