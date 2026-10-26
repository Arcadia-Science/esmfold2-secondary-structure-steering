"""ESMC sequence regularization using sampled masks and soft sequence targets."""

from __future__ import annotations
import math
from functools import cache

import torch
import torch.nn.functional as F
from esm.models.esmfold2.constants import PROTEIN_1TO3
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

from analysis.design.constants import ESMC_MASK_FRACTION, TOKENS


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    """Build a matrix of shape [ft_aas=20, lm_aas=20]."""
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[tok_3letter] for tok_3letter in TOKENS[2:22]]

    lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]

    ft_to_lm_aa_matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        lm_idx = lm_aas.index(ft_aa)
        ft_to_lm_aa_matrix[ft_idx, lm_idx] = 1

    return ft_to_lm_aa_matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(probs.dtype)


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    esmc_model: ESMCForMaskedLM,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
) -> torch.Tensor:
    """Algorithm 14 ESMC Pseudo-perplexity Sequence Regularization.

    Approximate pseudoperplexity NLL via multiple sampled masks."""
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = ESMCTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1
    input_ids[:, -1, tokenizer.eos_token_id] = 1
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    all_masked_sequences = []
    all_pass_masks = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError("ESMC pseudoperplexity score mask selected zero positions.")

        num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
        pass_masks[
            torch.arange(n_passes, device=device)[:, None],
            position_indices[masked_offsets],
        ] = True

        masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        all_masked_sequences.append(masked_sequences)
        all_pass_masks.append(pass_masks)

    logit_chunks = []
    masked_sequence_rows = torch.cat(all_masked_sequences, dim=0)
    for start in range(0, masked_sequence_rows.size(0), batch_size):
        chunk = masked_sequence_rows[start : start + batch_size]
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden, *_ = esmc.transformer(
                chunk @ esmc.embed.weight.to(chunk.dtype),
                sequence_id=None,
                layers_to_collect=[],
                output_attentions=False,
            )
            logit_chunks.append(esmc_model.lm_head(hidden))
    logits = torch.cat(logit_chunks, dim=0)

    losses = []
    for batch_idx, pass_masks in enumerate(all_pass_masks):
        start = batch_idx * n_passes
        stop = start + n_passes
        target_weights = target_esm[batch_idx]
        log_probs = logits[start:stop].log_softmax(dim=-1)[:, 1:-1, 4:24]
        nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
        losses.append(nlls[pass_masks].mean())

    return torch.stack(losses, dim=0)
