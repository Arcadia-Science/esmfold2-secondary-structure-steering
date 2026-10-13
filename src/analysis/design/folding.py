"""Input preparation and the ESMFold2 forward pass that yields the distogram.

Adapted from ``cookbook/tutorials/binder_design.py`` at upstream commit
``ba4d712``, lines 479-596. Optional prepared inputs and discrete conditioning
support exploratory forward comparisons; defaults preserve the design path.

``fold_and_get_distogram`` takes the argmax of the padded soft design to get the
designed sequence string, which is the one upstream inline comment dropped from
this file. It calls ``.cuda()`` on the stacked inputs, so this module requires a
GPU.
"""

from __future__ import annotations
from functools import cache

import torch
import torch.nn.functional as F
from esm.models.esmfold2 import (
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import PROTEIN_3TO1
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)

from analysis.design.constants import TOKENS


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, torch.Tensor]:
    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


def prepare_folding_inputs(
    target_seq: str,
    design: torch.Tensor,
    seed: int | None = None,
) -> dict:
    """Prepare reusable discrete features from the design's argmax sequence.

    This optional entry point lets exploratory comparisons hold actual atom
    features fixed. The ordinary design path still prepares them on each call.
    ``seed`` controls preparation only and defaults to the original behavior.
    """
    padding = (2, 11)
    padded_design = F.pad(design, padding, mode="constant", value=0)

    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list] for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    for seq in seq_list:
        sequences = {sequence: [str(idx)] for idx, sequence in enumerate(seq.split("|"))}
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=chain_id, sequence=sequence, msa=None)
                for sequence, chain_id in sequences.items()
            ]
        )
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms, seed=seed))

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda() for key in inputs_list[0]
    }
    return {"inputs": inputs, "inputs_list": inputs_list, "seq_list": seq_list}


def fold_and_get_distogram(
    model: ESMFold2ExperimentalModel,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
    *,
    prepared: dict | None = None,
    use_soft_sequence: bool = True,
) -> dict:
    """Fold a design, optionally reusing features or omitting soft conditioning.

    Defaults preserve the design algorithm. ``prepared`` must describe the
    same target and argmax binder sequence. ``use_soft_sequence=False`` uses
    ordinary discrete conditioning while retaining the other forward options.
    """
    if prepared is None:
        prepared = prepare_folding_inputs(target_seq, design)
    else:
        expected = [
            target_seq + "|" + "".join(PROTEIN_3TO1[TOKENS[int(i) + 2]] for i in row)
            for row in design.argmax(dim=-1).tolist()
        ]
        if prepared["seq_list"] != expected:
            raise ValueError("Prepared features do not match the target and argmax design")
    inputs = dict(prepared["inputs"])
    if "res_type_soft" in inputs:
        raise ValueError("Prepared discrete features must not contain res_type_soft")
    if use_soft_sequence:
        inputs["res_type_soft"] = torch.cat(
            (target_one_hot.repeat(design.size(0), 1, 1), F.pad(design, (2, 11))), dim=1
        )

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": prepared["inputs_list"],
        "output": output,
        "seq_list": prepared["seq_list"],
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result
