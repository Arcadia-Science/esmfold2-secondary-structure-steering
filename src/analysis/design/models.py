"""Load ESMFold2 checkpoints and compile selected model modules."""

from __future__ import annotations
from typing import Any

import torch
from transformers.models.esmfold2.modeling_esmfold2_common import (
    CUE_AVAILABLE,
    PairUpdateBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    MSAEncoder as ESMFold2MSAEncoder,
)

_ESMC_CACHE: dict[str, torch.nn.Module] = {}


def _load_hf_model(critic_name: str, lm_dropout: float, cache_esmc: bool, device: str) -> Any:
    """Load an ESMFold2 checkpoint, optionally sharing cached ESMC weights."""
    repo_id = f"biohub/{critic_name}"
    model = ESMFold2ExperimentalModel.from_pretrained(repo_id, load_esmc=not cache_esmc)
    if cache_esmc:
        esmc_id = model.config.esmc_id
        if esmc_id not in _ESMC_CACHE:
            model.load_esmc(esmc_id)
            assert model._esmc is not None
            _ESMC_CACHE[esmc_id] = model._esmc
        model._esmc = _ESMC_CACHE[esmc_id]
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    model.set_kernel_backend("cuequivariance" if CUE_AVAILABLE else None)
    return model.to(device=device).eval().requires_grad_(False)


def _apply_torch_compile(model: torch.nn.Module) -> None:
    """A helper for torch compiling the model."""
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if not isinstance(module, compile_targets):
            return
        module.forward = torch.compile(module.forward)

    model.apply(_maybe_compile_module)
