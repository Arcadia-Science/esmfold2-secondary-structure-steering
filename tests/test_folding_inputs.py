"""Check conditioning and gradient flow without importing the GPU model stack."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def folding(monkeypatch):
    # Execute the real function bodies; only feature/model dependencies are
    # replaced. This also works where the Biohub transformers fork is absent.
    file = Path(__file__).resolve().parents[1] / "src/analysis/design/folding.py"
    tree = ast.parse(file.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fold_and_get_distogram"
    ]
    namespace = {
        "torch": torch,
        "F": F,
        "TOKENS": ["pad", "gap", *list("ACDEFGHIKLMNPQRSTVWY"), *["X"] * 11],
        "PROTEIN_3TO1": {aa: aa for aa in "ACDEFGHIKLMNPQRSTVWYX"},
        "ESMFold2ExperimentalModel": object,
        "ProteinInput": SimpleNamespace,
        "StructurePredictionInput": SimpleNamespace,
        "seed_context": lambda seed: nullcontext(),
    }
    calls = []

    def prepare(value, **kwargs):
        calls.append((value, kwargs))
        return {"feature": torch.tensor([1.0, 2.0])}

    namespace["prepare_esmfold2_tensors"] = prepare
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(file), "exec"), namespace)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    return SimpleNamespace(**namespace, preparation_calls=calls)


def test_default_preserves_soft_conditioning_and_gradients(folding):
    logits = torch.randn(1, 6, 20, requires_grad=True)
    design = logits.softmax(-1)
    target = F.one_hot(torch.tensor([[2, 3]]), 33).float()
    received = {}

    def model(**kwargs):
        received.update(kwargs)
        return {"distogram_logits": kwargs["res_type_soft"][..., 2].sum()}

    result = folding.fold_and_get_distogram(model, "AC", target, design, seed=12)
    assert torch.equal(received["res_type_soft"], torch.cat((target, F.pad(design, (2, 11))), 1))
    assert received["num_loops"] == 0 and received["num_sampling_steps"] == 1
    assert received["seed"] == 12 and received["calculate_confidence"] is False
    assert folding.preparation_calls[0][1].get("seed") is None
    result["distogram_logits"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
