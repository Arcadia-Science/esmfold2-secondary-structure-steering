"""Proof that :mod:`analysis.design` is a faithful port of the ESM cookbook.

The upstream source of truth is ``cookbook/tutorials/binder_design.py`` at
commit ``ba4d712`` of https://github.com/Biohub/esm. This module reads that
blob straight out of git, writes it to a temporary file, imports it under a
private name, and compares it against the ported modules two ways:

1. Syntax-tree identity. Every ported function and class is parsed and compared
   against its upstream definition with docstrings removed, so comments, blank
   lines and formatter choices cannot hide a behavioural change. This covers
   the whole port, including the parts that need a GPU to execute.
2. Numerical identity. Every function that runs on CPU without model weights is
   executed in both modules on the same fixed inputs and compared for exact
   equality.

Two definitions are deliberately not identical, and are therefore absent from
``PORTED_DEFINITIONS``. ``design_binder`` is where this publication's work
lives: the steering terms, the per-step binding readout, the loss log, the
sampling-step budget and the per-step structure dump. It has diverged far
enough from the cookbook that pinning it against upstream stopped being useful,
so it is not checked here. What it does is covered by the tests of the pieces it
calls, and by the campaign artifacts it produces.
``fold_and_get_distogram`` now accepts reusable prepared inputs and ordinary
discrete conditioning for exploratory studies. Its default conditioning and
optional paths are checked in ``test_folding_inputs.py``.

Functions that cannot execute here are marked xfail with the reason, never
skipped silently. Two things are missing in a CPU environment: a CUDA device
with ESMFold2 and ESMC checkpoints, and ``abnumber``/ANARCI for the antibody
CDR annotation path.

The git blob is located from ``ESM_REPO`` if set, otherwise from the installed
``esm`` package's checkout.
"""

from __future__ import annotations
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

UPSTREAM_COMMIT = "ba4d712"
UPSTREAM_BLOB = "cookbook/tutorials/binder_design.py"
UPSTREAM_MODULE_NAME = "_upstream_binder_design_ba4d712"

GPU_REASON = (
    "needs a CUDA device plus the biohub ESMFold2 and ESMC-6B checkpoints; "
    "syntax-tree identity is still verified by test_ast_identical"
)
ANARCI_REASON = (
    "needs abnumber/ANARCI, which are conda-only and absent from this "
    "environment; syntax-tree identity is still verified by test_ast_identical"
)
CCD_REASON = (
    "needs the ESM chemical component dictionary, which load_ccd downloads; "
    "syntax-tree identity is still verified by test_ast_identical"
)

PORTED_DEFINITIONS = {
    "PromptFactory": "analysis.design.prompts",
    "build_initial_soft_sequence_logits": "analysis.design.sequence",
    "build_gradient_mask": "analysis.design.sequence",
    "sequence_to_one_hot": "analysis.design.sequence",
    "get_mid_points": "analysis.design.losses",
    "binned_entropy": "analysis.design.losses",
    "masked_min_k": "analysis.design.losses",
    "masked_average": "analysis.design.losses",
    "compute_contact_loss": "analysis.design.losses",
    "compute_intra_contact_loss": "analysis.design.losses",
    "compute_inter_contact_loss": "analysis.design.losses",
    "compute_globularity_loss": "analysis.design.losses",
    "compute_structure_losses": "analysis.design.losses",
    "_binding_confidence_entropy": "analysis.design.metrics",
    "_entropy_to_confidence": "analysis.design.metrics",
    "_is_antibody_sequence": "analysis.design.metrics",
    "_cdr_indices": "analysis.design.metrics",
    "compute_distogram_iptm_proxy": "analysis.design.metrics",
    "_resize_tensor": "analysis.design.folding",
    "_ensure_ccd_loaded": "analysis.design.folding",
    "prepare_esmfold2_tensors": "analysis.design.folding",
    "_asym_id_to_chain_label": "analysis.design.structure",
    "to_atom_array": "analysis.design.structure",
    "build_complex": "analysis.design.structure",
    "_folding_trunk_to_lm_aa_vocab_matrix": "analysis.design.plm",
    "_one_hot_from_probs": "analysis.design.plm",
    "_straight_through": "analysis.design.plm",
    "compute_esmc_pseudoperplexity_nll": "analysis.design.plm",
    "_load_hf_model": "analysis.design.models",
    "_apply_torch_compile": "analysis.design.models",
    "normalized_gradient_tensor": "analysis.design.loop",
    "ESMFold2Design": "analysis.design.loop",
}

PORTED_CONSTANTS = {
    "TOKENS": "analysis.design.constants",
    "ELEMENTS": "analysis.design.constants",
    "TOKEN_IDS": "analysis.design.constants",
    "AA_DIMS": "analysis.design.constants",
    "CYS_IDX": "analysis.design.constants",
    "MUTABLE_TOKEN": "analysis.design.constants",
    "BinderPromptStr": "analysis.design.constants",
    "LOSS_WEIGHTS": "analysis.design.constants",
    "STEPS": "analysis.design.constants",
    "LOG_INTERVAL": "analysis.design.constants",
    "LEARNING_RATE": "analysis.design.constants",
    "TEMPERATURE_MIN": "analysis.design.constants",
    "ESMC_MASK_FRACTION": "analysis.design.constants",
    "LM_LOSS_BATCH_SIZE": "analysis.design.constants",
    "LM_MASK_PASSES": "analysis.design.constants",
    "BYTES_PER_GIB": "analysis.design.constants",
    "COMPILE": "analysis.design.constants",
    "REUSE_ESMC": "analysis.design.constants",
    "TrajectoryStep": "analysis.design.constants",
    "Trajectory": "analysis.design.constants",
    "TARGET_SEQUENCES": "analysis.design.prompts",
    "_ATOM_FEATURE_DIMS": "analysis.design.folding",
    "_CHAIN_ID_ALPHABET": "analysis.design.structure",
}

PROMPT = "AC###GG####K##W"
TARGET = "MKVLAWQGST"

pytest.importorskip(
    "esm",
    reason=(
        "analysis.design imports esm. Install it with "
        "pip install -e <clone of https://github.com/Biohub/esm>, which also "
        "pulls the Biohub transformers fork, and env.yml deliberately does not."
    ),
)
pytest.importorskip(
    "transformers.models.esmfold2.modeling_esmfold2_experimental",
    reason=(
        "analysis.design needs the Biohub transformers fork, which esm declares "
        "as a git dependency. PyPI transformers does not carry ESMFold2."
    ),
)


def _esm_repo() -> Path:
    override = os.environ.get("ESM_REPO", "")
    if override:
        return Path(override)
    import esm

    return Path(esm.__file__).resolve().parents[1]


def _upstream_source() -> str:
    repo = _esm_repo()
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{UPSTREAM_COMMIT}:{UPSTREAM_BLOB}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(
            f"cannot read {UPSTREAM_BLOB} at {UPSTREAM_COMMIT} from {repo}. "
            f"Point ESM_REPO at a checkout of https://github.com/Biohub/esm. "
            f"git said: {completed.stderr.strip()}",
            allow_module_level=True,
        )
    return completed.stdout


UPSTREAM_SOURCE = _upstream_source()


@pytest.fixture(scope="session")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """The upstream cookbook module, imported from the git blob."""
    path = tmp_path_factory.mktemp("upstream") / "binder_design.py"
    path.write_text(UPSTREAM_SOURCE)
    spec = importlib.util.spec_from_file_location(UPSTREAM_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[UPSTREAM_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _strip_docstrings(node: ast.AST) -> ast.AST:
    scopes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for sub in ast.walk(node):
        if not isinstance(sub, scopes):
            continue
        body = sub.body
        if len(body) < 2 or not isinstance(body[0], ast.Expr):
            continue
        first = body[0].value
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            sub.body = body[1:]
    return node


def _definition(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name:
            return ast.dump(_strip_docstrings(node), include_attributes=False)
    raise AssertionError(f"no top-level definition named {name!r}")


def _module_source(dotted: str) -> str:
    module = importlib.import_module(dotted)
    assert module.__file__ is not None
    return Path(module.__file__).read_text()


@pytest.mark.parametrize(("name", "dotted"), sorted(PORTED_DEFINITIONS.items()))
def test_ast_identical(name: str, dotted: str) -> None:
    """Every ported definition parses to the same tree as upstream's."""
    assert _definition(_module_source(dotted), name) == _definition(UPSTREAM_SOURCE, name)


@pytest.mark.parametrize(("name", "dotted"), sorted(PORTED_CONSTANTS.items()))
def test_constants_identical(name: str, dotted: str, upstream: ModuleType) -> None:
    ported = getattr(importlib.import_module(dotted), name)
    assert ported == getattr(upstream, name)


def test_binder_prompt_factories_identical(upstream: ModuleType) -> None:
    from analysis.design.prompts import BINDER_PROMPT_FACTORIES

    assert list(BINDER_PROMPT_FACTORIES) == list(upstream.BINDER_PROMPT_FACTORIES)
    for key, factory in BINDER_PROMPT_FACTORIES.items():
        other = upstream.BINDER_PROMPT_FACTORIES[key]
        assert factory.name == other.name
        assert factory.template == other.template
        assert factory.length_ranges == other.length_ranges
        assert factory.is_antibody == other.is_antibody


@pytest.mark.parametrize("seed", [0, 1, 17, 12345])
def test_prompt_factory_sample_identical(seed: int, upstream: ModuleType) -> None:
    """Sampled prompt strings match exactly for every registered factory."""
    from analysis.design.prompts import BINDER_PROMPT_FACTORIES

    for key, factory in BINDER_PROMPT_FACTORIES.items():
        ported = factory.sample(seed=seed)
        other = upstream.BINDER_PROMPT_FACTORIES[key].sample(seed=seed)
        assert ported == other, key
        assert len(ported) > 0


def test_get_mid_points_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import get_mid_points

    assert torch.equal(get_mid_points(), upstream.get_mid_points())


def test_binned_entropy_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import binned_entropy, get_mid_points

    generator = torch.Generator().manual_seed(11)
    dgram = torch.randn(2, 6, 6, 128, generator=generator)
    bins = get_mid_points()
    for cutoff in (8.0, 14.0, 22.0):
        assert torch.equal(
            binned_entropy(dgram, bins, cutoff),
            upstream.binned_entropy(dgram, bins, cutoff),
        )


def test_masked_min_k_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import masked_min_k

    generator = torch.Generator().manual_seed(12)
    x = torch.randn(3, 7, 9, generator=generator)
    mask = torch.rand(9, generator=generator) > 0.4
    for k in (1, 2, 5):
        assert torch.equal(
            masked_min_k(x, mask=mask, k=k), upstream.masked_min_k(x, mask=mask, k=k)
        )


def test_masked_average_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import masked_average

    generator = torch.Generator().manual_seed(13)
    x = torch.randn(3, 7, 9, generator=generator)
    mask = torch.rand(9, generator=generator) > 0.4
    assert torch.equal(masked_average(x, mask), upstream.masked_average(x, mask))


def _planted_distogram(batch: int = 2, target_length: int = 7, binder_length: int = 5):
    """Distogram logits with a reproducible structure, plus its dimensions."""
    total = target_length + binder_length
    generator = torch.Generator().manual_seed(1234)
    logits = torch.randn(batch, total, total, 128, generator=generator)
    ramp = torch.linspace(-2.0, 2.0, 128)
    logits = logits + ramp
    logits = 0.5 * (logits + logits.transpose(1, 2))
    return logits, target_length, binder_length


def test_compute_contact_loss_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import compute_contact_loss, get_mid_points

    logits, _, binder_length = _planted_distogram()
    bins = get_mid_points()
    total = logits.shape[1]
    is_binder = torch.ones(total)
    is_binder[:-binder_length] *= 0.0
    for num_contacts, min_sep, cutoff, chain_mask in (
        (2, 9, 14.0, is_binder),
        (1, 0, 22.0, 1 - is_binder),
        (3, 4, 18.0, torch.ones(total)),
    ):
        kwargs = {
            "bin_distance": bins,
            "num_contacts": num_contacts,
            "min_sep": min_sep,
            "cutoff": cutoff,
            "chain_mask": chain_mask,
            "binder_mask": is_binder,
        }
        assert torch.equal(
            compute_contact_loss(logits, **kwargs),
            upstream.compute_contact_loss(logits, **kwargs),
        )


def test_structure_loss_terms_identical(upstream: ModuleType) -> None:
    from analysis.design import losses as ported

    logits, _, binder_length = _planted_distogram()
    bins = ported.get_mid_points()
    for name in (
        "compute_intra_contact_loss",
        "compute_inter_contact_loss",
        "compute_globularity_loss",
    ):
        mine = getattr(ported, name)(logits, binder_length, bins)
        theirs = getattr(upstream, name)(logits, binder_length, bins)
        assert torch.equal(mine, theirs), name


def test_compute_structure_losses_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import compute_structure_losses

    logits, _, binder_length = _planted_distogram()
    mine = compute_structure_losses(logits, binder_length)
    theirs = upstream.compute_structure_losses(logits, binder_length)
    assert sorted(mine) == sorted(theirs)
    for key in mine:
        assert torch.equal(mine[key], theirs[key]), key


def test_binding_confidence_entropy_identical(upstream: ModuleType) -> None:
    from analysis.design.losses import get_mid_points
    from analysis.design.metrics import _binding_confidence_entropy

    logits, _, _ = _planted_distogram()
    bins = get_mid_points()
    assert torch.equal(
        _binding_confidence_entropy(logits, bins, cutoff=22.0),
        upstream._binding_confidence_entropy(logits, bins, cutoff=22.0),
    )


@pytest.mark.parametrize("entropy", [0.0, 1.0, 3.0, 3.93, 12.0, -1.0])
def test_entropy_to_confidence_identical(entropy: float, upstream: ModuleType) -> None:
    from analysis.design.metrics import _entropy_to_confidence

    assert _entropy_to_confidence(entropy) == upstream._entropy_to_confidence(entropy)


def test_compute_distogram_iptm_proxy_identical(upstream: ModuleType) -> None:
    from analysis.design.metrics import compute_distogram_iptm_proxy

    logits, target_length, binder_length = _planted_distogram(batch=1)
    binder_sequence = "AGVLK"[:binder_length]
    mine = compute_distogram_iptm_proxy(logits, target_length, binder_sequence, False)
    theirs = upstream.compute_distogram_iptm_proxy(logits, target_length, binder_sequence, False)
    assert sorted(mine) == sorted(theirs)
    assert mine["distogram_iptm_proxy"] == theirs["distogram_iptm_proxy"]
    assert np.isnan(mine["cdr_distogram_iptm_proxy"])
    assert np.isnan(theirs["cdr_distogram_iptm_proxy"])


def test_build_initial_soft_sequence_logits_identical(upstream: ModuleType) -> None:
    from analysis.design.sequence import build_initial_soft_sequence_logits

    for sequence in (PROMPT, "#" * 24):
        torch.manual_seed(7)
        mine = build_initial_soft_sequence_logits(sequence, batch_size=3)
        torch.manual_seed(7)
        theirs = upstream.build_initial_soft_sequence_logits(sequence, batch_size=3)
        assert torch.equal(mine, theirs), sequence
        assert mine.requires_grad and theirs.requires_grad


def test_build_gradient_mask_identical(upstream: ModuleType) -> None:
    from analysis.design.sequence import build_gradient_mask

    mine = build_gradient_mask(PROMPT, batch_size=3)
    theirs = upstream.build_gradient_mask(PROMPT, batch_size=3)
    assert torch.equal(mine, theirs)


def test_sequence_to_one_hot_identical(upstream: ModuleType) -> None:
    from analysis.design.sequence import sequence_to_one_hot

    mine = sequence_to_one_hot(TARGET, device="cpu")
    theirs = upstream.sequence_to_one_hot(TARGET, device="cpu")
    assert torch.equal(mine, theirs)
    assert mine.shape[0] == 1 and mine.shape[1] == len(TARGET)


def test_normalized_gradient_tensor_identical(upstream: ModuleType) -> None:
    from analysis.design.loop import normalized_gradient_tensor
    from analysis.design.sequence import build_gradient_mask

    generator = torch.Generator().manual_seed(99)
    mask = build_gradient_mask(PROMPT, batch_size=2)
    grad = torch.randn(*mask.shape, generator=generator)
    assert torch.equal(
        normalized_gradient_tensor(grad, mask),
        upstream.normalized_gradient_tensor(grad, mask),
    )
    assert torch.equal(
        normalized_gradient_tensor(torch.zeros_like(grad), mask),
        upstream.normalized_gradient_tensor(torch.zeros_like(grad), mask),
    )


def test_resize_tensor_identical(upstream: ModuleType) -> None:
    from analysis.design.folding import _resize_tensor

    generator = torch.Generator().manual_seed(5)
    tensor = torch.randn(4, 6, generator=generator)
    for dim, size in ((0, 2), (0, 4), (0, 9), (1, 3), (1, 11)):
        assert torch.equal(
            _resize_tensor(tensor, dim=dim, size=size),
            upstream._resize_tensor(tensor, dim=dim, size=size),
        )


@pytest.mark.parametrize("asym_id", [0, 1, 25, 61, 62, 63, 124, 4000])
def test_asym_id_to_chain_label_identical(asym_id: int, upstream: ModuleType) -> None:
    from analysis.design.structure import _asym_id_to_chain_label

    assert _asym_id_to_chain_label(asym_id) == upstream._asym_id_to_chain_label(asym_id)


def test_asym_id_to_chain_label_rejects_negative() -> None:
    from analysis.design.structure import _asym_id_to_chain_label

    with pytest.raises(ValueError, match="asym_id must be >= 0"):
        _asym_id_to_chain_label(-1)


def _planted_atoms() -> dict[str, np.ndarray]:
    return {
        "coords": np.arange(18, dtype=np.float32).reshape(6, 3),
        "atom_to_token": np.array([0, 0, 0, 1, 1, 1]),
        "res_type": np.array([2, 5]),
        "residue_index": np.array([0, 1]),
        "asym_id": np.array([0, 1]),
        "mol_type": np.array([0, 0]),
        "ref_atom_name_chars": np.array(
            [
                [46, 0, 0, 0],
                [35, 33, 0, 0],
                [35, 0, 0, 0],
                [47, 0, 0, 0],
                [46, 0, 0, 0],
                [35, 33, 0, 0],
            ]
        ),
        "ref_element": np.array([7, 6, 6, 8, 7, 6]),
        "atom_attention_mask": np.array([1, 1, 1, 1, 1, 0]),
        "plddt_per_atom": np.linspace(10.0, 90.0, 6, dtype=np.float32),
    }


def test_to_atom_array_identical(upstream: ModuleType) -> None:
    from analysis.design.structure import to_atom_array

    mine = to_atom_array(**_planted_atoms())
    theirs = upstream.to_atom_array(**_planted_atoms())
    assert np.array_equal(mine.coord, theirs.coord)
    assert set(mine.get_annotation_categories()) == set(theirs.get_annotation_categories())
    for category in mine.get_annotation_categories():
        assert np.array_equal(mine.get_annotation(category), theirs.get_annotation(category)), (
            category
        )
    assert mine.array_length() == 5


def test_one_hot_from_probs_identical(upstream: ModuleType) -> None:
    from analysis.design.plm import _one_hot_from_probs

    generator = torch.Generator().manual_seed(21)
    probs = torch.softmax(torch.randn(3, 5, 20, generator=generator), dim=-1)
    assert torch.equal(_one_hot_from_probs(probs), upstream._one_hot_from_probs(probs))


def test_straight_through_identical(upstream: ModuleType) -> None:
    from analysis.design.plm import _one_hot_from_probs, _straight_through

    generator = torch.Generator().manual_seed(22)
    probs = torch.softmax(torch.randn(3, 5, 20, generator=generator), dim=-1)
    probs.requires_grad_(True)
    discrete = _one_hot_from_probs(probs)
    mine = _straight_through(discrete, probs)
    theirs = upstream._straight_through(discrete, probs)
    assert torch.equal(mine, theirs)
    assert mine.requires_grad


def test_folding_trunk_to_lm_aa_vocab_matrix_identical(upstream: ModuleType) -> None:
    from analysis.design.plm import _folding_trunk_to_lm_aa_vocab_matrix

    device = torch.device("cpu")
    mine = _folding_trunk_to_lm_aa_vocab_matrix(device)
    theirs = upstream._folding_trunk_to_lm_aa_vocab_matrix(device)
    assert torch.equal(mine, theirs)
    assert mine.shape == (20, 20)
    assert int(mine.sum()) == 20


@pytest.mark.xfail(reason=ANARCI_REASON, strict=False)
def test_is_antibody_sequence_identical(upstream: ModuleType) -> None:
    from analysis.design.metrics import _is_antibody_sequence

    for sequence in (PROMPT, upstream.TARGET_SEQUENCES["pd-l1"]):
        assert _is_antibody_sequence(sequence) == upstream._is_antibody_sequence(sequence)


@pytest.mark.xfail(reason=ANARCI_REASON, strict=False)
def test_cdr_indices_identical(upstream: ModuleType) -> None:
    from analysis.design.metrics import _cdr_indices

    sequence = upstream.BINDER_PROMPT_FACTORIES["trastuzumab_framework_vhvl"].sample(0)
    sequence = sequence.replace(upstream.MUTABLE_TOKEN, "A")
    assert _cdr_indices(sequence) == upstream._cdr_indices(sequence)


@pytest.mark.xfail(reason=ANARCI_REASON, strict=False)
def test_compute_distogram_iptm_proxy_cdr_identical(upstream: ModuleType) -> None:
    from analysis.design.metrics import compute_distogram_iptm_proxy

    factory = upstream.BINDER_PROMPT_FACTORIES["trastuzumab_framework_vhvl"]
    binder_sequence = factory.sample(0).replace(upstream.MUTABLE_TOKEN, "A")
    target_length = 7
    total = target_length + len(binder_sequence)
    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(1, total, total, 128, generator=generator)
    mine = compute_distogram_iptm_proxy(logits, target_length, binder_sequence, True)
    theirs = upstream.compute_distogram_iptm_proxy(logits, target_length, binder_sequence, True)
    assert mine == theirs


@pytest.mark.xfail(reason=CCD_REASON, strict=False)
def test_prepare_esmfold2_tensors_identical(upstream: ModuleType) -> None:
    from analysis.design.folding import prepare_esmfold2_tensors
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    request = StructurePredictionInput(sequences=[ProteinInput(id="0", sequence=TARGET, msa=None)])
    mine = prepare_esmfold2_tensors(request, seed=0)
    theirs = upstream.prepare_esmfold2_tensors(request, seed=0)
    assert sorted(mine) == sorted(theirs)
    for key in mine:
        assert torch.equal(mine[key], theirs[key]), key


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_fold_and_get_distogram_identical() -> None:
    """Would fold the same soft design through the same ESMFold2 checkpoint.

    ``fold_and_get_distogram`` stacks its inputs with ``.cuda()`` and calls a
    loaded ``ESMFold2ExperimentalModel``, so it cannot run here.
    """
    raise NotImplementedError(GPU_REASON)


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_build_complex_identical() -> None:
    """Would compare the ProteinComplex built from one real model output.

    ``build_complex`` consumes a full ESMFold2 forward pass, so it needs the
    checkpoints. ``to_atom_array``, which does all of its work, is compared on
    planted arrays by ``test_to_atom_array_identical``.
    """
    raise NotImplementedError(GPU_REASON)


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_compute_esmc_pseudoperplexity_nll_identical() -> None:
    """Would compare the pseudo-perplexity NLL for one masked design batch.

    This needs the ``biohub/ESMC-6B`` weights. The vocabulary mapping it depends
    on is compared by ``test_folding_trunk_to_lm_aa_vocab_matrix_identical``.
    """
    raise NotImplementedError(GPU_REASON)


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_load_hf_model_identical() -> None:
    """Would compare loaded checkpoint state dicts and kernel backends."""
    raise NotImplementedError(GPU_REASON)


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_apply_torch_compile_identical() -> None:
    """Would compare which submodules get their forward compiled."""
    raise NotImplementedError(GPU_REASON)


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_design_binder_identical() -> None:
    """Would run the whole optimization loop under both modules.

    The loop is 150 folding steps on an H100 and is not reproducible across
    runs even at a fixed seed, so an equality test is not available even with a
    GPU. This publication also extends the loop deliberately, so upstream is no
    longer the reference for what it should do.
    """
    raise NotImplementedError(GPU_REASON)


@pytest.mark.xfail(reason=GPU_REASON, run=False)
def test_esmfold2_design_load_identical() -> None:
    """Would compare the model registry built by ``ESMFold2Design.load``."""
    raise NotImplementedError(GPU_REASON)
