"""Proof that :mod:`analysis.steering` reproduces the pipeline it came from.

Every secondary-structure number quoted in this publication was produced by a
separate private pipeline. The terms were transcribed into
:mod:`analysis.steering` so that this repository holds one implementation of
the whole method, which means the transcription itself is a claim that needs
checking: a mis-copied gate or an off-by-one register would leave regenerated
campaigns disagreeing with the publication's own narrative, silently.

So this module reads the term definitions out of the pipeline's git history,
compiles just those definitions into a private namespace, and runs both
implementations on the same inputs. It compiles the definitions rather than
importing the package, because the package pulls in configuration this
repository has no business depending on.

The pipeline is not public, so this is skipped unless ``ESMFOLD2_PIPELINE_REPO``
points at a checkout. The two implementations differ only in that the ported
one drops the pipeline's optional torch-module injection and its guards for
binder lengths this publication never uses, so agreement is expected to be
exact rather than approximate.
"""

from __future__ import annotations
import ast
import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from analysis import steering

PIPELINE_ENV = "ESMFOLD2_PIPELINE_REPO"
PIPELINE_REF = "ss-steering-pub"
PIPELINE_BLOB = "src/esmfold2_pipeline/design/losses.py"

PORTED_FUNCTIONS = {
    "helix_penalty": "compute_helix_penalty_loss",
    "helix_log": "compute_helix_log_loss",
    "helix_logcap": "compute_helix_logcap_loss",
    "helix_i2": "compute_helix_i2_loss",
    "sheet_contact": "compute_sheet_contact_loss",
    "sheet_mean": "compute_sheet_mean_loss",
    "sheet_ladder": "compute_sheet_ladder_loss",
}

ORIGINAL_TERMS = frozenset({"helix_mean"})
"""Terms written in this repository rather than transcribed from the pipeline.

These have no counterpart to check against, so they are listed here instead.
Membership is deliberately explicit: a term that is neither ported nor listed
here fails :func:`test_every_registered_term_is_covered`, which is what stops a
new transcription from arriving unverified.
"""

SUPPORTING_FUNCTIONS = (
    "_torch",
    "get_mid_points",
    "_contact_probability_matrix",
    "helix_penalty_residue_scores",
    "sheet_contact_residue_scores",
    "sheet_ladder_residue_scores",
)

CONSTANT_PREFIXES = ("HELIX_", "SHEET_")

SHAPES = ((1, 80, 40), (2, 120, 96), (1, 40, 16))


def _pipeline_root() -> Path:
    location = os.environ.get(PIPELINE_ENV)
    if not location:
        pytest.skip(
            f"set {PIPELINE_ENV} to a checkout of the design pipeline to check "
            f"the steering terms against their source"
        )
    root = Path(location)
    if not (root / ".git").exists():
        pytest.skip(f"{PIPELINE_ENV}={root} is not a git checkout")
    return root


@pytest.fixture(scope="module")
def pipeline() -> ModuleType:
    """The pipeline's steering terms, compiled from git without their package."""
    root = _pipeline_root()
    source = subprocess.run(
        ["git", "-C", str(root), "show", f"{PIPELINE_REF}:{PIPELINE_BLOB}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    wanted = set(PORTED_FUNCTIONS.values()) | set(SUPPORTING_FUNCTIONS)
    keep: list[ast.stmt] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and all(n.startswith(CONSTANT_PREFIXES) for n in names):
                keep.append(node)

    found = {node.name for node in keep if isinstance(node, ast.FunctionDef)}
    assert wanted <= found, f"not in {PIPELINE_REF}:{PIPELINE_BLOB}: {sorted(wanted - found)}"

    module = ModuleType("_pipeline_steering_terms")
    module.Any = object
    code = compile(
        ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])),
        f"<{PIPELINE_REF}:{PIPELINE_BLOB}>",
        "exec",
    )
    exec(code, module.__dict__)
    return module


def _symmetric_distogram(
    batch: int, length: int, generator: torch.Generator
) -> torch.Tensor:
    logits = torch.randn(
        batch, length, length, 128, dtype=torch.float64, generator=generator
    )
    return logits + logits.transpose(1, 2)


def _saturated_helix_distogram(length: int, bins: torch.Tensor) -> torch.Tensor:
    """A distogram sharp on an ideal helix, which drives the barrier to its cap."""
    index = np.arange(length, dtype=np.float64)
    angle = np.deg2rad(100.0) * index
    coords = np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * index], axis=1)
    midpoints = bins.numpy()
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    logits = -8.0 * (midpoints[None, None, :] - distances[..., None]) ** 2
    return torch.tensor(logits, dtype=torch.float64).unsqueeze(0)


@pytest.mark.skipif(
    importlib.util.find_spec("esm") is None,
    reason="needs esm, which analysis.design.losses imports",
)
def test_pipeline_bin_layout_matches_the_cookbook(pipeline: ModuleType) -> None:
    from analysis.design.losses import get_mid_points

    assert torch.equal(pipeline.get_mid_points(torch), get_mid_points())


def test_pipeline_constants_match(pipeline: ModuleType) -> None:
    for name in dir(steering):
        if not name.startswith(CONSTANT_PREFIXES) or name.endswith("FAMILY"):
            continue
        assert hasattr(pipeline, name), name
        assert getattr(steering, name) == getattr(pipeline, name), name


@pytest.mark.parametrize(("batch", "length", "binder_length"), SHAPES)
@pytest.mark.parametrize("term", sorted(PORTED_FUNCTIONS))
def test_term_matches_the_pipeline_on_random_distograms(
    pipeline: ModuleType, term: str, batch: int, length: int, binder_length: int
) -> None:
    generator = torch.Generator().manual_seed(11 + length)
    logits = _symmetric_distogram(batch, length, generator)
    bins = pipeline.get_mid_points(torch).to(dtype=logits.dtype)
    theirs = getattr(pipeline, PORTED_FUNCTIONS[term])(
        logits, binder_length, bins, torch_module=torch
    )
    ours = steering.TERM_SPECS[term].compute(logits, binder_length, bins)
    assert torch.equal(theirs, ours)


@pytest.mark.parametrize("term", sorted(PORTED_FUNCTIONS))
def test_term_matches_the_pipeline_on_a_saturated_helix(
    pipeline: ModuleType, term: str
) -> None:
    """The interesting case: the capped barrier is at its clamp here."""
    bins = pipeline.get_mid_points(torch).to(dtype=torch.float64)
    logits = _saturated_helix_distogram(60, bins)
    theirs = getattr(pipeline, PORTED_FUNCTIONS[term])(
        logits, 60, bins, torch_module=torch
    )
    ours = steering.TERM_SPECS[term].compute(logits, 60, bins)
    assert torch.equal(theirs, ours)
    if term == "helix_logcap":
        assert float(ours[0]) == pytest.approx(steering.HELIX_LOG_CAP)


def test_every_registered_term_is_covered() -> None:
    assert not set(PORTED_FUNCTIONS) & ORIGINAL_TERMS
    assert set(PORTED_FUNCTIONS) | ORIGINAL_TERMS == set(steering.TERM_KEYS)
