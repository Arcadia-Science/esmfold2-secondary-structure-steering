"""Proof that the one extended cookbook definition is extended and no more.

Every other definition in :mod:`analysis.design` is byte-for-byte the ESM
cookbook's, which ``tests/test_design_port.py`` establishes by syntax-tree
identity. ``design_binder`` is the exception: it carries the secondary-structure
steering terms, the per-step binding readout, and the per-step loss log that
this publication is about.

An exception like that is where a port quietly stops being a port, so the
divergence is pinned rather than described. ``DESIGN_BINDER_DIVERGENCES`` names
every upstream statement that changed, what it changed into, and why. The test
fails if the real divergence is larger, smaller, or different, and a second test
requires every inserted statement to name something the extension introduced,
so an unrelated edit cannot hide among the additions.

Five upstream statements change. Four serve the steering and the loss log, and
the fifth adds per-chain pLDDT to the critic record.

This reads the ported source from disk instead of importing it, so the pin holds
on a machine with no ``esm`` and no GPU. Upstream still comes from git, so
``ESM_REPO`` has to point at a checkout of https://github.com/Biohub/esm.
"""

from __future__ import annotations
import ast
import copy
import os
import subprocess
from pathlib import Path

import pytest

UPSTREAM_COMMIT = "ba4d712"
UPSTREAM_BLOB = "cookbook/tutorials/binder_design.py"

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTED_SOURCE = REPO_ROOT / "src" / "analysis" / "design" / "loop.py"

EXTENDED_DEFINITION = "design_binder"

DESIGN_BINDER_DIVERGENCES = (
    (
        "def design_binder(inversion_models: dict[str, ESMFold2ExperimentalModel], "
        "hf_critic_models: dict[str, ESMFold2ExperimentalModel], "
        "esmc_model: ESMCForMaskedLM, target_name: str, "
        "target_sequence: str | None, binder_name: str, "
        "binder_sequence: str | None, is_antibody: bool | None, seed: int, "
        "batch_size: int=1) -> tuple[list[str], Trajectory, list[dict]]:\n    pass",
        "def design_binder(inversion_models: dict[str, ESMFold2ExperimentalModel], "
        "hf_critic_models: dict[str, ESMFold2ExperimentalModel], "
        "esmc_model: ESMCForMaskedLM, target_name: str, "
        "target_sequence: str | None, binder_name: str, "
        "binder_sequence: str | None, is_antibody: bool | None, seed: int, "
        "batch_size: int=1, schedule: LossSchedule=VANILLA, "
        "trace_path: Path | None=None) -> tuple[list[str], Trajectory, "
        "list[dict], tuple[loss_trace.StepRecord, ...]]:\n    pass",
        "two keyword parameters with defaults that reproduce upstream behaviour, "
        "and a fourth return value carrying the loss log",
    ),
    (
        "inversion_model = list(inversion_models.values())[replicate_choice]",
        "inversion_name = list(inversion_models)[replicate_choice]",
        "split in two so the chosen replicate can be named in the loss log; the "
        "random draw itself is untouched, so the same model is chosen",
    ),
    (
        "structure_loss = losses['total_loss']",
        "structure_loss = losses['total_loss'] + steering_losses.total",
        "the steering terms enter the objective here, and only here; with the "
        "default schedule the added total is a zero with no gradient",
    ),
    (
        "return (best_sequences, trajectory, critic_results)",
        "return (best_sequences, trajectory, critic_results, tuple(trace))",
        "the loss log is returned alongside the trajectory",
    ),
    (
        "critic_results.append({'is_antibody': is_antibody, 'critic_name': "
        "critic_name, 'batch_idx': batch_idx, 'designed_sequence': best_seq, "
        "'complex': pred_complex, 'final_loss': "
        "final_total_loss[batch_idx].item(), 'iptm': iptm, 'logits': "
        "logits[batch_idx].detach().cpu(), **iptm_proxy_scores})",
        "critic_results.append({'is_antibody': is_antibody, 'critic_name': "
        "critic_name, 'batch_idx': batch_idx, 'designed_sequence': best_seq, "
        "'complex': pred_complex, 'final_loss': "
        "final_total_loss[batch_idx].item(), 'iptm': iptm, 'plddt_complex': "
        "chain_plddt.complex_plddt, 'plddt_target': chain_plddt.target_plddt, "
        "'plddt_binder': chain_plddt.binder_plddt, 'logits': "
        "logits[batch_idx].detach().cpu(), **iptm_proxy_scores})",
        "three per-chain pLDDT means join the critic record. The cookbook "
        "writes per-atom pLDDT into the B-factor column of the complex but "
        "reports no per-chain number, and the viability rule needs the "
        "binder's separately from the complex it sits in",
    ),
)

EXTENSION_NAMES = frozenset(
    {
        "bin_distance",
        "binder_seq",
        "chain_plddt",
        "confidence",
        "inversion_name",
        "logged_losses",
        "loss_trace",
        "no_grad",
        "proxy",
        "record",
        "schedule",
        "steering_loss",
        "steering_losses",
        "step_iptm",
        "structure_loss",
        "trace",
        "trace_path",
        "weights",
    }
)


def _upstream_source() -> str:
    repo = os.environ.get("ESM_REPO", "")
    if not repo:
        pytest.skip(
            "set ESM_REPO to a checkout of https://github.com/Biohub/esm to pin "
            "the design_binder divergence against upstream"
        )
    completed = subprocess.run(
        ["git", "-C", repo, "show", f"{UPSTREAM_COMMIT}:{UPSTREAM_BLOB}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(
            f"cannot read {UPSTREAM_BLOB} at {UPSTREAM_COMMIT} from {repo}. "
            f"git said: {completed.stderr.strip()}"
        )
    return completed.stdout


@pytest.fixture(scope="module")
def upstream_source() -> str:
    return _upstream_source()


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


def _definition_node(source: str, name: str) -> ast.stmt:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name:
            return _strip_docstrings(node)
    raise AssertionError(f"no top-level definition named {name!r}")


def _statement_key(node: ast.stmt) -> str:
    """One statement as source, with any nested body replaced by ``pass``.

    Comparing these makes the check statement-by-statement rather than
    definition-by-definition, so an inserted line is distinguishable from a
    changed one.
    """
    clone = copy.deepcopy(node)
    for field in ("body", "orelse", "finalbody", "handlers"):
        if hasattr(clone, field):
            setattr(clone, field, [ast.Pass()] if field == "body" else [])
    if isinstance(clone, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        clone.decorator_list = []
    return ast.unparse(ast.fix_missing_locations(clone))


def _statements(definition: ast.stmt) -> list[str]:
    return [
        _statement_key(node)
        for node in ast.walk(definition)
        if isinstance(node, ast.stmt)
    ]

def test_design_binder_diverges_exactly_as_documented(upstream_source: str) -> None:
    """The extension to ``design_binder`` is the documented one and no larger."""
    upstream = _statements(_definition_node(upstream_source, EXTENDED_DEFINITION))
    ported = _statements(
        _definition_node(PORTED_SOURCE.read_text(), EXTENDED_DEFINITION)
    )

    changed = sorted(set(upstream) - set(ported))
    assert changed == sorted(statement for statement, _, _ in DESIGN_BINDER_DIVERGENCES)

    for _, replacement, reason in DESIGN_BINDER_DIVERGENCES:
        assert replacement in ported, reason


def test_design_binder_insertions_only_touch_the_extension(
    upstream_source: str,
) -> None:
    """Every added statement names something the extension introduced."""
    upstream = _statements(_definition_node(upstream_source, EXTENDED_DEFINITION))
    ported = _statements(
        _definition_node(PORTED_SOURCE.read_text(), EXTENDED_DEFINITION)
    )
    replacements = {replacement for _, replacement, _ in DESIGN_BINDER_DIVERGENCES}

    for statement in set(ported) - set(upstream) - replacements:
        assert any(name in statement for name in EXTENSION_NAMES), statement
