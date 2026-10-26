"""Artifact paths resolved from the repository root."""

from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data"
TARGETS = DATA / "targets"
CAMPAIGNS = DATA / "campaigns"

ASSETS = REPO_ROOT / "assets"
"""Small tracked files a notebook cell needs at execution time.

Static display inputs such as the illustrative complex. Scientific model
outputs live under the ignored data directories; target inputs are tracked.
"""
DERIVED = DATA / "derived"
REFERENCE = DATA / "reference"


def reference_table(name: str) -> Path:
    """Path to a preserved reference input."""
    return REFERENCE / name


def campaign_dir(target_key: str, arm: str) -> Path:
    """Directory holding one campaign, one target crossed with one loss arm."""
    return CAMPAIGNS / f"{target_key}__{arm}"


def campaign_results(target_key: str, arm: str) -> Path:
    """Per-design results table for one campaign."""
    return campaign_dir(target_key, arm) / "all_results.parquet"


def campaign_manifest(target_key: str, arm: str) -> Path:
    """Modal call manifest for one campaign, written at spawn time."""
    return campaign_dir(target_key, arm) / "manifest.csv"


def campaign_trace(target_key: str, arm: str) -> Path:
    """Preserved optimization diagnostics, including model scores at every step."""
    return campaign_dir(target_key, arm) / "step_trace.parquet"


def campaign_protocol(target_key: str, arm: str) -> Path:
    """Recorded settings for a design campaign."""
    return campaign_dir(target_key, arm) / "protocol.json"


def campaign_complexes(target_key: str, arm: str) -> Path:
    """Predicted complex structures for one campaign."""
    return campaign_dir(target_key, arm) / "all_complexes"


def campaign_step_complexes(target_key: str, arm: str, seed: int) -> Path:
    """Per-step predicted complexes for one design, when they were logged.

    One directory per seed, because a single design contributes as many
    structures as it has optimization steps and they would otherwise sit
    alongside the one final complex per seed in ``all_complexes``.
    """
    return campaign_dir(target_key, arm) / "step_complexes" / f"seed{seed:02d}"


def ensure_dirs() -> None:
    """Create every output directory the notebook writes into."""
    for path in (CAMPAIGNS, DERIVED, REFERENCE):
        path.mkdir(parents=True, exist_ok=True)
