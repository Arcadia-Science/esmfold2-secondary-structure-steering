"""Repository-relative paths for every artifact the notebook reads or writes."""

from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data"
TARGETS = DATA / "targets"
CAMPAIGNS = DATA / "campaigns"

ASSETS = REPO_ROOT / "assets"
"""Small tracked files a notebook cell needs at execution time.

Distinct from ``data``, which is generated, gitignored and fetched from the
deposit. An asset is committed because a cell cannot run without it and it is
too small to be worth depositing.
"""
DERIVED = DATA / "derived"
REFERENCE = DATA / "reference"


def reference_table(name: str) -> Path:
    """Path to one deposited reference table."""
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


def campaign_composition(target_key: str, arm: str) -> Path:
    """Compact per-design composition and confidence table for one campaign.

    This is the only campaign artifact the notebook needs, and it is small
    enough to track in git. The full results table and the predicted
    structures stay in the deposit.
    """
    return campaign_dir(target_key, arm) / "composition.parquet"


def campaign_trace(target_key: str, arm: str) -> Path:
    """Per-step loss log for one campaign, every seed concatenated.

    One row per optimization step per design: every loss term, the steering
    weight each term carried at that step, the per-step binding readout and the
    sequence entropy. At 30 seeds and 150 steps this is 4,500 rows and about
    450 kB, so it is tracked in git rather than left to the deposit. Most of
    that size is the designed sequence recorded at every step, which is what
    makes it possible to see when the optimizer stops moving.
    """
    return campaign_dir(target_key, arm) / "step_trace.parquet"


def campaign_summary(target_key: str, arm: str) -> Path:
    """Aggregate summary for one campaign."""
    return campaign_dir(target_key, arm) / "summary.json"


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
