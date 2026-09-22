"""The file-exists gate that keeps the notebook off the GPU.

Executing this notebook must not launch design campaigns. A campaign is roughly
five GPU-hours on an H100 and costs real money, and the design loop is not
seed-reproducible: across two runs of an identically configured arm, not one of
thirty seeds reproduced its own result, with per-seed interface pTM moving 0.14
to 0.20. Re-running would therefore produce different designs, not the ones in
the figures.

So every campaign read goes through :func:`load_campaign`, which returns the
deposited table when it is present and never computes. Regenerating a campaign
from scratch is an explicit, separate act: see :mod:`analysis.campaign`.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis import paths
from analysis.targets import ARM_SPECS, TARGET_SPECS

DEPOSIT_URL = "https://doi.org/10.5281/zenodo.PENDING"


@dataclass(frozen=True)
class CampaignStatus:
    """Which artifacts exist on disk for one target crossed with one arm."""

    target_key: str
    arm: str
    has_results: bool
    has_summary: bool
    has_manifest: bool
    has_complexes: bool
    has_trace: bool
    n_complexes: int

    @property
    def is_analysable(self) -> bool:
        """True when the per-design table is present, which is all figures need."""
        return self.has_results

    @property
    def is_spawned(self) -> bool:
        """True when jobs were launched, whether or not results were collected."""
        return self.has_manifest


def status(target_key: str, arm: str) -> CampaignStatus:
    """Report which artifacts exist for one campaign, touching nothing."""
    complexes = paths.campaign_complexes(target_key, arm)
    n = len(list(complexes.glob("*.pdb"))) if complexes.is_dir() else 0
    return CampaignStatus(
        target_key=target_key,
        arm=arm,
        has_results=paths.campaign_results(target_key, arm).is_file(),
        has_summary=paths.campaign_summary(target_key, arm).is_file(),
        has_manifest=paths.campaign_manifest(target_key, arm).is_file(),
        has_complexes=n > 0,
        has_trace=paths.campaign_trace(target_key, arm).is_file(),
        n_complexes=n,
    )


def status_table(
    target_keys: list[str] | None = None,
    arms: list[str] | None = None,
) -> pd.DataFrame:
    """One row per campaign, showing what is on disk.

    Call this first in the notebook. It is the honest answer to "can this
    notebook produce its figures from what is here".
    """
    keys = target_keys if target_keys is not None else list(TARGET_SPECS)
    names = arms if arms is not None else list(ARM_SPECS)
    rows = []
    for key in keys:
        for name in names:
            s = status(key, name)
            rows.append(
                {
                    "target": key,
                    "arm": name,
                    "results present (bool)": s.has_results,
                    "summary present (bool)": s.has_summary,
                    "manifest present (bool)": s.has_manifest,
                    "predicted complexes (count)": s.n_complexes,
                    "loss log present (bool)": s.has_trace,
                    "analysable (bool)": s.is_analysable,
                }
            )
    return pd.DataFrame(rows)


def missing(
    target_keys: list[str] | None = None,
    arms: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Campaigns with no per-design table, as (target, arm) pairs."""
    keys = target_keys if target_keys is not None else list(TARGET_SPECS)
    names = arms if arms is not None else list(ARM_SPECS)
    return [(key, name) for key in keys for name in names if not status(key, name).is_analysable]


def load_campaign(target_key: str, arm: str) -> pd.DataFrame:
    """Per-design results for one campaign, read from disk.

    Raises if the table is absent rather than computing it, and the message
    says both how to fetch the deposit and how to regenerate.
    """
    results = paths.campaign_results(target_key, arm)
    if not results.is_file():
        raise FileNotFoundError(
            f"No results for target {target_key!r} arm {arm!r}.\n"
            f"Expected: {results}\n"
            f"Fetch the deposited artifacts from {DEPOSIT_URL}, or regenerate "
            f"this campaign on your own Modal account with:\n"
            f"    python -m analysis.campaign run --target {target_key} --arm {arm}\n"
            f"Regenerating costs GPU time and will not reproduce these exact "
            f"designs, because the design loop is not seed-reproducible."
        )
    frame = pd.read_parquet(results)
    return frame.assign(target=target_key, arm=arm)


def load_all(
    target_keys: list[str] | None = None,
    arms: list[str] | None = None,
) -> pd.DataFrame:
    """Every available campaign concatenated, skipping absent ones.

    Unlike :func:`load_campaign` this does not raise on absence, so a partial
    deposit still yields a figure. The set actually loaded is reported by
    :func:`status_table`.
    """
    keys = target_keys if target_keys is not None else list(TARGET_SPECS)
    names = arms if arms is not None else list(ARM_SPECS)
    frames = [
        load_campaign(key, name)
        for key in keys
        for name in names
        if status(key, name).is_analysable
    ]
    if not frames:
        raise FileNotFoundError(
            f"No campaign results found under {paths.CAMPAIGNS}.\n"
            f"Fetch the deposited artifacts from {DEPOSIT_URL}."
        )
    return pd.concat(frames, ignore_index=True)


def derived(name: str) -> Path:
    """Path for a derived table the notebook computes and caches."""
    paths.DERIVED.mkdir(parents=True, exist_ok=True)
    return paths.DERIVED / name
