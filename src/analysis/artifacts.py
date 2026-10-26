"""Inspect and load preserved campaign artifacts without running design models."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis import deposit, paths
from analysis.targets import ARM_SPECS, TARGET_SPECS


@dataclass(frozen=True)
class CampaignStatus:
    """Which artifacts exist on disk for one target crossed with one arm."""

    target_key: str
    arm: str
    has_results: bool
    has_manifest: bool
    has_complexes: bool
    has_trace: bool
    n_complexes: int

    @property
    def is_analysable(self) -> bool:
        """Whether model scores and coordinates are available for measurement."""
        return self.has_results and self.has_complexes

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
        has_manifest=paths.campaign_manifest(target_key, arm).is_file(),
        has_complexes=n > 0,
        has_trace=paths.campaign_trace(target_key, arm).is_file(),
        n_complexes=n,
    )


def status_table(
    target_keys: list[str] | None = None,
    arms: list[str] | None = None,
) -> pd.DataFrame:
    """Report artifact availability for each requested campaign."""
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
    """Return target–arm pairs missing a results table or predicted complexes."""
    keys = target_keys if target_keys is not None else list(TARGET_SPECS)
    names = arms if arms is not None else list(ARM_SPECS)
    return [(key, name) for key in keys for name in names if not status(key, name).is_analysable]


def load_campaign(target_key: str, arm: str) -> pd.DataFrame:
    """Load campaign scores, fetching the input archive if the table is missing."""
    results = paths.campaign_results(target_key, arm)
    if not results.is_file():
        deposit.DESIGN_TABLES.fetch()
    frame = pd.read_parquet(results)
    return frame.assign(target=target_key, arm=arm)


def load_all(
    target_keys: list[str] | None = None,
    arms: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate locally analysable campaigns, skipping missing ones.

    Raise FileNotFoundError if none are available.
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
            "Fetch the published inputs with analysis.deposit.DESIGN_TABLES.fetch()."
        )
    return pd.concat(frames, ignore_index=True)


def derived(name: str) -> Path:
    """Path for a derived table the notebook computes and caches."""
    paths.DERIVED.mkdir(parents=True, exist_ok=True)
    return paths.DERIVED / name
