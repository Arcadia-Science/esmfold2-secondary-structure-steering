"""Inspect and load preserved campaign artifacts without running design models."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis import deposit, paths
from analysis.targets import ARM_SPECS, TARGET_SPECS


def expected_seeds(target_key: str, arm: str) -> set[int]:
    """Read the requested seed coverage, including seeds not yet submitted."""
    protocol = json.loads(paths.campaign_protocol(target_key, arm).read_text())
    count = protocol["n_seeds"]
    if type(count) is not int or count < 1:
        raise ValueError("Campaign n_seeds must be a positive integer")
    return set(range(count))


def is_complete(target_key: str, arm: str) -> bool:
    """Require every requested seed's scores, structure, and trace locally.

    Older archives need no collection summary: their protocol and artifacts
    establish coverage. Availability of a partial campaign is a separate check.
    """
    directory = paths.campaign_dir(target_key, arm)
    if (directory / ".collecting").exists():
        return False
    try:
        expected = expected_seeds(target_key, arm)
        summary_path = directory / "collection.json"
        if summary_path.exists() and json.loads(summary_path.read_text()).get("partial", True):
            return False
        scores = pd.read_parquet(paths.campaign_results(target_key, arm), columns=["seed"])
        trace = pd.read_parquet(paths.campaign_trace(target_key, arm), columns=["seed"])
        structures = {p.name for p in paths.campaign_complexes(target_key, arm).glob("seed*.pdb")}
        return (
            not scores.seed.duplicated().any()
            and set(scores.seed) == expected
            and set(trace.seed) == expected
            and structures == {f"seed{seed:02d}.pdb" for seed in expected}
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


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
                    "complete (bool)": is_complete(key, name),
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
