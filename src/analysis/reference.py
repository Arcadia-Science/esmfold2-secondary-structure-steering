"""Recalculate natural binder measurements from preserved reference inputs."""

from __future__ import annotations

import pandas as pd

from analysis import paths
from analysis.parallel import DEFAULT_WORKERS, validate_workers

COMPOSITION_TABLE = "natural_binder_composition.csv"
PAIRS_TABLE = "natural_binder_pairs.csv"
QUERY_RECORD = "natural_binder_query.json"
BINDER_LENGTH_MIN = 60
BINDER_LENGTH_MAX = 200


def table(name: str, *, workers: int = DEFAULT_WORKERS) -> pd.DataFrame:
    """Measure the requested reference table in memory; ignore old CSV caches."""
    from analysis import deposit, rcsb

    validate_workers(workers)
    if name not in (COMPOSITION_TABLE, PAIRS_TABLE):
        raise ValueError(f"Unknown reference table: {name}")
    if not paths.reference_table(rcsb.SEARCH_HITS).is_file():
        deposit.REFERENCE_INPUTS.fetch()
    return rcsb.build(include_excluded=name == PAIRS_TABLE, workers=workers)


def composition(*, workers: int = DEFAULT_WORKERS) -> pd.DataFrame:
    """Measure and select the natural binder reference."""
    return table(COMPOSITION_TABLE, workers=workers)


def pairs(*, workers: int = DEFAULT_WORKERS) -> pd.DataFrame:
    """Measure every candidate pair, retaining selection flags."""
    return table(PAIRS_TABLE, workers=workers)
