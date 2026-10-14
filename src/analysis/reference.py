"""Recalculate natural binder measurements from preserved reference inputs."""

from __future__ import annotations
import json

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


def recorded_query_settings() -> dict:
    """Read scientific settings from the archived query, without a live search."""
    from analysis import rcsb

    query = json.loads(paths.reference_table(QUERY_RECORD).read_text())
    criteria = {
        node["parameters"]["attribute"]: node["parameters"].get("value")
        for node in query["query"]["nodes"]
    }
    return {
        "Resolution limit (Å)": criteria["rcsb_entry_info.resolution_combined"],
        "Released by": criteria["rcsb_accession_info.initial_release_date"],
        "Sequence identity (%)": query["request_options"]["group_by"]["similarity_cutoff"],
        "Length range (residues)": criteria["entity_poly.rcsb_sample_sequence_length"],
        "Minimum resolved backbone fraction": rcsb.MIN_OBSERVED_FRACTION,
    }
