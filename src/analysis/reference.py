"""The natural binder reference distribution, loaded or fetched.

Question 1 asks how alpha-rich ESMFold2's unsteered designs are. That needs a
comparator, and the right one is not a standalone protein. A designed binder's
job is to bind something else, so the baseline is the small chain of a natural
heterodimer: a protein that is known to bind a partner, in a solved structure of
that complex.

Unlike a design campaign, this set is reproducible. It comes from a single RCSB
Search API query and public coordinates, with no GPU and no stochastic
optimization, so rebuilding it is legitimate. It is merely slow, on the order of
tens of minutes to download and measure a few thousand structures. So the
loading order is:

1. Read the table from ``data/reference`` if it is already there.
2. Otherwise download it from the deposit, which is what a reader gets.
3. Rebuild it from RCSB only when asked explicitly, via ``build=True``.

The notebook uses step 1 or 2 and never step 3.
"""

from __future__ import annotations
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import paths

COMPOSITION_TABLE = "natural_binder_composition.csv"
PAIRS_TABLE = "natural_binder_pairs.csv"
QUERY_RECORD = "natural_binder_query.json"

DEPOSIT_BASE = "https://zenodo.org/records/PENDING/files"

BUILD_COMMAND = (
    "replay the RCSB Search API query recorded in "
    f"data/reference/{QUERY_RECORD}"
)

BINDER_LENGTH_MIN = 60
BINDER_LENGTH_MAX = 200


def _deposit_url(name: str) -> str:
    return f"{DEPOSIT_BASE}/{name}?download=1"


def fetch(name: str) -> Path:
    """Download one reference table from the deposit into ``data/reference``.

    Returns the local path. Raises with the deposit URL and the rebuild command
    if the download fails, which it will until the deposit exists.
    """
    paths.REFERENCE.mkdir(parents=True, exist_ok=True)
    destination = paths.reference_table(name)
    url = _deposit_url(name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise FileNotFoundError(
            f"Could not fetch {name} from the deposit.\n"
            f"  URL:   {url}\n"
            f"  Local: {destination}\n"
            f"  Cause: {error}\n"
            f"Either place the file at the local path, or rebuild it from RCSB "
            f"with:\n    {BUILD_COMMAND}\n"
            f"Rebuilding needs no GPU but downloads and measures several "
            f"thousand structures, so budget tens of minutes."
        ) from error
    destination.write_bytes(payload)
    return destination


def table(name: str, build: bool = False) -> pd.DataFrame:
    """Load one reference table, fetching from the deposit if it is absent.

    Set ``build`` to rebuild from RCSB instead of fetching. That path is never
    taken by the notebook.
    """
    local = paths.reference_table(name)
    if local.is_file():
        return pd.read_csv(local)
    if build:
        raise NotImplementedError(
            f"Rebuild is a separate script rather than an import, to keep the "
            f"notebook off long-running network work. Run:\n    {BUILD_COMMAND}\n"
            f"then re-read {local}"
        )
    return pd.read_csv(fetch(name))


def composition(build: bool = False) -> pd.DataFrame:
    """Per-binder secondary-structure composition for the natural set."""
    return table(COMPOSITION_TABLE, build=build)


def pairs(build: bool = False) -> pd.DataFrame:
    """One row per natural binder-target pair, with covariates."""
    return table(PAIRS_TABLE, build=build)


def is_available() -> bool:
    """True when the composition table is already on disk, no network needed."""
    return paths.reference_table(COMPOSITION_TABLE).is_file()


def status() -> pd.DataFrame:
    """Which reference tables are present locally."""
    rows = [
        {
            "table": name,
            "present (bool)": paths.reference_table(name).is_file(),
            "path": str(paths.reference_table(name)),
        }
        for name in (COMPOSITION_TABLE, PAIRS_TABLE, QUERY_RECORD)
    ]
    return pd.DataFrame(rows)


def length_matched(
    natural: pd.DataFrame,
    design_lengths: pd.Series,
    length_column: str,
    bin_width: int = 10,
    draws: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Resample the natural set to match the design length distribution.

    Short natural proteins are helix-biased too, so a pooled comparison of
    means confounds composition with length. This draws natural chains inside
    each length bin in proportion to how many designs fall in that bin.
    """
    rng = np.random.default_rng(seed)
    edges = range(BINDER_LENGTH_MIN, BINDER_LENGTH_MAX + bin_width, bin_width)
    natural_bin = pd.cut(natural[length_column], bins=list(edges), right=False)
    design_bin = pd.cut(design_lengths, bins=list(edges), right=False)
    weights = design_bin.value_counts(normalize=True)

    picks: list[int] = []
    grouped = {
        key: frame.index.to_numpy()
        for key, frame in natural.groupby(natural_bin, observed=True)
    }
    for key, weight in weights.items():
        pool = grouped.get(key)
        if pool is None or len(pool) == 0:
            continue
        n = int(round(draws * float(weight)))
        if n == 0:
            continue
        picks.extend(rng.choice(pool, size=n, replace=True).tolist())
    return natural.loc[picks].reset_index(drop=True)
