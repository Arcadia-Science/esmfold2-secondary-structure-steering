"""The natural binder reference distribution, loaded or fetched.

Question 1 asks how alpha-rich ESMFold2's unsteered designs are. That needs a
comparator, and the right one is not a standalone protein. A designed binder's
job is to bind something else, so the baseline is the small chain of a natural
heterodimer: a protein that is known to bind a partner, in a solved structure of
that complex.

Unlike a design campaign, this set is reproducible. It comes from one RCSB
Search API query and public coordinates, with no GPU and no stochastic
optimization, so rebuilding it is legitimate rather than approximate. It is
merely slow, on the order of tens of minutes. So the loading order is:

1. Read the table from ``data/reference`` if it is there, which it is: the
   tables are tracked in this repository.
2. Otherwise rebuild it from the RCSB with :func:`analysis.rcsb.build`, which
   replays the recorded query against a pinned release-date snapshot and
   measures every structure through :mod:`analysis.secondary_structure`.

There is deliberately no third path through a deposit download. A set that can
be rebuilt from a recorded query should be rebuilt, not fetched, and building
it here rather than in a script is what keeps the reference set measured by the
same DSSP code as the designs.
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

BUILD_COMMAND = "python -c 'from analysis import rcsb; rcsb.build()'"

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


def table(name: str) -> pd.DataFrame:
    """Load one reference table, building it from the RCSB if it is absent.

    The tables are tracked in this repository, so the usual path is the first
    branch. The second exists so the set is reconstructible rather than merely
    downloadable: :func:`analysis.rcsb.build` runs the recorded query and
    measures every structure through :mod:`analysis.secondary_structure`, the
    same code that measures the designs.
    """
    local = paths.reference_table(name)
    if local.is_file():
        return pd.read_csv(local)

    from analysis import rcsb

    print(
        f"{local} is absent, so rebuilding the reference set from the RCSB. "
        f"This needs no GPU but downloads and measures a few thousand "
        f"structures, so budget tens of minutes."
    )
    rcsb.build()
    if not local.is_file():
        raise FileNotFoundError(
            f"the rebuild finished without writing {local}, which means {name!r} "
            f"is not one of the tables analysis.rcsb.build produces"
        )
    return pd.read_csv(local)


def composition() -> pd.DataFrame:
    """Per-binder secondary-structure composition for the natural set."""
    return table(COMPOSITION_TABLE)


def pairs() -> pd.DataFrame:
    """One row per natural binder-target pair, with covariates."""
    return table(PAIRS_TABLE)


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
