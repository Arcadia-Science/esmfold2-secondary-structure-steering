"""Per-design composition tables, the one campaign artifact the notebook needs.

A campaign's raw results table carries confidence scores but no secondary
structure, because composition has to be measured from the predicted
coordinates. Doing that in the notebook would mean shipping tens of megabytes
of structures per campaign, so it is measured once at collection time and
stored as a small table beside the campaign.

:func:`build` produces that table from predicted complexes. :func:`load` reads
it and never computes, so the notebook cannot accidentally depend on
structures it does not have.

:func:`viable` applies this publication's viability rule to such a table:
helix below 20%, sheet at least 15%, interface pTM at least 0.7, and binder
pLDDT at least 70. All four thresholds are named constants here rather than
numbers in a notebook cell, so a figure and a sentence cannot disagree about
what counts as viable.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

from analysis import paths, secondary_structure

SCORE_COLUMNS = {
    "critic_iptm": "iptm (0-1)",
    "critic_plddt_binder": "binder pLDDT (0-100)",
}

MAX_HELIX_FRACTION = 0.20
MIN_SHEET_FRACTION = 0.15
MIN_IPTM = 0.70
MIN_BINDER_PLDDT = 70.0

VIABILITY_COLUMNS = (
    "helix (fraction 0-1)",
    "sheet (fraction 0-1)",
    *SCORE_COLUMNS.values(),
)


def build(target_key: str, arm: str, complexes: Path, results: Path) -> pd.DataFrame:
    """Measure every predicted complex and join to the campaign's confidence scores.

    ``complexes`` holds ``seedNN.pdb`` files with the target as chain A and the
    binder as chain B. ``results`` is the campaign's per-design table.
    """
    frame = pd.read_parquet(results) if results.suffix == ".parquet" else pd.read_csv(results)
    rows = []
    for pdb in sorted(complexes.glob("seed*.pdb")):
        seed = int(pdb.stem.replace("seed", ""))
        comp = secondary_structure.binder_composition(pdb)
        rows.append(
            {
                "seed": seed,
                "binder length (residues)": comp.n_residues,
                "helix (fraction 0-1)": comp.helix_fraction,
                "sheet (fraction 0-1)": comp.sheet_fraction,
                "coil (fraction 0-1)": comp.coil_fraction,
            }
        )
    measured = pd.DataFrame(rows)

    absent = [column for column in (*SCORE_COLUMNS, "seed") if column not in frame.columns]
    if absent:
        raise KeyError(
            f"results table {results} is missing {absent}. Every column the "
            f"viability rule needs is returned by analysis.modal_app.design, so "
            f"a missing one means the deployed app predates it. Redeploy with:\n"
            f"    modal deploy src/analysis/modal_app.py"
        )
    scores = frame[["seed", *SCORE_COLUMNS]].rename(columns=SCORE_COLUMNS)
    measured = measured.merge(scores, on="seed", how="left")

    return measured.assign(target=target_key, arm=arm).sort_values("seed").reset_index(drop=True)


def viable(frame: pd.DataFrame) -> pd.Series:
    """Which designs clear every viability threshold at once.

    The rule is a conjunction on purpose. Steering that drives helix down while
    destroying the interface is not a result, so fold composition and
    confidence have to be read together rather than traded off by eye. The
    confidence half is ESMFold2 scoring its own output, which is a limitation
    rather than ground truth: a low score can mean a worse design or a scorer
    less confident about the fold class the design landed in.
    """
    absent = [column for column in VIABILITY_COLUMNS if column not in frame.columns]
    if absent:
        raise KeyError(
            f"cannot judge viability without {absent}; the frame has {sorted(frame.columns)}"
        )
    return (
        frame["helix (fraction 0-1)"].lt(MAX_HELIX_FRACTION)
        & frame["sheet (fraction 0-1)"].ge(MIN_SHEET_FRACTION)
        & frame["iptm (0-1)"].ge(MIN_IPTM)
        & frame["binder pLDDT (0-100)"].ge(MIN_BINDER_PLDDT)
    )


def write(target_key: str, arm: str, frame: pd.DataFrame) -> Path:
    """Persist a composition table beside its campaign."""
    destination = paths.campaign_composition(target_key, arm)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


def is_available(target_key: str, arm: str) -> bool:
    """True when the composition table exists, no network or structures needed."""
    return paths.campaign_composition(target_key, arm).is_file()


def load(target_key: str, arm: str) -> pd.DataFrame:
    """Read one campaign's composition table, or raise saying how to make it."""
    path = paths.campaign_composition(target_key, arm)
    if not path.is_file():
        raise FileNotFoundError(
            f"No composition table for {target_key!r} / {arm!r}.\n"
            f"Expected: {path}\n"
            f"It is written when a campaign is collected. To regenerate the "
            f"campaign:\n"
            f"    python -m analysis.campaign run --targets {target_key} --arms {arm}"
        )
    return pd.read_parquet(path)


def load_all(target_keys: list[str], arms: list[str]) -> pd.DataFrame:
    """Every available composition table concatenated, skipping absent ones."""
    frames = [load(key, name) for key in target_keys for name in arms if is_available(key, name)]
    if not frames:
        raise FileNotFoundError(f"No campaign composition tables under {paths.CAMPAIGNS}.")
    return pd.concat(frames, ignore_index=True)


def available(target_keys: list[str], arms: list[str]) -> list[tuple[str, str]]:
    """Which (target, arm) pairs have a composition table."""
    return [(key, name) for key in target_keys for name in arms if is_available(key, name)]
