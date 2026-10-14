"""Measure campaign composition from preserved predicted structures on every load.

Confidence scores are expensive model outputs; DSSP assignments, fractions and
screening summaries are derived in memory. No measurement cache is read or written.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import paths, secondary_structure

SCORE_COLUMNS = {
    "critic_iptm": "iptm (0-1)",
    "critic_plddt_binder": "binder pLDDT (0-100)",
}

MIN_SHEET_FRACTION = 0.40
MIN_IPTM = 0.85
MIN_BINDER_PLDDT = 80.0

VIABILITY_COLUMNS = (
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
    observed = [row["seed"] for row in rows]
    if frame.seed.duplicated().any() or sorted(observed) != sorted(frame.seed.tolist()):
        raise ValueError(f"Structures and score rows do not match: {target_key}/{arm}")

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
    """Which designs pass every composition and confidence threshold.

    This computational screen selects candidates for further evaluation;
    it does not establish that they fold or bind experimentally.
    """
    absent = [column for column in VIABILITY_COLUMNS if column not in frame.columns]
    if absent:
        raise KeyError(
            f"cannot judge viability without {absent}; the frame has {sorted(frame.columns)}"
        )
    return (
        frame["sheet (fraction 0-1)"].gt(MIN_SHEET_FRACTION)
        & frame["iptm (0-1)"].gt(MIN_IPTM)
        & frame["binder pLDDT (0-100)"].gt(MIN_BINDER_PLDDT)
    )


def is_available(target_key: str, arm: str) -> bool:
    """Whether saved model scores and predicted structures are present."""
    return paths.campaign_results(target_key, arm).is_file() and any(
        paths.campaign_complexes(target_key, arm).glob("seed*.pdb")
    )


def measure_campaign(target_key: str, arm: str) -> pd.DataFrame:
    """Recalculate composition from saved model outputs without running a model."""
    if not is_available(target_key, arm):
        from analysis import deposit

        deposit.DESIGN_TABLES.fetch()
    return build(
        target_key,
        arm,
        paths.campaign_complexes(target_key, arm),
        paths.campaign_results(target_key, arm),
    )


def load(target_key: str, arm: str) -> pd.DataFrame:
    """Compatibility entry point for measuring a preserved campaign."""
    return measure_campaign(target_key, arm)


def summarize_by_length(values: pd.Series, lengths: pd.Series, edges: np.ndarray) -> pd.DataFrame:
    """Chain-weighted mean, standard error, and count in left-closed length bins.

    Include the upper endpoint of the final bin. Standard errors use the
    sample standard deviation across chains, not residues.
    """
    inclusive_edges = np.asarray(edges, dtype=float).copy()
    inclusive_edges[-1] = np.nextafter(inclusive_edges[-1], np.inf)
    binned = pd.cut(lengths, bins=inclusive_edges, right=False)
    grouped = values.groupby(binned, observed=False)
    counts = grouped.size()
    means = grouped.mean()
    errors = grouped.std(ddof=1) / np.sqrt(counts.where(counts > 0))
    return pd.DataFrame(
        {
            "centre": [interval.mid for interval in means.index],
            "mean": means.to_numpy(),
            "error": errors.to_numpy(),
            "n": counts.to_numpy(),
        }
    )


def load_all(target_keys: list[str], arms: list[str]) -> pd.DataFrame:
    """Measure all requested campaigns; missing inputs are an error."""
    frames = [load(key, name) for key in target_keys for name in arms if is_available(key, name)]
    if not frames:
        raise FileNotFoundError(f"No campaign composition tables under {paths.CAMPAIGNS}.")
    return pd.concat(frames, ignore_index=True)


def available(target_keys: list[str], arms: list[str]) -> list[tuple[str, str]]:
    """Which (target, arm) pairs have a composition table."""
    return [(key, name) for key in target_keys for name in arms if is_available(key, name)]
