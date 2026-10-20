"""Recompute the numerical evidence used in the publication without running designs.

Run ``python -m analysis.publication`` in the publication environment. This
measures saved campaign and reference structures. It never starts a GPU campaign.
Missing inputs can be fetched once an archive is published.
Fractions here describe predicted structures, and screening counts are not
experimental hit rates. Medians of the trajectory proxy and of final DSSP
assignments are reported separately.
"""

from __future__ import annotations
import json

import numpy as np
import pandas as pd

from analysis import composition, loss_trace, reference, targets

HELIX = "helix (fraction 0-1)"
SHEET = "sheet (fraction 0-1)"
IPTM = "iptm (0-1)"
PLDDT = "binder pLDDT (0-100)"
PROXY = "helix_mean (0-1)"
STEP = "step (index)"


def checked_composition(target: str, arm: str) -> pd.DataFrame:
    """Fail on incomplete or duplicate seed sets rather than silently pooling them."""
    frame = composition.load(target, arm)
    if len(frame) != targets.N_SEEDS or set(frame.seed) != set(range(targets.N_SEEDS)):
        raise ValueError(f"Incomplete or duplicate seeds: {target}/{arm}")
    columns = [HELIX, SHEET, "coil (fraction 0-1)", IPTM, PLDDT]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError(f"Nonfinite measurements: {target}/{arm}")
    if not np.allclose(frame[[HELIX, SHEET, "coil (fraction 0-1)"]].sum(axis=1), 1):
        raise ValueError(f"Composition does not sum to one: {target}/{arm}")
    return frame.assign(target=target, arm=arm)


def describe(frame: pd.DataFrame) -> dict:
    confidence_pass = frame[IPTM].gt(composition.MIN_IPTM) & frame[PLDDT].gt(
        composition.MIN_BINDER_PLDDT
    )
    return {
        "n": len(frame),
        "median": frame[[HELIX, SHEET, "coil (fraction 0-1)", IPTM, PLDDT]].median().to_dict(),
        "mean": frame[[HELIX, SHEET, "coil (fraction 0-1)"]].mean().to_dict(),
        "zero_sheet_count": int(frame[SHEET].eq(0).sum()),
        "confidence_pass_count": int(confidence_pass.sum()),
        "joint_screen_pass_count": int(composition.viable(frame).sum()),
    }


def report() -> dict:
    """Summaries for the registered main grid and both PD-L1 schedule scans."""
    arms = list(targets.CAMPAIGN_ARMS)
    grid = pd.concat(
        [checked_composition(t, a) for t in targets.TARGET_SPECS for a in arms],
        ignore_index=True,
    )
    natural = reference.composition()
    natural_cols = [f"binder_{state}_fraction" for state in ("helix", "sheet", "coil")]
    result = {
        "joint_screen": {
            "helix_constraint": None,
            "sheet_above": composition.MIN_SHEET_FRACTION,
            "iptm_above": composition.MIN_IPTM,
            "binder_plddt_above_0_to_100": composition.MIN_BINDER_PLDDT,
        },
        "main_grid": {a: describe(grid[grid.arm == a]) for a in arms},
        "per_target": {
            t: {a: describe(grid[(grid.target == t) & (grid.arm == a)]) for a in arms}
            for t in targets.TARGET_SPECS
        },
        "natural_reference": {
            "unique_chains": len(natural),
            "unique_entries": int(natural.pdb_id.nunique()),
            "median": natural[natural_cols].median().to_dict(),
            "mean": natural[natural_cols].mean().to_dict(),
            "zero_sheet_fraction": float(natural.binder_sheet_fraction.eq(0).mean()),
        },
    }

    for name, scan in (("onset", targets.ONSET_SCAN), ("release", targets.RELEASE_SCAN)):
        result[name] = {
            step: describe(checked_composition("pdl1_igv", arm)) for step, arm in scan.items()
        }

    traces = loss_trace.load_all(list(targets.TARGET_SPECS), arms)
    if len(traces) != len(grid) * targets.SGD_STEPS:
        raise ValueError("Incomplete main-grid traces")
    if traces.duplicated(["target", "arm", "seed", STEP]).any():
        raise ValueError("Duplicate main-grid trace steps")
    result["helix_proxy_by_step"] = {
        a: traces[traces.arm == a].groupby(STEP)[PROXY].median().loc[[0, 5, 10, 149]].to_dict()
        for a in arms
    }
    final = traces[traces[STEP] == targets.SGD_STEPS - 1]
    cd45 = final[(final.target == "cd45_d3d4") & (final.arm == "helix_mean_w10")]
    result["cd45_weight1_final_proxy"] = {
        "count_below_0.15": int(cd45[PROXY].lt(0.15).sum()),
        "median": float(cd45[PROXY].median()),
        "maximum": float(cd45[PROXY].max()),
    }

    # Same nominal seed does not imply an identical numerical trajectory.
    baseline = traces[(traces.target == "pdl1_igv") & (traces.arm == "vanilla")].set_index(
        ["seed", STEP]
    )
    result["pre_onset_maximum_absolute_proxy_difference"] = {}
    for step in (1, 2, 4, 7, 10):
        delayed = loss_trace.load_campaign("pdl1_igv", targets.ONSET_SCAN[step]).set_index(
            ["seed", STEP]
        )
        prefix = delayed[delayed.index.get_level_values(STEP) < step]
        delta = prefix[PROXY] - baseline.reindex(prefix.index)[PROXY]
        if delta.isna().any():
            raise ValueError("Missing baseline step in onset comparison")
        result["pre_onset_maximum_absolute_proxy_difference"][step] = float(delta.abs().max())
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
