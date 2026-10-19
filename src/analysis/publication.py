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


def short_initialization_report() -> dict:
    """Check the short-trajectory claims using only their saved campaigns."""
    short = pd.concat(
        [
            composition.measure_steps(targets.INITIALIZATION_TARGET, arm)
            for arm in targets.SHORT_TRAJECTORY_ARMS.values()
        ],
        ignore_index=True,
    )
    loss_trace.validate_matching(short)
    # Match the notebook: exclude failed geometry only from coordinate readouts.
    short.loc[~short["geometry pass (bool)"], [HELIX, SHEET, "coil (fraction 0-1)"]] = np.nan
    final_step = targets.INITIALIZATION_SPECS["short"]["updates"] - 1
    return {
        f"short_step{final_step}_means": short[short[STEP] == final_step]
        .groupby("arm")[[HELIX, SHEET, PROXY, "sequence entropy (bits)"]]
        .mean()
        .to_dict(orient="index"),
    }


def full_initialization_report(arms: list[str] | None = None) -> dict:
    """Check the long-run comparison independently of the short and softness runs.

    Notebook analysis can use these same campaign readers directly, keeping
    the choice of arms, aggregations and comparisons visible in its cells.
    """
    if arms is None:
        arms = [*targets.INITIALIZATION_BASELINES, *targets.FULL_INITIALIZATION_ARMS.values()]
    designs, traces = {}, []
    for arm in arms:
        designs[arm] = checked_composition(targets.INITIALIZATION_TARGET, arm)
        trace = loss_trace.load_campaign(targets.INITIALIZATION_TARGET, arm)
        loss_trace.validate_steps(trace, n_seeds=targets.N_SEEDS, steps=targets.SGD_STEPS)
        traces.append(trace)
    lengths = pd.concat(designs.values()).pivot(
        index="seed", columns="arm", values="binder length (residues)"
    )
    if not lengths.nunique(axis=1).eq(1).all():
        raise ValueError("Unmatched binder lengths across full initialization arms")

    outcomes = {}
    for arm, frame in designs.items():
        outcomes[arm] = describe(frame)
        outcomes[arm]["mean"].update(frame[[IPTM, PLDDT]].mean().to_dict())
    trace = pd.concat(traces, ignore_index=True)
    first_below = trace[trace["sequence entropy (bits)"] < 2].groupby(["arm", "seed"])[STEP].min()
    medians = first_below.groupby("arm").median()
    return {
        "full_optimization": outcomes,
        "median_first_step_below_2_bits": {
            arm: float(medians[arm]) if arm in medians.index else None for arm in arms
        },
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
    result["initialization"] = short_initialization_report()
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
