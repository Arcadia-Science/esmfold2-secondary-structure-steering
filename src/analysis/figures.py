"""Publication figures.

Colors come from the Arcadia primary categorical palette, assigned in its fixed
order rather than cycled: aegean for the natural reference, amber for the
designs. Amber sits below 3:1 contrast against the chart surface, so every
figure using it carries a legend and direct labels rather than relying on color
alone.

Composition is always plotted against chain length. Short natural proteins are
helix-biased too, so a single pooled comparison would confound composition with
length and overstate the effect.
"""

from __future__ import annotations

import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NATURAL_COLOR = apc.palettes.primary.colors[0].hex_code
DESIGN_COLOR = apc.palettes.primary.colors[1].hex_code

LENGTH_MIN = 60
LENGTH_MAX = 200
BIN_WIDTH = 10


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) minus P(a < b), computed exactly over all pairs.

    A rank-only effect size, which is what these distributions need: helix and
    sheet fractions are bounded at zero and one, and the sheet distribution is
    heavily zero-inflated, so a mean and standard deviation describe a shape the
    data does not have.
    """
    ordered = np.sort(a)
    n = len(a) * len(b)
    lower = np.searchsorted(ordered, b, side="left").sum()
    upper = np.searchsorted(ordered, b, side="right").sum()
    return (1 - upper / n) - (lower / n)


def _binned_quantiles(
    frame: pd.DataFrame, length_column: str, value_column: str
) -> pd.DataFrame:
    edges = list(range(LENGTH_MIN, LENGTH_MAX + BIN_WIDTH, BIN_WIDTH))
    binned = pd.cut(frame[length_column], bins=edges, right=False)
    grouped = frame.groupby(binned, observed=True)[value_column]
    summary = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    summary.columns = ["q25", "q50", "q75"]
    summary["centre"] = [interval.mid for interval in summary.index]
    summary["n"] = grouped.size().to_numpy()
    return summary.reset_index(drop=True)


def composition_vs_length(
    natural: pd.DataFrame,
    designs: pd.DataFrame,
    natural_length: str = "binder_sample_length_residues",
    design_length: str = "binder length (residues)",
) -> plt.Figure:
    """Question 1 figure: helix and sheet against binder length, two panels.

    Natural binders are summarized as a median line with an interquartile band
    per 10-residue bin. Individual designs are drawn as points, so the reader
    can see the spread rather than only a summary of it.
    """
    panels = [
        ("binder_helix_fraction", "helix (fraction 0-1)", "Helix"),
        ("binder_sheet_fraction", "sheet (fraction 0-1)", "Sheet"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)

    for axis, (natural_column, design_column, label) in zip(axes, panels, strict=True):
        summary = _binned_quantiles(natural, natural_length, natural_column)
        axis.fill_between(
            summary["centre"],
            summary["q25"],
            summary["q75"],
            color=NATURAL_COLOR,
            alpha=0.22,
            linewidth=0,
        )
        axis.plot(
            summary["centre"],
            summary["q50"],
            color=NATURAL_COLOR,
            linewidth=2,
            label="Natural binders, median and IQR",
        )
        axis.scatter(
            designs[design_length],
            designs[design_column],
            s=26,
            color=DESIGN_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label="ESMFold2 vanilla designs",
        )
        axis.set_title(f"{label} content", fontsize=11)
        axis.set_xlabel("binder chain length (residues)")
        axis.set_ylim(-0.03, 1.0)
        axis.set_xlim(LENGTH_MIN - 5, LENGTH_MAX + 5)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

        delta = cliffs_delta(
            designs[design_column].to_numpy(), natural[natural_column].to_numpy()
        )
        axis.annotate(
            f"Cliff's delta {delta:+.2f}",
            xy=(0.97, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color="#4A4A4A",
        )

    axes[0].set_ylabel("fraction of binder residues (0-1)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "ESMFold2 designs are helix-rich and sheet-poor against natural binders "
        "of the same length",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure
