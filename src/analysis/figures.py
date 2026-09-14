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
from dataclasses import dataclass

import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

apc.mpl.setup()

HELIX_COLOR = apc.cinnabar.hex_code
SHEET_COLOR = apc.mustard.hex_code
LOOP_COLOR = apc.marine.hex_code
ERROR_COLOR = apc.charcoal.hex_code

LENGTH_MIN = 60
LENGTH_MAX = 200
LENGTH_BIN_COUNT = 5

FONT_SCALE = 1.2
"""How much to enlarge the house font sizes for these panels.

The house style sets absolute point sizes tuned to a full-width figure. These
panels render at about half the body column, so the type needs to be larger in
the source to read at the size it is displayed.
"""

FONT_SIZE_KEYS = (
    "font.size",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
)

BAR_WIDTH = 0.4

PANEL_SIZE_INCHES = (6.8, 5.4)
"""One panel of a three-panel row.

The house style sets absolute font sizes tuned to its own figure sizes, so a
panel drawn small enough to sit three-across on the page would have labels
that overflow it. Drawing at the house scale and letting Quarto scale the
image down keeps the type in proportion instead.
"""


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


@dataclass(frozen=True)
class State:
    """One secondary-structure state, and where to find it in each frame."""

    key: str
    label: str
    colour: str
    natural_column: str
    design_column: str


STATES: dict[str, State] = {
    "helix": State(
        key="helix",
        colour=HELIX_COLOR,
        label="Helix",
        natural_column="binder_helix_fraction",
        design_column="helix (fraction 0-1)",
    ),
    "sheet": State(
        key="sheet",
        colour=SHEET_COLOR,
        label="Beta sheet",
        natural_column="binder_sheet_fraction",
        design_column="sheet (fraction 0-1)",
    ),
    "loop": State(
        key="loop",
        colour=LOOP_COLOR,
        label="Loop",
        natural_column="binder_coil_fraction",
        design_column="coil (fraction 0-1)",
    ),
}


def _binned_means(
    values: pd.Series, lengths: pd.Series, edges: np.ndarray
) -> pd.DataFrame:
    """Mean, standard error and count of ``values`` per length bin.

    The mean rather than the median, because sheet fraction is zero-inflated:
    a median reads as exactly zero for both the designs and a fifth of the
    natural set, which hides the difference instead of showing it.
    """
    binned = pd.cut(lengths, bins=edges, right=False)
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


def composition_bars(
    natural: pd.DataFrame,
    designs: pd.DataFrame,
    state: str,
    natural_length: str = "binder_sample_length_residues",
    design_length: str = "binder length (residues)",
) -> plt.Figure:
    """Natural against designed composition for one state, binned by length.

    Five equally spaced length bins across the range the design loop samples
    binders from. Bars are bin means with standard-error whiskers. The
    reference whiskers are the small ones: each reference bar rests on about
    400 chains against 20 to 40 designs.

    One hue per state, so a panel is read by its own colour rather than by a
    shared two-colour key. Within each length bin the natural reference is the
    unfilled bar on the left and the designs are the filled bar on its right.

    The hues gesture at the structural-biology convention, red for helix and
    yellow for sheet, without following it into the red-green pairing that
    convention implies for loop: at the PyMOL defaults, yellow and green fall
    to a CVD colour difference of 4.8, well below the threshold for telling
    two categories apart. A cool neutral for loop is both accessible and
    defensible, since loop is the absence of regular structure.
    """
    spec = STATES[state]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _composition_bars(spec, natural, designs, natural_length, design_length)


def _composition_bars(
    spec: State,
    natural: pd.DataFrame,
    designs: pd.DataFrame,
    natural_length: str,
    design_length: str,
) -> plt.Figure:
    edges = np.linspace(LENGTH_MIN, LENGTH_MAX, LENGTH_BIN_COUNT + 1)
    reference = _binned_means(natural[spec.natural_column], natural[natural_length], edges)
    designed = _binned_means(designs[spec.design_column], designs[design_length], edges)

    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    index = np.arange(LENGTH_BIN_COUNT)
    for offset, frame, facecolor, linewidth, label in (
        (-BAR_WIDTH / 2, reference, "none", 1.6, "Natural binders (RCSB)"),
        (BAR_WIDTH / 2, designed, spec.colour, 0.8, "ESMFold2 designs"),
    ):
        axis.bar(
            index + offset,
            frame["mean"],
            width=BAR_WIDTH,
            facecolor=facecolor,
            edgecolor=spec.colour,
            linewidth=linewidth,
            label=label,
        )
        axis.errorbar(
            index + offset,
            frame["mean"],
            yerr=frame["error"],
            fmt="none",
            ecolor=ERROR_COLOR,
            elinewidth=1.0,
        )
    axis.set_xticks(index)
    axis.set_xticklabels(
        [f"{int(lo)}\u2013{int(hi)}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    )
    axis.set_xlabel("binder length (residues)")
    axis.set_ylabel(f"mean {spec.label.lower()} (fraction of binder residues, 0-1)")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.04, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.0,
    )
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure
