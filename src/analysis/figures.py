"""Publication figures.

Two colour schemes, each matched to what it encodes.

The Question 1 panels use one hue per secondary-structure state, so a panel is
read by its own colour rather than by a shared key. The Question 2 panels
encode steering weight, which is ordered, so they use a single-hue sequential
ramp from the Arcadia blue shades rather than categorical hues. Assigning five
categorical hues to the five targets was the obvious alternative and it fails:
the first five of the primary ordered palette put aegean against aster at a
colour difference of 4.1 under simulated dichromacy, and 6.8 even in normal
vision. Targets are therefore a categorical axis, not a colour.

The ramp is the darkest four steps of the house blue shades, vital through
dusk, taken in the palette's own order. Its worst adjacent separation is 11.3
under simulated deuteranopia and its worst pair anywhere is 10.4 under
simulated protanopia, both above the 8 needed to stay distinguishable. Its
lightest step sits below 3:1 contrast against the chart surface, so every
figure using it carries a legend or names its columns on an axis, and the
medians it plots are also tabulated in the surrounding text.

The unsteered arm takes bark, a warm grey off the ramp, because the absence of
a steering weight is not a step on the weight scale.

Composition in Question 1 is always plotted against chain length. Short natural
proteins are helix-biased too, so a single pooled comparison would confound
composition with length and overstate the effect.
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

WEIGHT_RAMP = (
    apc.vital.hex_code,
    apc.aegean.hex_code,
    apc.lapis.hex_code,
    apc.dusk.hex_code,
)
"""Steering weight as a light-to-dark ramp, the darkest step the strongest.

These are the darkest four steps of the house ``blue_shades`` palette, taken in
its own order rather than picked by eye. A panel's weights take the darkest
steps available, so the column where the fold has converted is the darkest one
whatever the panel's length, and the x axis carries the weight itself. Colour is
redundant with position rather than the only way to read a column.

Adjacent steps are separated by at least 11.3 under simulated deuteranopia,
which is the worst of the three vision models. Extending past four means
reaching for ``sky``, whose contrast against the chart surface is too low to
carry a dot, so a fifth weight needs a different encoding rather than a fifth
step.
"""

BASELINE_COLOR = apc.bark.hex_code
"""The unsteered column, a warm grey deliberately off the blue ramp.

Unsteered is the absence of a weight, not a step on the scale, so it sits off
the ramp. Its worst separation from a ramp step is 10.4 under simulated
protanopia, above the 8 needed to stay distinguishable, but the palette's worst
normal-vision pair is 11.4, below the 15 a categorical palette would need. That
is tolerable only because these columns are separated by position and named on
the x axis, so colour never carries identity on its own here.
"""

ARM_COLORS = {
    "vanilla": BASELINE_COLOR,
    "helix_mean_w05": WEIGHT_RAMP[0],
    "helix_mean_w10": WEIGHT_RAMP[1],
    "helix_mean_w20": WEIGHT_RAMP[2],
    "helix_mean_w50": WEIGHT_RAMP[3],
}
"""The grid's arms on the shared ramp, so a weight is the same colour in every
panel of the publication."""

ARM_LABELS = {
    "vanilla": "Unsteered",
    "helix_mean_w05": "0.5",
    "helix_mean_w10": "1.0",
    "helix_mean_w20": "2.0",
    "helix_mean_w50": "5.0",
}
"""Bare weights, with the quantity named once in the legend title instead.

Five entries reading "Weight 0.5" need their own column and take 40% of a
panel's height. Bare numbers fit on one row above the axes.
"""

ARM_WEIGHT_LABELS = {
    "vanilla": "0",
    "helix_mean_w05": "0.5",
    "helix_mean_w10": "1.0",
    "helix_mean_w20": "2.0",
    "helix_mean_w50": "5.0",
}
"""Weights as bare numbers, with the unsteered arm written as the zero it is.

Panels whose y axis label is long enough to narrow the axes cannot fit the word
"Unsteered" beside a neighbouring tick, and the x axis already names the
quantity, so the unsteered arm is a weight of zero like any other.
"""

LEGEND_TITLE = "Steering weight (dimensionless)"

LABEL_MIN_GAP_FRACTION = 0.07
"""Least vertical room a direct label needs, as a fraction of the axis span.

Panels in this module label their lines rather than carrying a legend. A
legend of five arms does not fit: one entry per row takes 40% of the panel
height, and any layout wide enough to fit on fewer rows overflows the panel
width and makes tight_layout shrink the axes instead. Labelling needs its own
collision rule, which is what this sets.
"""


def _spread_labels(values: list[float], minimum_gap: float) -> list[float]:
    """Nudge label positions apart while keeping their order.

    Two lines can finish close enough that their labels overlap and one hides
    the other, which is worse than a cramped panel because the hidden series
    looks absent. Walking the labels from the bottom and pushing each one up to
    clear its neighbour keeps every label visible and keeps them in the same
    order as the values they name, so a label never sits beside the wrong line.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = list(values)
    for earlier, later in zip(order, order[1:], strict=False):
        shortfall = minimum_gap - (adjusted[later] - adjusted[earlier])
        if shortfall > 0:
            adjusted[later] += shortfall
    return adjusted


DOT_SIZE_POINTS = 46
DOT_OFFSET = 0.26
TRACE_LINE_WIDTH = 2.0

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


def _binned_means(values: pd.Series, lengths: pd.Series, edges: np.ndarray) -> pd.DataFrame:
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


TARGET_LABELS = {
    "egfr_d3": "EGFR",
    "pdgfrb_d2d3": "PDGFRβ",
    "pdl1_igv": "PD-L1",
    "ctla4_igv": "CTLA-4",
    "cd45_d3d4": "CD45",
}
"""Protein names only. Which domain of each is the construct is a fact for the
figure caption, not for five x-axis labels that have to stay narrow."""


@dataclass(frozen=True)
class Metric:
    """One per-design quantity, and how to label an axis carrying it."""

    key: str
    column: str
    axis_label: str


METRICS: dict[str, Metric] = {
    "helix": Metric(
        key="helix",
        column="helix (fraction 0-1)",
        axis_label="median helix fraction (0-1)",
    ),
    "sheet": Metric(
        key="sheet",
        column="sheet (fraction 0-1)",
        axis_label="median sheet fraction (0-1)",
    ),
    "iptm": Metric(
        key="iptm",
        column="iptm (0-1)",
        axis_label="median interface pTM (0-1)",
    ),
    "plddt": Metric(
        key="plddt",
        column="binder pLDDT (0-100)",
        axis_label="median binder pLDDT (0-100)",
    ),
}


def _median_and_quartiles(values: pd.Series) -> tuple[float, float, float]:
    """Median with the distances to the first and third quartile."""
    median = float(values.median())
    lower = median - float(values.quantile(0.25))
    upper = float(values.quantile(0.75)) - median
    return median, lower, upper


def steering_dots(
    designs: pd.DataFrame,
    metric: str,
    target_order: list[str],
    arms: list[str],
) -> plt.Figure:
    """One quantity against target, with a dot per steering weight.

    Dots rather than bars because two of the four quantities this draws,
    interface pTM and binder pLDDT, do not live on a scale whose zero means
    anything. A bar would assert a baseline the data does not have. Whiskers
    span the interquartile range.

    The median rather than the mean, because helix fraction under steering is
    bimodal and sheet fraction is zero-inflated, so neither distribution is
    described by a centre and a width.
    """
    spec = METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _steering_dots(spec, designs, target_order, arms)


def _steering_dots(
    spec: Metric,
    designs: pd.DataFrame,
    target_order: list[str],
    arms: list[str],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    index = np.arange(len(target_order))
    offsets = np.linspace(-DOT_OFFSET, DOT_OFFSET, len(arms))

    for arm, offset in zip(arms, offsets, strict=True):
        centres, lowers, uppers = [], [], []
        for target in target_order:
            values = designs.loc[
                (designs["target"] == target) & (designs["arm"] == arm), spec.column
            ]
            centre, lower, upper = _median_and_quartiles(values)
            centres.append(centre)
            lowers.append(lower)
            uppers.append(upper)
        axis.errorbar(
            index + offset,
            centres,
            yerr=np.vstack([lowers, uppers]),
            fmt="none",
            ecolor=ARM_COLORS[arm],
            elinewidth=1.2,
        )
        axis.scatter(
            index + offset,
            centres,
            s=DOT_SIZE_POINTS,
            color=ARM_COLORS[arm],
            edgecolor=apc.parchment.hex_code,
            linewidth=0.8,
            zorder=3,
            label=ARM_LABELS[arm],
        )

    axis.set_xticks(index)
    axis.set_xticklabels([TARGET_LABELS[target] for target in target_order])
    axis.set_xlim(-0.6, len(target_order) - 0.4)
    axis.set_ylabel(spec.axis_label)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        title=LEGEND_TITLE,
        alignment="left",
        frameon=False,
        handlelength=0.8,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure


STRIP_DOT_SIZE_POINTS = 26
STRIP_JITTER_WIDTH = 0.16
STRIP_MEDIAN_HALF_WIDTH = 0.28
STRIP_JITTER_SEED = 0
"""Fixed so the jitter is identical every render; the offsets carry no meaning."""


def scan_strip(
    designs: pd.DataFrame,
    target: str,
    baseline_arm: str,
    weight_arms: list[str],
    weight_labels: list[str],
    metric: str,
) -> plt.Figure:
    """One dot per design, in a column per steering weight.

    A dot per design rather than a median and an interval, because what
    separates the two steering terms is the shape of the distribution and not
    its centre. The capped barrier is bimodal: at the weight where it has moved
    half of its designs, those designs sit at one end or the other and almost
    none in between. The averaged term fills the middle at the comparable
    weight. A median hides exactly that, and an interquartile bar drawn on a
    bimodal sample describes a shape the sample does not have.

    Panels are drawn on a fixed 0 to 1 axis and each repeats the unsteered arm,
    so a column can be compared against one in the neighbouring panel and
    neither panel has to be read through the other.
    """
    if len(weight_arms) > len(WEIGHT_RAMP):
        raise ValueError(
            f"{len(weight_arms)} weights exceeds the {len(WEIGHT_RAMP)}-step ramp; "
            "adding a step means revalidating the whole palette for contrast "
            "and dichromat separation"
        )
    spec = METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _scan_strip(spec, designs, target, baseline_arm, weight_arms, weight_labels)


def _scan_strip(
    spec: Metric,
    designs: pd.DataFrame,
    target: str,
    baseline_arm: str,
    weight_arms: list[str],
    weight_labels: list[str],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    generator = np.random.default_rng(STRIP_JITTER_SEED)

    arms = [baseline_arm, *weight_arms]
    colours = [BASELINE_COLOR, *WEIGHT_RAMP[-len(weight_arms) :]]
    labels = ["Unsteered", *weight_labels]

    for position, (arm, colour) in enumerate(zip(arms, colours, strict=True)):
        values = designs.loc[
            (designs["target"] == target) & (designs["arm"] == arm), spec.column
        ].to_numpy()
        if not len(values):
            raise ValueError(f"no designs for {target} / {arm}")
        jitter = generator.uniform(-STRIP_JITTER_WIDTH, STRIP_JITTER_WIDTH, len(values))
        axis.scatter(
            position + jitter,
            values,
            s=STRIP_DOT_SIZE_POINTS,
            color=colour,
            edgecolor=apc.parchment.hex_code,
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )
        median = float(np.median(values))
        axis.plot(
            [position - STRIP_MEDIAN_HALF_WIDTH, position + STRIP_MEDIAN_HALF_WIDTH],
            [median, median],
            color=ERROR_COLOR,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=4,
        )

    axis.set_xticks(np.arange(len(arms)))
    axis.set_xticklabels(labels)
    axis.set_xlim(-0.6, len(arms) - 0.4)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("steering weight (dimensionless)")
    axis.set_ylabel(spec.column)
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure


LINE_LABEL_PAD = 0.12


def _widest_gap_position(rows: list[list[float]]) -> int:
    """The x position whose series values are furthest apart.

    Direct labels are only legible where the lines they name are separated, and
    which position that is depends on the quantity. Median helix fans out at
    the lowest steering weight and converges on zero by the highest; interface
    pTM does the reverse, with two targets within 0.003 of each other at low
    weight and the spread only opening at high weight. Choosing the position by
    hand suited one panel and silently hid a line on the other, so it is
    measured: for each position, take the smallest gap between neighbouring
    values, and keep the position where that smallest gap is largest.
    """
    best_position, best_gap = 0, -1.0
    for position in range(len(rows[0])):
        ordered = sorted(row[position] for row in rows)
        gap = min(
            (later - earlier for earlier, later in zip(ordered, ordered[1:], strict=False)),
            default=0.0,
        )
        if gap > best_gap:
            best_position, best_gap = position, gap
    return best_position


def scan_lines(
    designs: pd.DataFrame,
    target_order: list[str],
    arms: list[str],
    metric: str,
) -> plt.Figure:
    """Median of one quantity against steering weight, a labelled line per target.

    Lines carry direct labels and the panel has no legend. Five series need
    either five legend rows, which take 40% of the panel height, or one row
    that does not fit the width; and giving five targets five hues fails on
    colour difference, which is why targets are a categorical axis everywhere
    else in this module. A single hue with the names written on the lines
    avoids all three.

    Weight is placed at equal spacing rather than in proportion, so the x axis
    is a sequence of conditions and not a numeric scale. Reading a slope off it
    as a rate would be wrong; what it shows is order and direction.
    """
    spec = METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _scan_lines(spec, designs, target_order, arms)


def _scan_lines(
    spec: Metric,
    designs: pd.DataFrame,
    target_order: list[str],
    arms: list[str],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    positions = np.arange(len(arms))

    rows = [
        [
            float(
                designs.loc[
                    (designs["target"] == target) & (designs["arm"] == arm), spec.column
                ].median()
            )
            for arm in arms
        ]
        for target in target_order
    ]
    label_position = _widest_gap_position(rows)

    for target, medians in zip(target_order, rows, strict=True):
        axis.plot(
            positions,
            medians,
            color=apc.aegean.hex_code,
            linewidth=TRACE_LINE_WIDTH,
            marker="o",
            markersize=5,
            markeredgecolor=apc.parchment.hex_code,
            markeredgewidth=0.8,
            zorder=3,
        )
        axis.text(
            label_position + LINE_LABEL_PAD,
            medians[label_position],
            TARGET_LABELS[target],
            color=ERROR_COLOR,
            va="center",
            ha="left",
            fontsize=plt.rcParams["xtick.labelsize"],
            bbox={
                "facecolor": apc.parchment.hex_code,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.15",
            },
            zorder=4,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([ARM_LABELS[arm] for arm in arms])
    trailing = 0.95 if label_position == len(arms) - 1 else 0.3
    axis.set_xlim(-0.3, len(arms) - 1 + trailing)
    axis.set_xlabel(LEGEND_TITLE)
    axis.set_ylabel(spec.axis_label)
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure


SCATTER_DOT_SIZE_POINTS = 14
SCATTER_ALPHA = 0.6


def proxy_against_critic(paired: pd.DataFrame, arms: list[str]) -> plt.Figure:
    """Each design's in-loop interface score against the score it is judged by.

    The optimizer is steered by a distogram proxy computed every step; the
    design is reported with an interface pTM from the critics, which refold the
    discretised sequence at higher sampling. Those are different quantities, and
    the proxy is the one the gradient sees, so whether it ranks designs the way
    the critics do is worth showing rather than assuming.

    Both axes span 0 to 1 at equal aspect so the identity line lies at 45
    degrees and the proxy's offset from the critics reads as distance from it
    rather than as something to infer from two scales. Colour is the steering
    weight, which is what says whether the proxy degrades as the term pushes
    harder.

    The legend sits inside the axes at lower right, under a title broken across
    two lines. The critics score above the proxy on 76% of designs, so that
    corner holds 9 of 750 points and is the only sparse one; the title has to
    wrap because on one line it makes the key wide enough to reach back into
    the cloud, and moving the key outside the axes instead leaves no room for
    the axis labels once the panel is square.
    """
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _proxy_against_critic(paired, arms)


def _proxy_against_critic(paired: pd.DataFrame, arms: list[str]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=ERROR_COLOR,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    for arm in arms:
        subset = paired[paired["arm"] == arm]
        axis.scatter(
            subset["distogram iptm proxy (0-1)"],
            subset["iptm (0-1)"],
            s=SCATTER_DOT_SIZE_POINTS,
            color=ARM_COLORS[arm],
            edgecolor=apc.parchment.hex_code,
            linewidth=0.3,
            alpha=SCATTER_ALPHA,
            label=ARM_LABELS[arm],
            zorder=3,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xlabel("final-step distogram interface pTM proxy (0-1)")
    axis.set_ylabel("critic interface pTM (0-1)")
    axis.legend(
        loc="lower right",
        ncol=1,
        title="Steering weight\n(dimensionless)",
        alignment="left",
        frameon=False,
        handlelength=0.8,
        handletextpad=0.3,
        labelspacing=0.3,
    )
    apc.mpl.style_plot(axis)
    figure.tight_layout()
    return figure


STACK_EDGE_WIDTH = 0.8


def composition_stack(designs: pd.DataFrame, target: str, seed: int, arms: list[str]) -> plt.Figure:
    """One design's secondary-structure composition across the weight scan.

    The dot and line panels reduce each arm to a median, and the per-seed trace
    panels show that an arm is not one behaviour. This shows neither: it follows
    a single seed, whose inversion is deterministic given the weight, so the
    five bars are the same search run five times with one number changed. The
    three states sum to the whole chain, so a stacked bar carries the whole
    composition without a second axis, and what the panel is for is the
    exchange between states rather than the level of any one of them.

    States take the same three colours as the composition panels in Question 1,
    so a reader who has learned them once does not learn them again.
    """
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _composition_stack(designs, target, seed, arms)


def _composition_stack(
    designs: pd.DataFrame, target: str, seed: int, arms: list[str]
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    positions = np.arange(len(arms))

    bottoms = np.zeros(len(arms))
    for state in STATES.values():
        heights = []
        for arm in arms:
            row = designs[
                (designs["target"] == target) & (designs["arm"] == arm) & (designs["seed"] == seed)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"expected one design for {target} / {arm} / seed {seed}, got {len(row)}"
                )
            heights.append(float(row.iloc[0][state.design_column]))
        axis.bar(
            positions,
            heights,
            width=BAR_WIDTH,
            bottom=bottoms,
            color=state.colour,
            edgecolor=apc.parchment.hex_code,
            linewidth=STACK_EDGE_WIDTH,
            label=state.label,
            zorder=3,
        )
        bottoms += np.asarray(heights)

    axis.set_xticks(positions)
    axis.set_xticklabels([ARM_WEIGHT_LABELS[arm] for arm in arms])
    axis.set_xlim(-0.6, len(arms) - 0.4)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(LEGEND_TITLE)
    axis.set_ylabel("Chain in each state (fraction 0-1)")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure


@dataclass(frozen=True)
class TraceMetric:
    """One per-step quantity from the loss log."""

    key: str
    column: str
    axis_label: str
    invert: bool = False
    """Plot ``1 - column`` rather than the column itself.

    The sheet terms are shortfalls: each computes one minus a contact score, so
    a low value means the binder has more sheet. The helix term is the opposite,
    a mean contact probability where high means more helix. Plotting both raw
    puts "more structure" at opposite ends of the two panels, which invites a
    reader comparing them to draw the conclusion backwards. Inverting the
    shortfall restores a score, so up is more sheet in the panel and more helix
    in its neighbour.
    """


TRACE_METRICS: dict[str, TraceMetric] = {
    "helix_mean": TraceMetric(
        key="helix_mean",
        column="helix_mean (0-1)",
        axis_label="mean helix rung fraction (0-1)",
    ),
    "sheet_mean": TraceMetric(
        key="sheet_mean",
        column="sheet_mean (0-1)",
        axis_label="mean sheet contact score (0-1)",
        invert=True,
    ),
    "iptm_proxy": TraceMetric(
        key="iptm_proxy",
        column="distogram iptm proxy (0-1)",
        axis_label="distogram interface pTM proxy (0-1)",
    ),
}


def _oriented(values: pd.Series, spec: TraceMetric) -> pd.Series:
    """The column as the panel plots it, flipping shortfalls back into scores."""
    return 1.0 - values if spec.invert else values


SEED_LINE_WIDTH = 0.8
SEED_LINE_ALPHA = 0.45


def seed_traces(trace: pd.DataFrame, target: str, arm: str, metric: str) -> plt.Figure:
    """Every seed's own trajectory for one target and weight, over the median.

    The per-step panels elsewhere plot a median across all designs in an arm,
    which is the right summary for comparing weights but hides that the arm is
    not one behaviour. At a weight in the middle of the transition the seeds do
    not track the median: some are held near the floor for the whole run and
    others climb back to where an unsteered design sits, and the median lands
    between two groups rather than describing either. Drawing the seeds shows
    what the median is averaging over.

    The seeds are drawn thin and translucent and the median over them heavy, so
    the panel reads as a spread with a summary rather than as thirty separate
    series competing to be followed.
    """
    spec = TRACE_METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _seed_traces(spec, trace, target, arm)


def _seed_traces(spec: TraceMetric, trace: pd.DataFrame, target: str, arm: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    subset = trace[(trace["target"] == target) & (trace["arm"] == arm)]
    if subset.empty:
        raise ValueError(f"no loss log for {target} / {arm}")

    for _seed, group in subset.groupby("seed"):
        ordered = group.sort_values("step (index)")
        axis.plot(
            ordered["step (index)"].to_numpy(),
            _oriented(ordered[spec.column], spec).to_numpy(),
            color=apc.vital.hex_code,
            linewidth=SEED_LINE_WIDTH,
            alpha=SEED_LINE_ALPHA,
            zorder=2,
        )

    grouped = _oriented(subset[spec.column], spec).groupby(subset["step (index)"]).median()
    axis.plot(
        grouped.index.to_numpy(),
        grouped.to_numpy(),
        color=ERROR_COLOR,
        linewidth=TRACE_LINE_WIDTH,
        label=f"median of {subset['seed'].nunique()} seeds",
        zorder=3,
    )

    axis.set_xlabel("optimization step (index)")
    axis.set_ylabel(spec.axis_label)
    axis.set_xlim(0, subset["step (index)"].max())
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
    )
    apc.mpl.style_plot(axis)
    figure.tight_layout()
    return figure


def trace_lines(trace: pd.DataFrame, metric: str, arms: list[str]) -> plt.Figure:
    """One per-step quantity against optimization step, a line per arm.

    The line is the per-step median across every design in that arm. No spread
    band: the interquartile ranges of the arms overlap across most of the run
    and drawing them hid the lines they belonged to. What this panel is for is
    timing, and the per-design spread is already on the dot panels.

    The quantity plotted for the steering term is the term itself, the mean
    i to i+4 contact probability, so the panel shows the optimizer driving down
    the thing it is scored on rather than a correlate of it.
    """
    spec = TRACE_METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _trace_lines(spec, trace, arms)


def _trace_lines(spec: TraceMetric, trace: pd.DataFrame, arms: list[str]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)

    endpoints = []
    for arm in arms:
        subset = trace[trace["arm"] == arm]
        oriented = _oriented(subset[spec.column], spec)
        steps = np.asarray(sorted(subset["step (index)"].unique()))
        medians = oriented.groupby(subset["step (index)"]).median().to_numpy()
        axis.plot(
            steps,
            medians,
            color=ARM_COLORS[arm],
            linewidth=TRACE_LINE_WIDTH,
            zorder=3,
        )
        endpoints.append(float(medians[-1]))

    last_step = float(trace["step (index)"].max())
    span = max(endpoints) - min(endpoints)
    placed = _spread_labels(endpoints, span * LABEL_MIN_GAP_FRACTION)
    for arm, height in zip(arms, placed, strict=True):
        axis.text(
            last_step * 1.02,
            height,
            ARM_LABELS[arm],
            color=ARM_COLORS[arm],
            va="center",
            ha="left",
            fontsize=plt.rcParams["xtick.labelsize"],
        )

    axis.set_xlabel("optimization step (index)")
    axis.set_ylabel(spec.axis_label)
    axis.set_xlim(0, trace["step (index)"].max() * 1.18)
    apc.mpl.style_plot(axis)
    figure.tight_layout()
    return figure
