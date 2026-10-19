"""Publication figures.

Three colour schemes, each matched to what it encodes.

The Question 1 panels use one hue per secondary-structure state, so a panel is
read by its own colour rather than by a shared key.

Steering weight is ordered, so most panels that encode it use a single-hue
sequential ramp rather than categorical hues, and that ramp is grey: the
quantity plotted against it is already carried by a hue of its own, and a grey
weight scale leaves that hue to it. The unsteered arm is the ramp's lightest
step rather than a colour beside it, because a weight of zero is the low end
of the scale and not a separate condition. Five steps run light to dark from
chateau to pitch, with a worst pair anywhere of 12.7 over the three vision
models.

Secondary-structure boxplots use their state's colour at uniform opacity.

Targets use the five cool tones in ``TARGET_COLORS``, selected by comparing
palette variants of the same target-specific boxplots.

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
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

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
    apc.bark.hex_code,
    apc.mud.hex_code,
    apc.crow.hex_code,
    apc.pitch.hex_code,
)
"""The steered weights, continuing the ramp down from the unsteered step.

Grey because the quantity these panels follow already has a hue: the helix term
is red wherever it is measured here, and a coloured weight scale would compete
with it. Colour is redundant with position rather than the only way to read a
line, since the x axis carries the weight itself.

With the unsteered step ahead of them the five run 0.76, 0.63, 0.48, 0.28 and
0.14 in OKLab lightness, so darkness increases with weight the whole way and no
step is lighter than a weaker one. Adjacent steps are separated by at least
12.7 over normal vision, simulated protanopia and simulated deuteranopia.
"""

BASELINE_COLOR = apc.chateau.hex_code
"""The unsteered arm, the lightest step of the same ramp.

A weight of zero is the low end of the weight scale, so it takes the low end of
the ramp rather than a colour off it. The ramp stops here rather than reaching
for a paler grey, because a pale grey line has no contrast against the chart
surface to spend.
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

At this font "Unsteered" is wider than the one unit between ticks, so a narrow
panel cannot spell it without the label running into its neighbour. The x axis
names the quantity, so the unsteered arm becomes a weight of zero like any
other.
"""

LEGEND_TITLE = "Steering weight"

LABEL_MIN_GAP_FRACTION = 0.07
"""Least vertical room a direct label needs, as a fraction of the axis span.

Panels in this module label their lines rather than carrying a legend. A
legend of five arms does not fit: one entry per row takes 40% of the panel
height, and any layout wide enough to fit on fewer rows overflows the panel
width and makes tight_layout shrink the axes instead. Labelling needs its own
collision rule, which is what this sets.
"""


@dataclass(frozen=True)
class _LabelBand:
    """The room a column of direct labels has, and the least gap between two.

    ``floor`` and ``ceiling`` are the axis limits pulled in by half a gap: a
    gap is the room one label needs, so half of it is the room from a label's
    centre to its own edge, and a label placed at the floor then clears the
    spine instead of sitting on it.
    """

    gap: float
    floor: float
    ceiling: float


@dataclass(frozen=True)
class _LabelCluster:
    """Labels close enough to each other that they are placed as one block."""

    members: list[int]
    total: float
    band: _LabelBand

    @property
    def height(self) -> float:
        return (len(self.members) - 1) * self.band.gap

    @property
    def start(self) -> float:
        centre = self.total / len(self.members)
        return max(min(centre - self.height / 2, self.band.ceiling - self.height), self.band.floor)

    @property
    def top(self) -> float:
        return self.start + self.height

    def merged(self, other: _LabelCluster) -> _LabelCluster:
        return _LabelCluster(self.members + other.members, self.total + other.total, self.band)


def _spread_labels(
    values: list[float], minimum_gap: float, bounds: tuple[float, float]
) -> list[float]:
    """Nudge label positions apart while keeping their order.

    Two lines can finish close enough that their labels overlap and one hides
    the other, which is worse than a cramped panel because the hidden series
    looks absent.

    Labels that collide are placed as a block, evenly spaced and centred on the
    mean of the positions they came from, so a colliding pair splits the
    distance between them rather than one label carrying all of it. Pushing
    every label up to clear the one below is simpler, but it leaves the upper
    label of a pair a full gap away from its own line while the lower one still
    sits exactly on its own.

    A block is held inside ``bounds``, the limits of the axis it is drawn on,
    because centring one that reaches an edge would put a label on the spine or
    past it. Blocks merge until none of them collide, which is a loop rather
    than a single pass because holding a block inside the bounds can move it
    into its neighbour. Every label stays visible and in the same order as the
    values they name, so a label never sits beside the wrong line.
    """
    band = _LabelBand(minimum_gap, bounds[0] + minimum_gap / 2, bounds[1] - minimum_gap / 2)
    order = sorted(range(len(values)), key=lambda index: values[index])
    clusters = [_LabelCluster([index], values[index], band) for index in order]
    collided = True
    while collided:
        collided = False
        for position in range(len(clusters) - 1):
            lower, upper = clusters[position], clusters[position + 1]
            if upper.start - lower.top < minimum_gap:
                clusters[position : position + 2] = [lower.merged(upper)]
                collided = True
                break
    adjusted = list(values)
    for cluster in clusters:
        for rank, index in enumerate(cluster.members):
            adjusted[index] = cluster.start + rank * minimum_gap
    return adjusted


TRACE_LINE_WIDTH = 2.0
SCAN_LINE_WIDTH = 3.4
SCAN_MARKER_SIZE = 9
"""Heavier than the per-step traces. Those carry 150 points each and thin lines
keep them apart; these carry five, so the line is the mark."""

SCAN_TRACE_SIZE_INCHES = (5.25, 5.8)
SCAN_COMPOSITION_SIZE_INCHES = (5.3, 5.8)
"""Sized to fill a half-width layout cell rather than to a chosen aspect.

Two of these sit in a row of a body column that stops at 850 CSS pixels, so a
cell is about 425 and a figure has to reach it or leave the difference as a gap
between the panels. At 150 dpi Quarto writes half the rendered pixels into the
tag, and the tight bounding box takes in whatever sits outside the axes, so the
two panel kinds reach the same rendered width from different figure widths: the
trace panel hangs its line labels past the axes and the composition panel does
not.
"""

SCAN_LEGEND_ANCHOR = (0.5, -0.42)

PANEL_SIZE_INCHES = (6.8, 5.4)
"""One panel of a three-panel row.

The house style sets absolute font sizes tuned to its own figure sizes, so a
panel drawn small enough to sit three-across on the page would have labels
that overflow it. Drawing at the house scale and letting Quarto scale the
image down keeps the type in proportion instead.
"""


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
    """Compatibility wrapper for the chain-level length summary."""
    from analysis.composition import summarize_by_length

    return summarize_by_length(values, lengths, edges)


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
    Each natural reference chain contributes once to its length bin.

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

    return _composition_comparison(spec, reference, designed, edges)


def composition_comparison(
    reference: pd.DataFrame, designed: pd.DataFrame, state: str, edges: np.ndarray
) -> plt.Figure:
    """Draw precomputed chain means and standard errors in the supplied length bins."""
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _composition_comparison(STATES[state], reference, designed, edges)


def _composition_comparison(spec, reference, designed, edges) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    index = np.arange(len(edges) - 1)
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

TARGET_COLORS = {
    "egfr_d3": "#305F8C",
    "pdgfrb_d2d3": "#2B8C91",
    "pdl1_igv": "#6C70B5",
    "ctla4_igv": "#6B9A80",
    "cd45_d3d4": "#A16F9B",
}
"""One hue per target, attached to the target rather than to its rank, so a
panel that reorders the targets does not repaint them.

The approved cool-tones palette is option 6 in the target-palette comparison.
"""


@dataclass(frozen=True)
class Metric:
    """One per-design quantity, and how to label an axis carrying it."""

    key: str
    column: str
    axis_label: str
    design_label: str


METRICS: dict[str, Metric] = {
    "helix": Metric(
        key="helix",
        column="helix (fraction 0-1)",
        axis_label="Median helix (fraction 0-1)",
        design_label="Helix (fraction 0-1)",
    ),
    "sheet": Metric(
        key="sheet",
        column="sheet (fraction 0-1)",
        axis_label="Median sheet (fraction 0-1)",
        design_label="Sheet (fraction 0-1)",
    ),
    "iptm": Metric(
        key="iptm",
        column="iptm (0-1)",
        axis_label="Median interface pTM (0-1)",
        design_label="Interface pTM (0-1)",
    ),
    "plddt": Metric(
        key="plddt",
        column="binder pLDDT (0-100)",
        axis_label="Median binder pLDDT (0-100)",
        design_label="Binder pLDDT (0-100)",
    ),
}
"""Labels are written capitalized rather than left to the house styling, which
capitalizes an axis label only when every letter in it is lowercase and so
would pass over pTM and pLDDT."""


STRIP_DOT_SIZE_POINTS = 104
STRIP_JITTER_WIDTH = 0.16
STRIP_MEDIAN_HALF_WIDTH = 0.28
STRIP_MEDIAN_WIDTH = 3.2
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

    A dot per design rather than a median and an interval, because the shape of
    a column is the result: at weight 1.0 the designs are spread across the
    whole range rather than collected at either end, and an interquartile bar
    drawn there describes a shape the sample does not have.

    The axis is fixed to 0 through 1 and the unsteered arm is a column like any
    other, so a column is read against its neighbours rather than against a
    reference drawn somewhere else. Columns take the weight ramp, the same
    greys the per-step panels give the same arms.
    """
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
    labels = ["Unsteered", *weight_labels]

    for position, arm in enumerate(arms):
        colour = ARM_COLORS[arm]
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
            linewidth=STRIP_MEDIAN_WIDTH,
            solid_capstyle="round",
            zorder=4,
        )

    axis.set_xticks(np.arange(len(arms)))
    axis.set_xticklabels(labels)
    axis.set_xlim(-0.6, len(arms) - 0.4)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(LEGEND_TITLE)
    axis.set_ylabel(spec.design_label)
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure


def scan_violin(designs: pd.DataFrame, arms: list[str], metric: str) -> plt.Figure:
    """Pooled design distributions, with empirical medians and quartiles.

    Violin densities are evaluated only over each group's observed range.
    Each violin has the same maximum width; width does not encode sample size.
    """
    spec = METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
        values = [designs.loc[designs["arm"] == arm, spec.column].to_numpy() for arm in arms]
        if any(len(group) == 0 for group in values):
            raise ValueError("every steering weight needs at least one design")
        positions = np.arange(len(arms))
        violins = axis.violinplot(
            values,
            positions=positions,
            widths=0.8,
            points=200,
            showextrema=False,
            bw_method="scott",
        )
        fill_color = STATES[metric].colour if metric in STATES else ERROR_COLOR
        for body in violins["bodies"]:
            body.set_alpha(1.0)
            body.set_facecolor(fill_color)
            body.set_edgecolor("none")
            body.set_linewidth(0)
        for position, group in zip(positions, values, strict=True):
            low, median, high = np.quantile(group, [0.25, 0.5, 0.75])
            axis.vlines(position, low, high, color="black", linewidth=3, zorder=3)
            axis.scatter(
                position,
                median,
                s=26,
                color=apc.parchment.hex_code,
                edgecolor=ERROR_COLOR,
                linewidth=0.8,
                zorder=4,
            )
        axis.set_xticks(positions, [ARM_LABELS[arm] for arm in arms])
        axis.set_xlim(-0.6, len(arms) - 0.4)
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_xlabel(LEGEND_TITLE)
        axis.set_ylabel(spec.design_label)
        apc.mpl.style_plot(axis, categorical_axes="x")
        figure.tight_layout()
        return figure


def scan_boxplots(
    designs: pd.DataFrame,
    target_order: list[str],
    arms: list[str],
    metrics: tuple[str, ...] = ("helix", "sheet", "loop"),
) -> plt.Figure:
    """A compact grid with shared target headings and steering-weight labels."""
    # Reduce category spacing and canvas width together to retain box width on the page.
    positions = np.arange(len(arms)) * 0.82
    figure, axes = plt.subplots(
        len(metrics),
        len(target_order),
        figsize=(6.25, 1.6 * len(metrics) + 0.4),
        dpi=160,
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for row, metric in enumerate(metrics):
        state = STATES.get(metric)
        if state is not None:
            value_column = state.design_column
            label = f"{state.label} fraction"
            limits = (-0.035, 1.035)
            ticks = [0, 0.25, 0.5, 0.75, 1]
        else:
            value_column = METRICS[metric].column
            label = {"iptm": "ipTM", "plddt": "Binder pLDDT"}[metric]
            limits = confidence_limits(designs, metric, target_order, arms)
            ticks = [
                tick
                for tick in MaxNLocator(nbins=4).tick_values(*limits)
                if limits[0] <= tick <= limits[1]
            ]
        # The same designs receive the same horizontal offsets in every row.
        generator = np.random.default_rng(42)
        for column, (axis, target) in enumerate(zip(axes[row], target_order, strict=True)):
            for position, arm in zip(positions, arms, strict=True):
                values = designs.loc[
                    (designs["target"] == target) & (designs["arm"] == arm),
                    value_column,
                ].to_numpy()
                axis.boxplot(
                    values,
                    positions=[position],
                    widths=0.62,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    whis=1.5,
                    boxprops={
                        "facecolor": to_rgba(
                            state.colour if state is not None else TARGET_COLORS[target], 0.75
                        ),
                        "edgecolor": "black",
                        "linewidth": 0.6,
                    },
                    whiskerprops={"color": "black", "linewidth": 0.6},
                    capprops={"color": "black", "linewidth": 0.6},
                    medianprops={"color": "black", "linewidth": 0.9},
                )
                axis.scatter(
                    position + generator.uniform(-0.19, 0.19, len(values)),
                    values,
                    s=4,
                    color=ERROR_COLOR,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.1,
                    zorder=3,
                )
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=10, pad=7)
            axis.set_xticks(
                positions,
                [f"{float(ARM_WEIGHT_LABELS[arm]):g}" for arm in arms],
            )
            axis.set_xlim(positions[0] - 0.48, positions[-1] + 0.48)
            axis.set_ylim(*limits)
            apc.mpl.style_plot(axis, categorical_axes="x")
            axis.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
            axis.tick_params(axis="both", labelsize=9, pad=3)
            axis.tick_params(axis="y", left=True, labelleft=column == 0, length=2.5)
            axis.tick_params(
                axis="x",
                bottom=row == len(metrics) - 1,
                labelbottom=row == len(metrics) - 1,
            )
            axis.spines["bottom"].set_visible(True)
            axis.spines["left"].set_visible(True)
            if column == 0:
                axis.set_ylabel(label, fontsize=10)
    height = figure.get_figheight()
    figure.subplots_adjust(
        left=0.104,
        right=0.994,
        bottom=0.52 / height,
        top=1 - 0.312 / height,
        wspace=0.08,
        hspace=0.12,
    )
    figure.text(0.549, 0.104 / height, "Steering weight", ha="center", fontsize=10)
    return figure


def scan_lines(
    designs: pd.DataFrame,
    target_order: list[str],
    arms: list[str],
    metric: str,
    legend_loc: str,
) -> plt.Figure:
    """Median of one quantity against steering weight, a line per target.

    Targets carry a hue each and the panel carries a legend, which needs the
    legend to sit where the lines are not. Helix falls to the floor and leaves
    the top right of its panel empty; sheet rises and leaves the top left. The
    corner is therefore an argument rather than a default.

    Weight is placed at equal spacing rather than in proportion, so the x axis
    is a sequence of conditions and not a numeric scale. Reading a slope off it
    as a rate would be wrong; what it shows is order and direction.
    """
    spec = METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _scan_lines(spec, designs, target_order, arms, legend_loc)


def _scan_lines(
    spec: Metric,
    designs: pd.DataFrame,
    target_order: list[str],
    arms: list[str],
    legend_loc: str,
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

    for target, medians in zip(target_order, rows, strict=True):
        axis.plot(
            positions,
            medians,
            color=TARGET_COLORS[target],
            linewidth=SCAN_LINE_WIDTH,
            marker="o",
            markersize=SCAN_MARKER_SIZE,
            markeredgecolor=apc.parchment.hex_code,
            markeredgewidth=0.8,
            label=TARGET_LABELS[target],
            zorder=3,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([ARM_LABELS[arm] for arm in arms])
    axis.set_xlim(-0.3, len(arms) - 1 + 0.3)
    axis.set_xlabel(LEGEND_TITLE)
    axis.set_ylabel(spec.axis_label)
    apc.mpl.style_plot(axis, categorical_axes="x")
    axis.legend(
        loc=legend_loc,
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    figure.tight_layout()
    return figure


WIDE_SIZE_INCHES = (11.0, 7.4)
"""Two stacked panels of twenty-five columns each.

The width is capped rather than chosen. The theme scales a figure to its column
only inside a layout cell; a single figure keeps the pixel width Quarto writes
into the tag, which is half the rendered pixels, so at 150 dpi it is 75 CSS
pixels per inch against a body column that stops at 850. Anything past 11.3
inches leaves the page scrolling sideways at every viewport.
"""

BOX_ALPHA = 0.14
BOX_LINE_WIDTH = 1.4
BOX_NEUTRAL = apc.slate.hex_code
"""How the confidence panels draw a box: a washed fill inside solid edges.

The fill is pale enough that the dots read through it, since the dots are the
data and the box is a summary of them.

The box is drawn in slate wherever it sits over its own dots: one neutral
grey, opaque on the edges and washed out behind them. A box summarises the
same thirty points it sits behind, so giving it a hue as well says nothing the
dots have not said. Slate rather than the cool greys of the house palette,
because a blue cast on a grey box reads as a colour of its own.

The fill carries its alpha in its own colour rather than on the patch, which
would fade the edge with it.
"""


def _style_box(
    drawn: dict[str, list], fill: str, edge: str, alpha: float, median_width: float
) -> None:
    """Recolour one drawn boxplot in place."""
    for patch in drawn["boxes"]:
        patch.set_facecolor(to_rgba(fill, alpha))
        patch.set_edgecolor(edge)
        patch.set_linewidth(BOX_LINE_WIDTH)
    for part in ("whiskers", "caps"):
        for line in drawn[part]:
            line.set_color(edge)
            line.set_linewidth(BOX_LINE_WIDTH)
    for line in drawn["medians"]:
        line.set_color(edge)
        line.set_linewidth(median_width)
        line.set_zorder(4)


COST_MEDIAN_WIDTH = 3.0
COST_DOT_SIZE_POINTS = 18
COST_DOT_EDGE_WIDTH = 0.4
COST_AXIS_PAD_FRACTION = 0.04
COST_METRICS = ("iptm", "plddt")


CONFIDENCE_PANEL_SIZE_INCHES = (5.0, 5.4)
CONFIDENCE_BOX_WIDTH = 0.8
CONFIDENCE_JITTER_WIDTH = 0.24
CONFIDENCE_DOT_SIZE_POINTS = 26
CONFIDENCE_FONT_SCALE = 1.3
"""Type runs larger in these panels than elsewhere in the module.

Five panels across a row are each a third of a page column, so type set for a
half-column panel comes out small against the marks once the document has
scaled the image down. The boxes are wide for the same reason: at unit spacing
a 0.55 box leaves almost half the panel as gaps between columns, which is
width the columns themselves can use.
"""


def confidence_limits(
    designs: pd.DataFrame,
    metric: str,
    target_order: list[str],
    arms: list[str],
) -> tuple[float, float]:
    """The y range one metric needs to hold every design in a set of panels.

    Panels laid out by the document are separate figures, so a shared axis has
    to be computed once and handed to each of them. The range is the data and a
    small pad, not zero to one: nothing in these panels is read by length from
    an origin, and the weights move the medians through a band that a
    zero-anchored axis would squeeze into the top third.
    """
    spec = METRICS[metric]
    wanted = designs["target"].isin(target_order) & designs["arm"].isin(arms)
    values = designs.loc[wanted, spec.column]
    if values.empty:
        raise ValueError(f"no designs for {target_order} across {arms}")
    pad = (values.max() - values.min()) * COST_AXIS_PAD_FRACTION
    return float(values.min() - pad), float(values.max() + pad)


def confidence_panel(
    designs: pd.DataFrame,
    metric: str,
    target: str,
    arms: list[str],
    limits: tuple[float, float],
) -> plt.Figure:
    """One target's confidence against steering weight, every design drawn.

    A median says how a typical run comes out and nothing about how the best
    one does, and for a campaign that returns thirty designs and keeps the ones
    clearing a bar those are different questions. Every design is a dot and the
    box over them carries the quartiles, the median and whiskers at 1.5 times
    the interquartile range, so a column answers both: the box for the rate,
    the top of the dot column for the ceiling.

    One target per figure, laid out beside the others by the document rather
    than inside one canvas, so each panel is its own image at the document's
    own width. That costs a shared axis, which ``confidence_limits`` restores
    by measuring the whole set once; without it the panels would each scale to
    their own target and stop being comparable. Which target a panel holds is
    named by the document's own subcaption rather than drawn into the image.

    A panel holds one target and the subcaption says which, so nothing in the
    panel is coloured by target. What colour carries instead is the weight, on
    the same grey ramp the per-step panels give the same arms, over a box in
    the one neutral grey.
    """
    spec = METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE * CONFIDENCE_FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _confidence_panel(spec, designs, target, arms, limits)


def _confidence_panel(
    spec: Metric,
    designs: pd.DataFrame,
    target: str,
    arms: list[str],
    limits: tuple[float, float],
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=CONFIDENCE_PANEL_SIZE_INCHES)
    generator = np.random.default_rng(STRIP_JITTER_SEED)

    for position, arm in enumerate(arms):
        values = designs.loc[
            (designs["target"] == target) & (designs["arm"] == arm), spec.column
        ].to_numpy()
        if not len(values):
            raise ValueError(f"no designs for {target} / {arm}")
        _style_box(
            axis.boxplot(
                values,
                positions=[position],
                widths=CONFIDENCE_BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                zorder=2,
            ),
            BOX_NEUTRAL,
            BOX_NEUTRAL,
            BOX_ALPHA,
            COST_MEDIAN_WIDTH,
        )
        jitter = generator.uniform(-CONFIDENCE_JITTER_WIDTH, CONFIDENCE_JITTER_WIDTH, len(values))
        axis.scatter(
            position + jitter,
            values,
            s=CONFIDENCE_DOT_SIZE_POINTS,
            color=ARM_COLORS[arm],
            edgecolor=apc.parchment.hex_code,
            linewidth=COST_DOT_EDGE_WIDTH,
            zorder=3,
        )

    axis.set_xticks(np.arange(len(arms)))
    axis.set_xticklabels([ARM_WEIGHT_LABELS[arm] for arm in arms])
    axis.set_xlim(-0.6, len(arms) - 0.4)
    axis.set_ylim(*limits)
    axis.set_xlabel(LEGEND_TITLE)
    axis.set_ylabel(spec.design_label)
    apc.mpl.style_plot(axis, categorical_axes="x")
    figure.tight_layout()
    return figure


SCAN_BASELINE_COLOR = apc.cloud.hex_code
SCAN_RAMP_SPAN = (0.0, 0.8)
SCAN_NEVER_LABEL = "never"
SCAN_SWITCH_MARKER_SIZE = 7
SCAN_TIGHT_PAD = 0.3
"""Outer padding these panels leave, against the 1.08 of a standalone figure.

Four of them are composed into a grid by the document, so whatever each one
leaves around its own content is added to the gap between them. The axis
labels set the real margin; this only stops matplotlib adding more.
"""

SCAN_TRACE_CEILING = 0.83
SCAN_COMPOSITION_CEILING = 0.9
"""Where these panels stop, rather than at one.

Both scans are half as wide as the panels they replaced, so the vertical room
matters. The highest per-step median anywhere in either scan is 0.752 and the
highest upper quartile of a finished composition is 0.856, so these ceilings
crop only space nothing reaches.
"""

SCAN_LABEL_ROOM = 1.02
"""How far past the last step the axis runs.

Barely at all: the line labels sit outside the axes and the tight bounding box
takes them in, which is what keeps the traces filling the width they are drawn
on. Holding the labels inside the axes instead costs half the panel to empty
space and leaves the lines looking stubby.
"""
SCAN_BAND_ALPHA = 0.18
"""Where a scan over the step a weight switches takes its colours.

The house ``wine`` gradient rather than the grey weight ramp, because what
varies along these panels is a step and not a weight, and the two scans are
read against each other. Sampled over the first four fifths of the gradient,
six steps are separated by at least 9.0 over normal vision, simulated
protanopia and simulated deuteranopia; the pale end of the gradient is left
out because a line has to hold its own against the chart surface.

The unsteered curve is cloud, a cool grey, which clears every step of that
ramp by 9.5. Bark, the baseline colour everywhere else in this module, reaches
only 5.6 against the middle of the ramp.
"""


def _scan_ramp(count: int) -> list[str]:
    """``count`` colours along the wine gradient, dark first."""
    low, high = SCAN_RAMP_SPAN
    gradient = apc.gradients.wine.to_mpl_cmap()
    return [to_hex(gradient(position)).upper() for position in np.linspace(low, high, count)]


def _scan_label(step: int, never: int) -> str:
    return SCAN_NEVER_LABEL if step == never else str(step)


@dataclass(frozen=True)
class SwitchNotation:
    """How a scan over a switching step names its own lines.

    Each line carries its condition at its right-hand end rather than a legend
    entry, so the wording has to be short enough to sit in the margin of a
    narrow panel and still say which condition it names. The two ends of a
    scan are conditions rather than steps, so they are named rather than
    numbered.
    """

    prefix: str
    baseline: str
    never: str


ONSET_NOTATION = SwitchNotation(prefix="from", baseline="unsteered", never="unsteered")
RELEASE_NOTATION = SwitchNotation(prefix="until", baseline="unsteered", never="always")


def _switch_label(notation: SwitchNotation, step: int, baseline: int, never: int) -> str:
    if step == never:
        return notation.never
    if step == baseline:
        return notation.baseline
    return f"{notation.prefix} {step}"


def schedule_traces(
    trace: pd.DataFrame,
    scan: dict[int, str],
    drawn: tuple[int, ...],
    metric: str,
    notation: SwitchNotation,
    baseline: int,
    never: int,
) -> plt.Figure:
    """Per-step median of one quantity, one line per switching step.

    A dot on each line marks the step its weight switches, which is the whole
    point of the panel: the lines share a trajectory until then and separate
    after. ``drawn`` selects which steps of the scan to draw, because a scan
    wide enough to find the transition is wider than a legible panel, and the
    steps past the transition sit on top of one another.

    Lines carry their condition at their own right-hand end instead of a
    legend. A legend of seven conditions needs either seven rows, which take
    most of a panel this narrow, or one row that does not fit its width, and
    the lines are already separated at the end of the run, which is where the
    labels go.

    The ramp runs darkest for the arm that carried the most steering, which is
    the step furthest from ``baseline`` in either scan: a weight that arrives
    earliest acts for longest, and so does one withdrawn latest.

    ``baseline`` is the step whose arm never applies the weight, drawn in the
    baseline grey rather than on the ramp because it is the reference the rest
    are read against. ``never`` is the step that falls outside the run, which
    carries no marker because nothing switches inside the run and is labelled
    as such. The two coincide for a scan over the step a weight arrives and sit
    at opposite ends of one over the step it is withdrawn.
    """
    spec = TRACE_METRICS[metric]
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        medians = trace.groupby(["arm", "step (index)"])[spec.column].median().reset_index()
        return _schedule_traces(spec, medians, scan, drawn, notation, baseline, never)


def _schedule_traces(
    spec: TraceMetric,
    trace: pd.DataFrame,
    scan: dict[int, str],
    drawn: tuple[int, ...],
    notation: SwitchNotation,
    baseline: int,
    never: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=SCAN_TRACE_SIZE_INCHES)
    on_ramp = [step for step in drawn if step != baseline]
    by_strength = sorted(on_ramp, key=lambda step: -abs(step - baseline))
    ramp = dict(zip(by_strength, _scan_ramp(len(by_strength)), strict=True))

    endpoints: list[float] = []
    colours: list[str] = []
    for step in drawn:
        subset = trace[trace["arm"] == scan[step]]
        if subset.empty:
            raise ValueError(f"no loss log for {scan[step]}")
        median = subset.set_index("step (index)")[spec.column].sort_index()
        colour = SCAN_BASELINE_COLOR if step == baseline else ramp[step]
        axis.plot(
            median.index,
            median.to_numpy(),
            color=colour,
            linewidth=TRACE_LINE_WIDTH,
            zorder=3,
        )
        endpoints.append(float(median.to_numpy()[-1]))
        colours.append(colour)
        if step not in (baseline, never):
            axis.plot(
                [step],
                [median.loc[step]],
                marker="o",
                markersize=SCAN_SWITCH_MARKER_SIZE,
                color=colour,
                markeredgecolor=apc.parchment.hex_code,
                markeredgewidth=1.0,
                zorder=4,
            )

    last_step = float(trace["step (index)"].max())
    placed = _spread_labels(endpoints, LABEL_MIN_GAP_FRACTION, (0.0, SCAN_TRACE_CEILING))
    for step, height, colour in zip(drawn, placed, colours, strict=True):
        axis.text(
            last_step * 1.03,
            height,
            _switch_label(notation, step, baseline, never),
            color=colour,
            va="center",
            ha="left",
            fontsize=plt.rcParams["xtick.labelsize"],
        )

    axis.set_xlim(0, last_step * SCAN_LABEL_ROOM)
    axis.set_xticks([tick for tick in axis.get_xticks() if 0 <= tick <= last_step])
    axis.set_ylim(0.0, SCAN_TRACE_CEILING)
    axis.set_xlabel("Optimization step (index)")
    axis.set_ylabel(f"Median {spec.axis_label}")
    apc.mpl.style_plot(axis)
    figure.tight_layout(pad=SCAN_TIGHT_PAD)
    return figure


def schedule_composition(
    designs: pd.DataFrame,
    scan: dict[int, str],
    axis_label: str,
    never: int,
) -> plt.Figure:
    """Final helix and sheet against the step the weight switches.

    Steps are equally spaced rather than placed in proportion, because the
    scan is dense where the design is still deciding and sparse afterwards,
    and a proportional axis would spend nine tenths of its width on the part
    where nothing changes. The axis is a sequence of conditions.

    Loop is left out although the three states sum to one. What these panels
    are for is the exchange between the two states the term acts on, and a
    third line drawn from their complement adds no measurement.
    """
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _schedule_composition(designs, scan, axis_label, never)


def _schedule_composition(
    designs: pd.DataFrame,
    scan: dict[int, str],
    axis_label: str,
    never: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=SCAN_COMPOSITION_SIZE_INCHES)
    steps = sorted(scan)
    positions = np.arange(len(steps))

    for state in ("helix", "sheet"):
        spec = METRICS[state]
        columns = []
        for step in steps:
            values = designs.loc[designs["arm"] == scan[step], spec.column]
            if values.empty:
                raise ValueError(f"no designs for {scan[step]}")
            columns.append(values)
        axis.fill_between(
            positions,
            [values.quantile(0.25) for values in columns],
            [values.quantile(0.75) for values in columns],
            color=STATES[state].colour,
            alpha=SCAN_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            positions,
            [values.median() for values in columns],
            color=STATES[state].colour,
            linewidth=SCAN_LINE_WIDTH,
            marker="o",
            markersize=SCAN_MARKER_SIZE,
            markeredgecolor=apc.parchment.hex_code,
            markeredgewidth=0.8,
            label=STATES[state].label,
            zorder=3,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([_scan_label(step, never) for step in steps], rotation=45, ha="center")
    axis.set_xlim(-0.4, len(steps) - 0.6)
    axis.set_ylim(0.0, SCAN_COMPOSITION_CEILING)
    axis.set_xlabel(axis_label)
    axis.set_ylabel("Final composition (0-1)")
    apc.mpl.style_plot(axis, categorical_axes="x")
    axis.tick_params(axis="x", which="major", length=plt.rcParams["xtick.major.size"])
    axis.legend(
        loc="upper center",
        bbox_to_anchor=SCAN_LEGEND_ANCHOR,
        ncol=2,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    figure.tight_layout(pad=SCAN_TIGHT_PAD)
    return figure


SCHEDULE_BOX_WIDTH = 0.52
SCHEDULE_MEDIAN_WIDTH = 2.8
SCHEDULE_JITTER_WIDTH = 0.15
"""A box over the dots rather than a bar up from the axis.

The axis starts at its lowest design and the quantity of interest is the
spread, so the box carries the quartiles and the whiskers reach 1.5 times the
interquartile range. Outliers are not drawn on the box, because every design
is already a dot.
"""


def schedule_confidence(
    designs: pd.DataFrame,
    scan: dict[int, str],
    axis_label: str,
    baseline: int,
    never: int,
    limits: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    """Final confidence distributions, with optional shared limits across schedules."""
    figure, axes = plt.subplots(2, 1, figsize=SCAN_TRACE_SIZE_INCHES, dpi=160, sharex=True)
    steps = sorted(scan)
    positions = np.arange(len(steps)) * 0.82
    target = designs["target"].unique().item()
    for axis, key in zip(axes, COST_METRICS, strict=True):
        spec = METRICS[key]
        generator = np.random.default_rng(42)
        for position, step in zip(positions, steps, strict=True):
            values = designs.loc[designs["arm"] == scan[step], spec.column].to_numpy()
            if not len(values):
                raise ValueError(f"no designs for {scan[step]}")
            colour = BASELINE_COLOR if step == baseline else TARGET_COLORS[target]
            axis.boxplot(
                values,
                positions=[position],
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                whis=1.5,
                boxprops={
                    "facecolor": to_rgba(colour, 0.75),
                    "edgecolor": "black",
                    "linewidth": 0.6,
                },
                whiskerprops={"color": "black", "linewidth": 0.6},
                capprops={"color": "black", "linewidth": 0.6},
                medianprops={"color": "black", "linewidth": 0.9},
            )
            axis.scatter(
                position + generator.uniform(-0.19, 0.19, len(values)),
                values,
                s=6,
                color=ERROR_COLOR,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.1,
                zorder=3,
            )
        bounds = (
            limits[key]
            if limits is not None
            else confidence_limits(designs, key, [target], list(scan.values()))
        )
        axis.set_ylim(*bounds)
        axis.set_ylabel({"iptm": "ipTM", "plddt": "Binder pLDDT"}[key], fontsize=13)
        apc.mpl.style_plot(axis, categorical_axes="x")
        ticks = [
            tick
            for tick in MaxNLocator(nbins=4).tick_values(*bounds)
            if bounds[0] <= tick <= bounds[1]
        ]
        axis.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        axis.tick_params(axis="both", labelsize=11, pad=3)
        axis.spines["bottom"].set_visible(True)
        axis.spines["left"].set_visible(True)
    axes[-1].set_xticks(
        positions, [_scan_label(step, never) for step in steps], rotation=45, ha="right"
    )
    axes[-1].set_xlim(positions[0] - 0.48, positions[-1] + 0.48)
    axes[-1].set_xlabel(axis_label, fontsize=13)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    figure.subplots_adjust(left=0.17, right=0.99, bottom=0.19, top=0.97, hspace=0.14)
    return figure


STACK_EDGE_WIDTH = 0.8

STACK_BAR_WIDTH = 0.7
NATURAL_LABEL = "Natural\nproteins"
STACK_TICK_SCALE = 0.72
"""The reference column leads the weights at the same spacing as any of them.

Equal spacing costs the panel the room to spell "Unsteered", which at the
figure's usual tick size is wider than the one unit between ticks. The x tick
labels therefore run smaller here than in the panel beside this one, which is
the cheaper of the two compromises: a reader reads the word, and the axis
spacing stays honest about the six columns being six equal categories.
"""
"""Wider than the grouped bars of Question 1, which share a position between
two bars; a stack has its position to itself."""

LEGEND_BELOW_AXES = (0.5, -0.36)
"""Where a one-row legend sits, in axes fractions, clear of the x axis label.

The legend is built after ``apc.mpl.style_plot`` rather than before it, so the
house separator rule is not added. That rule is inserted into the first column
of the legend alone, which over three columns underlines a third of the row.
"""


def composition_stack(
    designs: pd.DataFrame, target: str | None, arms: list[str], natural: pd.DataFrame
) -> plt.Figure:
    """Seed-averaged secondary-structure composition across the weight scan.

    Passing ``target=None`` pools designs from all targets in the input table.

    The three states sum to the whole scored chain on every design, so their
    means over a seed set sum to one as well and each bar is still a whole
    chain. What the panel carries that a helix axis does not is the exchange
    between states: the helix a weight removes has to be held by something, and
    a stacked bar says by what, without a second axis.

    The natural reference set is the same composition measured the same way, so
    it stacks into a bar like an arm does and answers the question the panel
    otherwise leaves open: what the weight is steering towards. It sits past a
    gap because it is not a weight.

    States take the same three colours as the composition panels in Question 1,
    so a reader who has learned them once does not learn them again.
    """
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    scaled["xtick.labelsize"] = scaled["xtick.labelsize"] * STACK_TICK_SCALE
    with plt.rc_context(scaled):
        return _composition_stack(designs, target, arms, natural)


def _composition_stack(
    designs: pd.DataFrame, target: str | None, arms: list[str], natural: pd.DataFrame
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    selected = designs if target is None else designs[designs["target"] == target]
    rows = {arm: selected[selected["arm"] == arm] for arm in arms}
    counts = {arm: len(frame) for arm, frame in rows.items()}
    if len(set(counts.values())) != 1 or min(counts.values()) == 0:
        raise ValueError(
            f"a mean per arm needs the same non-zero design count in every arm, "
            f"and {target or 'all targets'} has {counts}"
        )

    means = pd.DataFrame(
        {
            state.key: [natural[state.natural_column].mean()]
            + [rows[arm][state.design_column].mean() for arm in arms]
            for state in STATES.values()
        },
        index=["natural", *arms],
    )
    return _composition_means(means, figure, axis)


def composition_means(means: pd.DataFrame) -> plt.Figure:
    """Draw precomputed state means, with the natural reference first."""
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    scaled["xtick.labelsize"] *= STACK_TICK_SCALE
    with plt.rc_context(scaled):
        figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
        return _composition_means(means, figure, axis)


def _composition_means(means, figure, axis) -> plt.Figure:
    arms = list(means.index[1:])
    positions = np.arange(len(arms))
    natural_position = -1.0
    bottoms = np.zeros(len(means))
    for state in STATES.values():
        heights = means[state.key].to_numpy()
        axis.bar(
            [natural_position, *positions],
            heights,
            width=STACK_BAR_WIDTH,
            bottom=bottoms,
            color=state.colour,
            edgecolor=apc.parchment.hex_code,
            linewidth=STACK_EDGE_WIDTH,
            label=state.label,
            zorder=3,
        )
        bottoms += heights

    axis.set_xticks([natural_position, *positions])
    axis.set_xticklabels([NATURAL_LABEL, *(ARM_LABELS[arm] for arm in arms)])
    left, right = natural_position - 0.6, len(arms) - 0.4
    axis.set_xlim(left, right)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(
        LEGEND_TITLE,
        x=((len(arms) - 1) / 2 - left) / (right - left),
        ha="center",
    )
    axis.set_ylabel("Mean chain in each state (0-1)")
    apc.mpl.style_plot(axis, categorical_axes="x")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=LEGEND_BELOW_AXES,
        ncol=len(STATES),
        frameon=False,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    figure.tight_layout()
    return figure


@dataclass(frozen=True)
class TraceMetric:
    """One per-step quantity from the loss log."""

    key: str
    column: str
    axis_label: str


TRACE_METRICS: dict[str, TraceMetric] = {
    "helix": TraceMetric("helix", "helix (fraction 0-1)", "Residues assigned helix (%)"),
    "sheet": TraceMetric("sheet", "sheet (fraction 0-1)", "Residues assigned sheet (%)"),
    "helix_mean": TraceMetric(
        key="helix_mean",
        column="helix_mean (0-1)",
        axis_label="mean helix rung fraction (0-1)",
    ),
    "iptm_proxy": TraceMetric(
        key="iptm_proxy",
        column="distogram iptm proxy (0-1)",
        axis_label="distogram interface pTM proxy (0-1)",
    ),
    "entropy": TraceMetric(
        key="entropy",
        column="sequence entropy (bits)",
        axis_label="sequence entropy (bits/residue)",
    ),
}


def _prior_style(arm: str) -> tuple[str, str]:
    from analysis.targets import arm as get_arm

    spec = get_arm(arm)
    colors = {"uniform": "#60676C", "helix": HELIX_COLOR, "sheet": "#347DA8"}
    return spec.label, colors[spec.initialization_prior]


def _trajectory_axis(axis, metric: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E7E7E7", lw=0.6)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.set_ylabel(TRACE_METRICS[metric].axis_label)
    if metric in {"helix", "sheet"}:
        axis.set(ylim=(-2, 100), yticks=[0, 20, 40, 60, 80, 100])
    elif metric == "helix_mean":
        axis.set_ylim(0, 1)


def prior_trajectory_panel(
    trace: pd.DataFrame,
    means: pd.DataFrame,
    arms: list[str],
    metric: str,
    *,
    individuals: bool = True,
    legend: bool = False,
) -> plt.Figure:
    """One panel from measured traces and caller-computed means indexed by arm/step.

    Filtering coordinate readouts, selecting steps and aggregating matched
    starts belong in the calling analysis. Separate figures allow Quarto to
    arrange panels and supply their captions.
    """
    spec = TRACE_METRICS[metric]
    scale = 100 if metric in {"helix", "sheet"} else 1
    fig, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    for arm in arms:
        label, color = _prior_style(arm)
        if individuals:
            for _, group in trace[trace.arm == arm].groupby("seed"):
                group = group.sort_values("step (index)")
                axis.plot(
                    group["step (index)"],
                    scale * group[spec.column],
                    color=color,
                    alpha=0.18,
                    lw=0.9,
                )
        mean = means.xs(arm, level="arm")[spec.column].sort_index()
        axis.plot(
            mean.index,
            scale * mean,
            color=color,
            lw=2.7,
            label=label,
            marker=None if individuals else "o",
            ms=5,
        )
    _trajectory_axis(axis, metric)
    axis.set_xlabel("Sequence updates completed")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if metric == "entropy":
        from analysis.priors import AA

        uniform = np.log2(len(AA))
        axis.axhline(uniform, color="#999999", lw=1, ls=(0, (6, 4)), zorder=0)
        axis.annotate(
            f"Uniform: {uniform:.3f} bits",
            xy=(0.98, uniform),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=10,
        )
        axis.set_ylim(top=uniform + 0.1)
    if legend:
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            loc="lower left",
            fontsize=11,
        )
    return fig


def matched_composition_panel(
    final: pd.DataFrame, means: pd.DataFrame, arms: list[str], metric: str
) -> plt.Figure:
    """Connect matched starts at one caller-selected state; black bars show supplied means."""
    column = TRACE_METRICS[metric].column
    values = 100 * final.pivot(index="seed", columns="arm", values=column).reindex(columns=arms)
    fig, axis = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    labels, colors = zip(*[_prior_style(arm) for arm in arms], strict=True)
    for offset, (_, row) in zip(
        np.linspace(-0.18, 0.18, len(values)), values.iterrows(), strict=True
    ):
        x = np.arange(len(arms)) + offset
        axis.plot(x, row.to_numpy(), color="#9B9B9B", alpha=0.6, lw=0.9, zorder=1)
        axis.scatter(x, row.to_numpy(), c=colors, s=42, zorder=2)
    for x, arm in enumerate(arms):
        mean = 100 * means.loc[arm, column]
        axis.plot([x - 0.20, x + 0.20], [mean, mean], color="black", lw=2.5, zorder=3)
        axis.text(x + 0.24, max(mean, 1.5), f"{round(mean, 10):.1f}%", va="center", fontsize=11)
    _trajectory_axis(axis, metric)
    axis.set(xticks=range(len(arms)), xticklabels=labels, xlim=(-0.4, len(arms) - 0.3))
    return fig


SEED_LINE_WIDTH = 0.8
SEED_LINE_ALPHA = 0.45
SEED_LINE_COLOR = HELIX_COLOR
"""The helix hue, because what these lines carry is the helix term rather than
the weight that produced them; the arm is one condition and needs no colour."""


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
            ordered[spec.column].to_numpy(),
            color=SEED_LINE_COLOR,
            linewidth=SEED_LINE_WIDTH,
            alpha=SEED_LINE_ALPHA,
            zorder=2,
        )

    grouped = subset.groupby("step (index)")[spec.column].median()
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
    medians = trace.groupby(["arm", "step (index)"])[spec.column].median().reset_index()
    return _trace_summary(spec, medians, arms)


def trace_summary(medians: pd.DataFrame, metric: str, arms: list[str]) -> plt.Figure:
    """Draw precomputed per-step medians for each condition."""
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    with plt.rc_context(scaled):
        return _trace_summary(TRACE_METRICS[metric], medians, arms)


def _trace_summary(spec, trace, arms) -> plt.Figure:
    if spec.key == "entropy":
        return _entropy_trace_summary(spec, trace, arms)
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    endpoints = []
    for arm in arms:
        subset = trace[trace["arm"] == arm].sort_values("step (index)")
        steps = subset["step (index)"].to_numpy()
        medians = subset[spec.column].to_numpy()
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
    placed = _spread_labels(endpoints, span * LABEL_MIN_GAP_FRACTION, axis.get_ylim())
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


def _entropy_trace_summary(spec, trace, arms) -> plt.Figure:
    """Show soft-sequence entropy against the uniform initialization baseline."""
    figure, axis = plt.subplots(figsize=PANEL_SIZE_INCHES)
    for arm in arms:
        subset = trace[trace["arm"] == arm].sort_values("step (index)")
        axis.plot(
            subset["step (index)"],
            subset[spec.column],
            color=ARM_COLORS[arm],
            linewidth=TRACE_LINE_WIDTH,
            label=ARM_LABELS[arm],
            zorder=3,
        )
    # Cysteine is excluded, leaving 19 allowed amino acids at each position.
    axis.axhline(np.log2(19), color="#737373", linewidth=1, linestyle=(0, (6, 4)))
    last_step = trace["step (index)"].max()
    annotation_size = plt.rcParams["xtick.labelsize"]
    axis.text(
        last_step, 4.34, "Uniform: 4.248 bits", ha="right", va="bottom",
        fontsize=annotation_size,
    )
    axis.set(
        xlim=(0, last_step), ylim=(0, 4.7),
        xlabel="optimization step (index)", ylabel=spec.axis_label,
    )
    axis.set_xticks([0, 50, 100, last_step])
    apc.mpl.style_plot(axis)
    axis.legend(
        loc="lower left", bbox_to_anchor=(0.05, 0.06),
        ncol=1, frameon=False, fontsize=annotation_size,
        handlelength=1.3, handletextpad=0.4,
    )
    figure.tight_layout()
    return figure


def schedule_trace_panels(medians, scans, drawn_steps) -> list[plt.Figure]:
    """Draw onset and withdrawal traces from precomputed per-step medians."""
    from analysis.targets import SGD_STEPS

    panels = []
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
    for name, notation, baseline in (
        ("onset", ONSET_NOTATION, SGD_STEPS),
        ("withdrawal", RELEASE_NOTATION, 0),
    ):
        with plt.rc_context(scaled):
            panels.append(
                _schedule_traces(
                    TRACE_METRICS["helix_mean"],
                    medians[name],
                    scans[name],
                    drawn_steps[name],
                    notation,
                    baseline,
                    SGD_STEPS,
                )
            )
    return panels


def initialization_prior(
    prior: pd.Series, initialized: pd.Series, state: str, chosen: str, *, legend: bool = False
) -> plt.Figure:
    """Compare supplied prior and initialized probabilities at one sequence position."""
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    prior = prior.reindex(amino_acids, fill_value=0)
    initialized = initialized.reindex(amino_acids, fill_value=0)
    color = {"uniform": LOOP_COLOR, "helix": HELIX_COLOR, "sheet": SHEET_COLOR}[state]
    selected_color = "#493466"
    colors = [selected_color if aa == chosen else color for aa in amino_acids]
    outline_colors = [selected_color if aa == chosen else ERROR_COLOR for aa in amino_acids]
    positions = np.arange(len(amino_acids))
    offset, width = 0.2, 0.36
    figure, axis = plt.subplots(figsize=(8, 2.6))
    axis.bar(
        positions - offset, 100 * prior, width=width, facecolor="none",
        edgecolor=outline_colors, linewidth=1, zorder=3,
    )
    axis.bar(positions + offset, 100 * initialized, width=width, color=colors, zorder=3)
    axis.set(
        xlim=(-0.7, 19.7),
        ylim=(0, 1.18 * 100 * max(prior.max(), initialized.max())),
        ylabel="Probability (%)",
    )
    axis.set_xticks(positions, amino_acids)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", length=0, pad=5)
    for tick, aa in zip(axis.get_xticklabels(), amino_acids, strict=True):
        if aa == chosen:
            tick.set_color(selected_color)
            tick.set_fontweight("bold")
        elif aa == "C":
            tick.set_color(BASELINE_COLOR)
    axis.annotate(
        f"{100 * initialized[chosen]:.2f}%",
        (amino_acids.index(chosen) + offset, 100 * initialized[chosen]),
        xytext=(0, 5), textcoords="offset points", ha="center",
        color=selected_color,
    )
    if legend:
        axis.legend(
            handles=[Patch(facecolor=selected_color, label="Chosen")],
            loc="upper right", frameon=False,
        )
    figure.tight_layout()
    return figure
