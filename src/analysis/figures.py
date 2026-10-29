"""Render publication figures from measurements and summaries."""

from __future__ import annotations
from dataclasses import dataclass

import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba
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
"""Scale factor for panel font sizes."""

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
"""Steered-arm colours ordered from lower to higher weight."""

BASELINE_COLOR = apc.chateau.hex_code
"""Colour for the unsteered arm, the lightest step of the weight ramp."""

ARM_COLORS = {
    "vanilla": BASELINE_COLOR,
    "helix_mean_w05": WEIGHT_RAMP[0],
    "helix_mean_w10": WEIGHT_RAMP[1],
    "helix_mean_w20": WEIGHT_RAMP[2],
    "helix_mean_w50": WEIGHT_RAMP[3],
}
"""Map weight-scan arms to their colours."""

ARM_LABELS = {
    "vanilla": "Unsteered",
    "helix_mean_w05": "0.5",
    "helix_mean_w10": "1.0",
    "helix_mean_w20": "2.0",
    "helix_mean_w50": "5.0",
}
"""Legend labels for weight-scan arms."""

ARM_WEIGHT_LABELS = {
    "vanilla": "0",
    "helix_mean_w05": "0.5",
    "helix_mean_w10": "1.0",
    "helix_mean_w20": "2.0",
    "helix_mean_w50": "5.0",
}
"""Numeric steering-weight labels, including zero for the unsteered arm."""

LEGEND_TITLE = "Steering weight"

LABEL_MIN_GAP_FRACTION = 0.07
"""Scale for direct-label spacing."""


@dataclass(frozen=True)
class _LabelBand:
    """Allowed label centres and the minimum vertical gap between adjacent labels."""

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
    """Separate colliding labels while preserving their value order."""
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
"""Marker size for scan summary lines, in points."""

SCAN_TRACE_SIZE_INCHES = (5.25, 5.8)
SCAN_COMPOSITION_SIZE_INCHES = (5.3, 5.8)
"""Figure size for schedule-composition panels, in inches."""

SCAN_LEGEND_ANCHOR = (0.5, -0.42)

PANEL_SIZE_INCHES = (6.8, 5.4)
"""Default figure size for individual panels, in inches."""

WIDE_PANEL_SIZE_INCHES = (10.0, 5.4)
"""Figure size for panels stacked at full text width, in inches."""


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
        label="Sheet",
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
    """Compare chain-mean composition in five length bins, with standard errors.

    Natural-reference bars are unfilled; design bars use the state colour.
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
    axis.set_ylabel(f"mean {spec.label.lower()} (fraction of binder residues, 0–1)")
    apc.mpl.style_plot(axis, categorical_axes="x")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.04, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.0,
    )
    figure.tight_layout()
    return figure


TARGET_LABELS = {
    "egfr_d3": "EGFR",
    "pdgfrb_d2d3": "PDGFRβ",
    "pdl1_igv": "PD-L1",
    "ctla4_igv": "CTLA-4",
    "cd45_d3d4": "CD45",
}
"""Short protein names for target axis labels."""

TARGET_COLORS = {
    "egfr_d3": "#305F8C",
    "pdgfrb_d2d3": "#2B8C91",
    "pdl1_igv": "#6C70B5",
    "ctla4_igv": "#6B9A80",
    "cd45_d3d4": "#A16F9B",
}
"""Target colours independent of plotting order."""


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
"""Value columns and axis labels for per-design metrics."""


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
    """Draw individual designs and medians by steering weight on a fixed 0–1 axis."""
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
        figure, axis = plt.subplots(figsize=WIDE_PANEL_SIZE_INCHES)
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
    """Plot per-target medians at equally spaced steering-weight categories.

    ``legend_loc`` sets the legend position.
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
"""Figure size for wide layouts, in inches."""

BOX_ALPHA = 0.14
BOX_LINE_WIDTH = 1.4
BOX_NEUTRAL = apc.slate.hex_code
"""Neutral colour for confidence boxes with translucent fills."""


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
"""Additional font scale for confidence panels."""


def confidence_limits(
    designs: pd.DataFrame,
    metric: str,
    target_order: list[str],
    arms: list[str],
) -> tuple[float, float]:
    """Return the selected designs’ minimum and maximum with proportional padding."""
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
    """Draw one target’s confidence scores and boxplots using the supplied axis limits."""
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
"""Padding passed to tight_layout for schedule panels."""

SCAN_TRACE_CEILING = 0.83
SCAN_COMPOSITION_CEILING = 0.9
"""Upper y-axis limit for schedule-composition panels."""

SCAN_LABEL_ROOM = 1.02
"""Factor extending the trace x-axis beyond the last optimization step."""
SCAN_BAND_ALPHA = 0.18
"""Opacity of the interquartile bands in schedule-composition panels."""


def _scan_ramp(count: int) -> list[str]:
    """``count`` colours along the wine gradient, dark first."""
    low, high = SCAN_RAMP_SPAN
    gradient = apc.gradients.wine.to_mpl_cmap()
    return [to_hex(gradient(position)).upper() for position in np.linspace(low, high, count)]


def _scan_label(step: int, never: int) -> str:
    return SCAN_NEVER_LABEL if step == never else str(step)


@dataclass(frozen=True)
class SwitchNotation:
    """Short labels for switching steps and the scan endpoints."""

    prefix: str
    baseline: str
    never: str


ONSET_NOTATION = SwitchNotation(prefix="From", baseline="Unsteered", never="Unsteered")
RELEASE_NOTATION = SwitchNotation(prefix="Until", baseline="Unsteered", never="Always")


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
    """Plot per-step medians for selected switching conditions, with direct labels.

    ``baseline`` selects the unsteered reference. ``never`` selects the endpoint
    whose switch lies outside the run. Mark other switches at their step.
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
    for step, height in zip(drawn, placed, strict=True):
        axis.text(
            last_step * 1.03,
            height,
            _switch_label(notation, step, baseline, never),
            color="black",
            va="center",
            ha="left",
            fontsize=plt.rcParams["xtick.labelsize"],
        )

    axis.set_xlim(0, last_step * SCAN_LABEL_ROOM)
    axis.set_xticks([tick for tick in axis.get_xticks() if 0 <= tick <= last_step])
    axis.set_ylim(0.0, SCAN_TRACE_CEILING)
    axis.set_xlabel("Optimization step (index)")
    axis.set_ylabel(f"Median {spec.axis_label[:1].lower()}{spec.axis_label[1:]}")
    apc.mpl.style_plot(axis)
    figure.tight_layout(pad=SCAN_TIGHT_PAD)
    return figure


def schedule_composition(
    designs: pd.DataFrame,
    scan: dict[int, str],
    axis_label: str,
    never: int,
) -> plt.Figure:
    """Plot final helix and sheet medians and quartiles by switching-step category."""
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
"""Horizontal jitter half-width for schedule dots."""


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

LEGEND_ABOVE_AXES = (0.5, 1.02)
"""Legend anchor above the axes, in axes coordinates."""


def composition_stack(
    designs: pd.DataFrame, target: str | None, arms: list[str], natural: pd.DataFrame
) -> plt.Figure:
    """Plot mean state fractions, with the natural reference first at equal spacing.

    ``target=None`` pools designs across targets. Require the same nonzero design
    count in every arm.
    """
    scaled = {key: plt.rcParams[key] * FONT_SCALE for key in FONT_SIZE_KEYS}
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
    with plt.rc_context(scaled):
        figure, axis = plt.subplots(figsize=WIDE_PANEL_SIZE_INCHES)
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
    axis.set_ylabel("Mean chain in each state (0–1)")
    apc.mpl.style_plot(axis, categorical_axes="x")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=LEGEND_ABOVE_AXES,
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
        axis_label="Helix proxy (0–1)",
    ),
    "iptm_proxy": TraceMetric(
        key="iptm_proxy",
        column="distogram iptm proxy (0-1)",
        axis_label="Distogram ipTM proxy (0–1)",
    ),
    "entropy": TraceMetric(
        key="entropy",
        column="sequence entropy (bits)",
        axis_label="sequence entropy (bits/residue)",
    ),
}


SEED_LINE_WIDTH = 0.8
SEED_LINE_ALPHA = 0.45
SEED_LINE_COLOR = HELIX_COLOR
"""Colour used for individual seed trajectories."""


def seed_traces(trace: pd.DataFrame, target: str, arm: str, metric: str) -> plt.Figure:
    """Plot each seed’s trajectory and the per-step median for one target and arm."""
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
        label=f"Median of {subset['seed'].nunique()} seeds",
        zorder=3,
    )

    axis.set_xlabel("optimization step (index)")
    axis.set_ylabel(spec.axis_label)
    axis.set_xlim(0, subset["step (index)"].max())
    apc.mpl.style_plot(axis)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
    )
    figure.tight_layout()
    return figure


def trace_lines(trace: pd.DataFrame, metric: str, arms: list[str]) -> plt.Figure:
    """Plot the per-step median across designs for each requested arm."""
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
    lower, upper = axis.get_ylim()
    span = upper - lower
    placed = _spread_labels(endpoints, span * LABEL_MIN_GAP_FRACTION, axis.get_ylim())
    for arm, height in zip(arms, placed, strict=True):
        axis.text(
            last_step * 1.02,
            height,
            ARM_LABELS[arm],
            color="black",
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
    """Plot soft-sequence entropy with a uniform 19-amino-acid reference line."""
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
        last_step,
        4.34,
        "Uniform: 4.248 bits",
        ha="right",
        va="bottom",
        fontsize=annotation_size,
    )
    axis.set(
        xlim=(0, last_step),
        ylim=(0, 4.7),
        xlabel="optimization step (index)",
        ylabel=spec.axis_label,
    )
    axis.set_xticks([0, 50, 100, last_step])
    apc.mpl.style_plot(axis)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.05, 0.06),
        ncol=1,
        frameon=False,
        fontsize=annotation_size,
        handlelength=1.3,
        handletextpad=0.4,
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
