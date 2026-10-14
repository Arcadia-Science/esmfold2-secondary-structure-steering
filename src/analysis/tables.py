"""Render publication tables whose measurements and summaries are computed in the notebook."""

import pandas as pd
from IPython.display import HTML
from pandas.io.formats.style import Styler

from analysis.figures import ARM_LABELS, TARGET_LABELS


def _campaign_style(table: pd.DataFrame) -> Styler:
    """Label campaign axes and separate the outer row groups visually."""
    table = (
        table.rename(index=TARGET_LABELS, level="Target")
        .rename(columns={**ARM_LABELS, "vanilla": "0 (unsteered)"})
        .rename_axis(columns=None)
    )
    table.columns = pd.MultiIndex.from_product([["Steering weight"], table.columns])
    styles = [
        {"selector": "", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        {
            "selector": "th, td",
            "props": [("padding", "0.35rem 0.75rem"), ("white-space", "nowrap")],
        },
        {
            "selector": "td, th.col_heading",
            "props": [
                ("min-width", "4rem"),
                ("text-align", "center"),
                ("font-variant-numeric", "tabular-nums"),
            ],
        },
        {
            "selector": "th.col_heading.level0",
            "props": [("padding-bottom", "0.5rem"), ("border-bottom", "1px solid #bfc1c3")],
        },
        {
            "selector": "th.row_heading, th.index_name",
            "props": [("text-align", "left"), ("vertical-align", "middle")],
        },
    ]
    previous_group = None
    for row_number, group in enumerate(table.index.get_level_values(0), start=1):
        if row_number == 1 or group != previous_group:
            styles.append(
                {
                    "selector": f"tbody tr:nth-child({row_number}) > *",
                    "props": [("border-top", "1px solid #bfc1c3")],
                }
            )
        previous_group = group
    return table.style.set_table_styles(styles)


def confidence(table: pd.DataFrame) -> HTML:
    """Format already calculated confidence summaries at each metric's precision."""
    return HTML(
        _campaign_style(table)
        .format("{:.3f}", subset=pd.IndexSlice[("ipTM", slice(None)), :])
        .format("{:.1f}", subset=pd.IndexSlice[("Binder pLDDT", slice(None)), :])
        .to_html()
    )


def screen_counts(table: pd.DataFrame) -> HTML:
    """Format already calculated screen counts and rates, emphasizing totals."""
    return HTML(
        _campaign_style(table)
        .set_table_styles(
            [
                {
                    "selector": "th.row_heading.level0",
                    "props": [("white-space", "normal"), ("max-width", "12rem")],
                }
            ],
            overwrite=False,
        )
        .format("{:.0f}")
        .format("{:.1%}", subset=pd.IndexSlice[(slice(None), "Pass rate"), :])
        .set_properties(
            subset=pd.IndexSlice[(slice(None), ["Total", "Pass rate"]), :],
            **{"font-weight": "bold"},
        )
        .to_html()
    )


def initialization_summary(table: pd.DataFrame) -> HTML:
    """Format precomputed initialization means and screen counts."""
    from analysis.initialization import LABELS

    return HTML(
        table.rename(index=LABELS)
        .style.format(
            {
                "Mean helix (%)": "{:.1f}",
                "Mean sheet (%)": "{:.1f}",
                "Mean ipTM": "{:.3f}",
                "Mean pLDDT": "{:.1f}",
                "Screen passes / 30": "{:.0f}",
            }
        )
        .to_html()
    )
