from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import DATASET_ORDER, VARIANT_ORDER


def build_per_dataset_latency_bars(
    lat_wide: pd.DataFrame,
    cmap: dict[str, str],
    theme: str,
) -> go.Figure:
    """
    Grouped vertical bars: x = dataset, one series per variant (mean latency ms).
    """
    if lat_wide is None or lat_wide.empty or "Variant" not in lat_wide.columns:
        return go.Figure()
    datasets = [c for c in DATASET_ORDER if c in lat_wide.columns]
    if not datasets:
        return go.Figure()
    variants = [v for v in VARIANT_ORDER if v in set(lat_wide["Variant"].astype(str))]
    if not variants:
        return go.Figure()

    fig = go.Figure()
    for v in variants:
        row = lat_wide[lat_wide["Variant"] == v]
        if row.empty:
            continue
        try:
            y = [float(row[ds].iloc[0]) for ds in datasets]
        except (TypeError, ValueError, KeyError):
            continue
        fig.add_trace(
            go.Bar(
                name=v,
                x=datasets,
                y=y,
                marker_color=cmap.get(v, "#999999"),
                marker_line_width=0,
            )
        )

    if not fig.data:
        return go.Figure()

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        title="Per-dataset inference latency (ms) by variant",
        xaxis_title="Dataset",
        yaxis_title="Latency (ms / image)",
        barmode="group",
        bargap=0.12,
        bargroupgap=0.06,
        legend_title_text="Variant",
        **plotly_font_kwargs(),
        height=420,
        margin=dict(l=56, r=24, t=56, b=48),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig
