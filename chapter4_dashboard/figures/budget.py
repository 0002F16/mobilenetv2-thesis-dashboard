from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import VARIANT_ORDER


def build_budget_bars(
    df_eff: pd.DataFrame,
    cmap: dict[str, str],
    theme: str,
    variants: list[str] | None = None,
) -> go.Figure:
    """
    Horizontal grouped bars: Params and FLOPs, with +10% reference vs baseline.
    """
    if df_eff.empty:
        return go.Figure()
    eff = df_eff.set_index("variant")
    if "Baseline" not in eff.index:
        return go.Figure()

    base_p = float(eff.loc["Baseline", "params_M"])
    base_f = float(eff.loc["Baseline", "flops_M"])
    if variants is None:
        variants = [v for v in ["Baseline", "DualConv-only", "ECA-only", "Hybrid"] if v in eff.index]
    else:
        variants = [v for v in variants if v in eff.index]

    vals_p = [float(eff.loc[v, "params_M"]) / base_p * 100.0 - 100.0 for v in variants]
    vals_f = [float(eff.loc[v, "flops_M"]) / base_f * 100.0 - 100.0 for v in variants]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Params",
            y=variants,
            x=vals_p,
            orientation="h",
            marker_color=[cmap.get(v, "#999999") for v in variants],
        )
    )
    fig.add_trace(
        go.Bar(
            name="FLOPs",
            y=variants,
            x=vals_f,
            orientation="h",
            marker_color=[cmap.get(v, "#999999") for v in variants],
            opacity=0.65,
        )
    )

    fig.update_layout(
        barmode="group",
        template=plotly_template(theme),  # type: ignore[arg-type]
        title="Parameter and FLOP Overhead Relative to Baseline MobileNetV2 (< +10% budget)",
        xaxis_title="Overhead vs Baseline (%)",
        yaxis_title="Variant",
        **plotly_font_kwargs(),
        height=max(360, 90 * len(variants)),
        legend_title_text="Metric",
    )
    # reference lines
    fig.add_vline(x=10, line_width=1, line_dash="dash", line_color="#666666")
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="#333333")
    return fig


def build_budget_compliance_normalized_bars(
    df_eff: pd.DataFrame,
    theme: str,
    variants: list[str] | None = None,
) -> go.Figure:
    """
    Grouped vertical bars: x = variant; Params, FLOPs, and model size as % of Baseline (100%).
    """
    if df_eff.empty:
        return go.Figure()
    eff = df_eff.set_index("variant")
    if "Baseline" not in eff.index:
        return go.Figure()
    base_p = float(eff.loc["Baseline", "params_M"])
    base_f = float(eff.loc["Baseline", "flops_M"])
    base_s = float(eff.loc["Baseline", "size_mb"])
    if base_p <= 0 or base_f <= 0 or base_s <= 0:
        return go.Figure()

    if variants is None:
        variants = [v for v in VARIANT_ORDER if v in eff.index]
    else:
        variants = [v for v in variants if v in eff.index]
    if not variants:
        return go.Figure()

    y_p = [float(eff.loc[v, "params_M"]) / base_p * 100.0 for v in variants]
    y_f = [float(eff.loc[v, "flops_M"]) / base_f * 100.0 for v in variants]
    y_s = [float(eff.loc[v, "size_mb"]) / base_s * 100.0 for v in variants]

    # Metric colors (Params / FLOPs / Size), distinct from per-variant coloring used elsewhere.
    c_params, c_flops, c_size = "#4C72B0", "#DD8452", "#55A868"

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Params (M)", x=variants, y=y_p, marker_color=c_params))
    fig.add_trace(go.Bar(name="FLOPs (M)", x=variants, y=y_f, marker_color=c_flops))
    fig.add_trace(go.Bar(name="Size (MB)", x=variants, y=y_s, marker_color=c_size))

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        title="Budget compliance: Params, FLOPs, and size (% of Baseline)",
        xaxis_title="Variant",
        yaxis_title="% of Baseline (100% = parity)",
        barmode="group",
        bargap=0.14,
        bargroupgap=0.04,
        legend_title_text="Metric",
        **plotly_font_kwargs(),
        height=440,
        margin=dict(l=56, r=24, t=56, b=80),
    )
    fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="#666666")
    fig.update_yaxes(rangemode="tozero")
    return fig

