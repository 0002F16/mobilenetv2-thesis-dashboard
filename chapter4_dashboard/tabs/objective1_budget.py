from __future__ import annotations

import pandas as pd
import streamlit as st

from chapter4_dashboard.figures.budget import build_budget_bars
from chapter4_dashboard.utils.export import figure_to_png_bytes, dataframe_to_csv_bytes
from chapter4_dashboard.utils.styling import style_budget_table


def _budget_table(df_eff: pd.DataFrame) -> pd.DataFrame:
    if df_eff.empty:
        return pd.DataFrame()
    eff = df_eff.set_index("variant")
    if "Baseline" not in eff.index:
        return pd.DataFrame()
    base_p = float(eff.loc["Baseline", "params_M"])
    base_f = float(eff.loc["Baseline", "flops_M"])

    rows = []
    for v in ["Baseline", "DualConv-only", "ECA-only", "Hybrid"]:
        if v not in eff.index:
            continue
        params = float(eff.loc[v, "params_M"])
        flops = float(eff.loc[v, "flops_M"])
        dpp = (params - base_p) / base_p * 100.0
        dff = (flops - base_f) / base_f * 100.0
        # Budget constraint: overhead must be < +10% vs baseline (negative deltas allowed).
        compliant = (dpp < 10.0) and (dff < 10.0)
        rows.append(
            {
                "Variant": v,
                "Params (M)": params,
                "Δ Params %": dpp,
                "FLOPs (M)": flops,
                "Δ FLOPs %": dff,
                "Size (MB)": float(eff.loc[v, "size_mb"]),
                "Latency (ms)": float(eff.loc[v, "latency_ms"]),
                "Budget Compliant": "✓" if compliant else "✗",
            }
        )
    return pd.DataFrame(rows)


def render_tab_budget(df_eff: pd.DataFrame, cmap: dict[str, str]) -> None:
    st.subheader("Tab 2 — Objective 1: Budget Compliance")
    st.info(
        "“Specific Objective 1: Ensure all proposed architectural variants remain under a +10% overhead "
        "vs the baseline MobileNetV2 parameter and FLOP budget.”"
    )

    tbl = _budget_table(df_eff)
    if tbl.empty:
        st.warning("Budget table requires `df_efficiency` with a Baseline row.")
        return

    st.markdown("**Budget Compliance Summary**")
    sty = style_budget_table(tbl).format(
        {
            "Params (M)": "{:.2f}",
            "Δ Params %": "{:+.2f}",
            "FLOPs (M)": "{:.1f}",
            "Δ FLOPs %": "{:+.2f}",
            "Size (MB)": "{:.2f}",
            "Latency (ms)": "{:.2f}",
        }
    )
    st.dataframe(sty, use_container_width=True, height=240)
    st.download_button(
        "Download Budget table (CSV)",
        dataframe_to_csv_bytes(tbl),
        file_name="budget_compliance_table.csv",
        mime="text/csv",
    )

    fig = build_budget_bars(df_eff, cmap, st.session_state.theme, variants=list(tbl["Variant"]))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download budget figure (PNG)",
        figure_to_png_bytes(fig),
        file_name="budget_compliance_bars.png",
        mime="image/png",
    )

    compliant = (tbl["Budget Compliant"] == "✓").sum()
    total = len(tbl)
    max_flops = float(tbl["Δ FLOPs %"].max())
    max_row = tbl.iloc[int(tbl["Δ FLOPs %"].idxmax())]
    if compliant == total:
        st.success(
            f"All **{total}** variants satisfy the < +10% computational budget constraint. "
            f"Maximum observed overhead: **{max_flops:.2f}% FLOPs** ({max_row['Variant']})."
        )
    else:
        st.error(
            f"Only **{compliant}/{total}** variants satisfy the < +10% computational budget constraint. "
            f"Maximum observed overhead: **{max_flops:.2f}% FLOPs** ({max_row['Variant']})."
        )

