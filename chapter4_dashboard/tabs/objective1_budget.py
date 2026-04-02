from __future__ import annotations

import pandas as pd
import streamlit as st

from chapter4_dashboard.figures.budget import build_budget_bars
from chapter4_dashboard.utils.constants import DATASET_ORDER, VARIANT_ORDER
from chapter4_dashboard.utils.export import figure_to_png_bytes, dataframe_to_csv_bytes
from chapter4_dashboard.utils.styling import style_budget_table, style_per_dataset_latency_table


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


def _per_dataset_latency_table(df_latency: pd.DataFrame) -> pd.DataFrame:
    """Wide table: Variant × {CIFAR-10, CIFAR-100, Tiny-ImageNet} mean latency (ms)."""
    need = {"dataset", "variant", "latency_mean_ms"}
    if df_latency is None or df_latency.empty or not need.issubset(set(df_latency.columns)):
        return pd.DataFrame()
    sub = df_latency[df_latency["dataset"].isin(DATASET_ORDER)].copy()
    if sub.empty:
        return pd.DataFrame()
    try:
        pivot = sub.pivot(index="variant", columns="dataset", values="latency_mean_ms")
    except ValueError:
        pivot = sub.pivot_table(index="variant", columns="dataset", values="latency_mean_ms", aggfunc="mean")
    row_order = [v for v in VARIANT_ORDER if v in pivot.index]
    if not row_order:
        return pd.DataFrame()
    pivot = pivot.reindex(row_order)
    col_order = [c for c in DATASET_ORDER if c in pivot.columns]
    if not col_order:
        return pd.DataFrame()
    out = pivot[col_order].reset_index().rename(columns={"variant": "Variant"})
    return out


def render_tab_budget(df_eff: pd.DataFrame, df_latency: pd.DataFrame, cmap: dict[str, str]) -> None:
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

    lat_wide = _per_dataset_latency_table(df_latency)
    st.markdown("**Per-dataset inference latency (ms)**")
    st.caption(
        "Mean latency per image (batch size 1) for each variant on CIFAR-10, CIFAR-100, and Tiny-ImageNet. "
        "Despite near-zero FLOPs overhead, ECA-only records the highest latency across all datasets."
    )
    if lat_wide.empty:
        st.info(
            "Upload or provide `latency_results.json` (see Raw Data / sidebar) to populate per-dataset latency."
        )
    else:
        ds_cols = [c for c in DATASET_ORDER if c in lat_wide.columns]
        fmt = {c: "{:.2f}" for c in ds_cols}
        lat_sty = style_per_dataset_latency_table(lat_wide).format(fmt)
        st.dataframe(lat_sty, use_container_width=True, height=240)
        st.download_button(
            "Download per-dataset latency table (CSV)",
            dataframe_to_csv_bytes(lat_wide),
            file_name="per_dataset_latency_ms.csv",
            mime="text/csv",
            key="dl_per_ds_latency",
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

