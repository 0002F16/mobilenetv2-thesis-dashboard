from __future__ import annotations

import argparse
import base64
import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd

from chapter4_dashboard.data.loader import load_disk_auto
from chapter4_dashboard.stats.tests import compute_table_b
from chapter4_dashboard.utils.colors import variant_color_map
from chapter4_dashboard.utils.export import figure_to_png_bytes

from chapter4_dashboard.figures.bars import build_accuracy_bars
from chapter4_dashboard.figures.budget import build_budget_bars
from chapter4_dashboard.figures.paired import build_paired_delta_grid
from chapter4_dashboard.figures.radar import build_efficiency_radar
from chapter4_dashboard.figures.scatter import build_accuracy_efficiency_scatter
from chapter4_dashboard.figures.curves import (
    build_training_diagnostic_plot,
    convergence_summary_table,
)
from chapter4_dashboard.figures.early_stop import build_early_stopping_distribution

from chapter4_dashboard.tabs.objective1_budget import _budget_table, _per_dataset_latency_table
from chapter4_dashboard.tabs.objective2_performance import _main_results_table


def _b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def _html_escape(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    view = df.head(max_rows).copy()
    # Keep it predictable across environments
    return view.to_html(index=False, escape=True, classes="tbl", border=0)


def _section(title: str, body_html: str, anchor: str | None = None) -> str:
    aid = f' id="{_html_escape(anchor)}"' if anchor else ""
    return f"""
    <section class="sec"{aid}>
      <h2>{_html_escape(title)}</h2>
      {body_html}
    </section>
    """


def _figure_block(fig_title: str, png_bytes: bytes) -> str:
    b64 = _b64_png(png_bytes)
    return f"""
    <figure class="fig">
      <figcaption>{_html_escape(fig_title)}</figcaption>
      <img class="img" alt="{_html_escape(fig_title)}" src="data:image/png;base64,{b64}" />
    </figure>
    """


def export_report_html(
    out_path: Path,
    *,
    theme: str = "Publication",
    include_training_diagnostics: bool = True,
    training_dataset: str = "CIFAR-100",
    training_metric: str = "Val Top-1",
    training_view_mode: str = "All seeds",
    training_seed: int | None = None,
    training_smoothing: int = 3,
) -> dict[str, Any]:
    df_runs, df_eff, df_curves, df_latency, meta = load_disk_auto()
    cmap = variant_color_map(theme)  # type: ignore[arg-type]

    stats_b = compute_table_b(df_runs) if (df_runs is not None and not df_runs.empty) else pd.DataFrame()

    budget_tbl = _budget_table(df_eff)
    per_ds_latency_tbl = _per_dataset_latency_table(df_latency)
    main_results_tbl = _main_results_table(df_runs, df_eff, df_latency)
    if not main_results_tbl.empty:
        main_results_tbl_disp = main_results_tbl.drop(columns=["_top1_mean", "_latency_mean_ms"], errors="ignore")
    else:
        main_results_tbl_disp = main_results_tbl

    figures: list[tuple[str, object]] = []
    # Objective 1
    figures.append(("Budget compliance bars", build_budget_bars(df_eff, cmap, theme)))
    # Objective 2
    figures.append(("Top-1 accuracy bars (absolute)", build_accuracy_bars(df_runs, "top1", cmap, theme, True, False)))
    figures.append(("Top-1 accuracy bars (Δ vs Baseline)", build_accuracy_bars(df_runs, "top1", cmap, theme, True, True)))
    figures.append(("Top-5 accuracy bars (absolute)", build_accuracy_bars(df_runs, "top5", cmap, theme, True, False)))
    figures.append(("Top-5 accuracy bars (Δ vs Baseline)", build_accuracy_bars(df_runs, "top5", cmap, theme, True, True)))
    # Objective 3
    figures.append(
        (
            "Per-seed paired differences grid (Δ Top-1 vs Baseline)",
            build_paired_delta_grid(
                df_runs,
                datasets=["CIFAR-100", "Tiny-ImageNet"],
                variants=["DualConv-only", "ECA-only", "Hybrid"],
                theme=theme,
            ),
        )
    )
    # Comparative
    figures.append(("Accuracy–Efficiency scatter (Top-1 vs FLOPs)", build_accuracy_efficiency_scatter(df_runs, df_eff, cmap, theme)))
    figures.append(("Efficiency radar (normalized ÷ Baseline)", build_efficiency_radar(df_runs, df_eff, cmap, theme, normalized=True)))
    figures.append(("Efficiency radar (absolute)", build_efficiency_radar(df_runs, df_eff, cmap, theme, normalized=False)))

    # Training diagnostics (optional)
    conv_tbl = pd.DataFrame()
    if include_training_diagnostics and (df_curves is not None) and (not df_curves.empty):
        view_mode = "Single seed" if training_view_mode.lower().startswith("single") else "All seeds"
        fig_diag = build_training_diagnostic_plot(
            df_curves=df_curves,
            dataset=training_dataset,
            metric=training_metric,
            view_mode=view_mode,
            seed=training_seed,
            smoothing=int(training_smoothing),
            cmap=cmap,
            theme=theme,
        )
        figures.append((f"Training diagnostics — {training_dataset} — {training_metric}", fig_diag))
        conv_tbl = convergence_summary_table(df_curves, training_dataset)
        fig_es = build_early_stopping_distribution(df_curves, training_dataset, cmap, theme)
        if getattr(fig_es, "data", None):
            figures.append((f"Early stopping distribution — {training_dataset}", fig_es))

    # Render HTML
    generated_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "MobileNetV2 Hybrid — Dashboard Report"

    toc = """
    <nav class="toc">
      <div class="toc-title">Contents</div>
      <ol>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#data">Loaded data</a></li>
        <li><a href="#obj1">Objective 1 — Budget compliance</a></li>
        <li><a href="#obj2">Objective 2 — Performance results</a></li>
        <li><a href="#obj3">Objective 3 — Ablation</a></li>
        <li><a href="#comparative">Comparative analysis</a></li>
        <li><a href="#training">Training diagnostics</a></li>
      </ol>
    </nav>
    """

    overview = f"""
    <div class="meta">
      <div><strong>Generated:</strong> {_html_escape(generated_at)}</div>
      <div><strong>Theme:</strong> {_html_escape(theme)}</div>
      <div><strong>Auto-selected artifacts:</strong> {_html_escape(meta.get("auto_selected_folder", "none"))}</div>
      <div><strong>Artifact version:</strong> {_html_escape(meta.get("auto_selected_version", meta.get("artifact_version", "unknown")))}</div>
      <div><strong>Runs loaded:</strong> {_html_escape(len(df_runs) if df_runs is not None else 0)}</div>
      <div><strong>Curves rows:</strong> {_html_escape(len(df_curves) if df_curves is not None else 0)}</div>
      <div><strong>Latency rows:</strong> {_html_escape(len(df_latency) if df_latency is not None else 0)}</div>
    </div>
    """

    data_sec = f"""
    <h3>df_runs (per-run accuracy)</h3>
    {_df_to_html_table(df_runs, max_rows=80)}
    <h3>df_efficiency</h3>
    {_df_to_html_table(df_eff, max_rows=80)}
    <h3>df_curves (training curves)</h3>
    {_df_to_html_table(df_curves, max_rows=80)}
    <h3>df_latency (aggregated from latency_results.json)</h3>
    {_df_to_html_table(df_latency, max_rows=80)}
    """

    obj1_sec = f"""
    <h3>Budget compliance table</h3>
    {_df_to_html_table(budget_tbl, max_rows=50)}
    <h3>Per-dataset inference latency (ms)</h3>
    <p>
      Mean latency per image (batch size 1) for each variant on CIFAR-10, CIFAR-100, and Tiny-ImageNet.
      Despite near-zero FLOPs overhead, ECA-only records the highest latency across all datasets.
    </p>
    {_df_to_html_table(per_ds_latency_tbl, max_rows=50)}
    """

    obj2_sec = f"""
    <h3>Main results table</h3>
    {_df_to_html_table(main_results_tbl_disp, max_rows=200)}
    <h3>Statistical validation (Table B)</h3>
    {_df_to_html_table(stats_b, max_rows=200)}
    """

    obj3_sec = """
    <p>
      This section includes the paired-differences grid figure used for ablation diagnostics.
      (The Streamlit dashboard also renders additional narrative tables; those can be added next if you want them in the report too.)
    </p>
    """

    comparative_sec = """
    <p>
      Accuracy–efficiency scatter and radar plots summarize the trade-offs across variants.
    </p>
    """

    training_sec = ""
    if include_training_diagnostics:
        training_sec += f"""
        <h3>Convergence summary — { _html_escape(training_dataset) }</h3>
        {_df_to_html_table(conv_tbl, max_rows=100)}
        """
    else:
        training_sec = "<p><em>Skipped.</em></p>"

    figs_html = "\n".join(_figure_block(name, figure_to_png_bytes(fig)) for name, fig in figures)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(title)}</title>
  <style>
    :root {{
      --bg: #0b0d10;
      --card: #12161c;
      --ink: #e9eef5;
      --muted: #a9b4c3;
      --line: rgba(255,255,255,0.08);
      --accent: #79b8ff;
    }}
    @media (prefers-color-scheme: light) {{
      :root {{
        --bg: #fafafa;
        --card: #ffffff;
        --ink: #111827;
        --muted: #4b5563;
        --line: rgba(17,24,39,0.10);
        --accent: #2563eb;
      }}
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      background: radial-gradient(1200px 700px at 10% 0%, rgba(121,184,255,0.10), transparent 55%),
                  radial-gradient(1000px 600px at 85% 10%, rgba(99,102,241,0.10), transparent 60%),
                  var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 18px 64px;
    }}
    header {{
      padding: 18px 18px 8px;
      background: linear-gradient(180deg, rgba(255,255,255,0.06), transparent);
      border: 1px solid var(--line);
      border-radius: 16px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      letter-spacing: -0.02em;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px 14px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
    }}
    .toc {{
      margin-top: 16px;
      padding: 14px 16px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .toc-title {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .toc a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .sec {{
      margin-top: 18px;
      padding: 16px 16px 8px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .sec h2 {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: -0.01em;
    }}
    h3 {{
      margin: 16px 0 8px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .fig {{
      margin: 14px 0 18px;
      padding: 12px 12px 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(0,0,0,0.10);
    }}
    .figcaption {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}
    .img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
    }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      overflow-x: auto;
      display: block;
    }}
    .tbl thead th {{
      position: sticky;
      top: 0;
      background: rgba(255,255,255,0.04);
      backdrop-filter: blur(6px);
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      white-space: nowrap;
    }}
    .tbl td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      white-space: nowrap;
    }}
    .tbl tr:hover td {{
      background: rgba(121,184,255,0.06);
    }}
    @media print {{
      body {{ background: #fff; color: #000; }}
      header, .toc, .sec {{ border: 1px solid #ddd; background: #fff; }}
      .img {{ break-inside: avoid; }}
      a {{ color: #000; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{_html_escape(title)}</h1>
      {overview}
    </header>
    {toc}
    {_section("Overview", "<p>This report bundles all processed tables and exported dashboard figures into one file.</p>" + overview, anchor="overview")}
    {_section("Loaded data", data_sec, anchor="data")}
    {_section("Objective 1 — Budget compliance", obj1_sec, anchor="obj1")}
    {_section("Objective 2 — Performance results", obj2_sec, anchor="obj2")}
    {_section("Objective 3 — Ablation", obj3_sec, anchor="obj3")}
    {_section("Comparative analysis", comparative_sec, anchor="comparative")}
    {_section("Training diagnostics", training_sec, anchor="training")}
    {_section("Figures (embedded PNG exports)", figs_html, anchor="figures")}
  </div>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return {
        "out_path": str(out_path),
        "meta": meta,
        "rows": {
            "df_runs": int(len(df_runs)) if df_runs is not None else 0,
            "df_efficiency": int(len(df_eff)) if df_eff is not None else 0,
            "df_curves": int(len(df_curves)) if df_curves is not None else 0,
            "df_latency": int(len(df_latency)) if df_latency is not None else 0,
            "stats_table_b": int(len(stats_b)) if stats_b is not None else 0,
        },
        "figures": [name for name, _ in figures],
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a single self-contained HTML report from dashboard artifacts.")
    p.add_argument("--out", type=str, default="report/dashboard_report.html", help="Output HTML path.")
    p.add_argument("--theme", type=str, default="Publication", choices=["Publication", "Dark", "Pastel"])
    p.add_argument("--no-training", action="store_true", help="Skip training diagnostics section/figures.")
    p.add_argument("--training-dataset", type=str, default="CIFAR-100", choices=["CIFAR-10", "CIFAR-100", "Tiny-ImageNet"])
    p.add_argument("--training-metric", type=str, default="Val Top-1", choices=["Val Top-1", "Train Loss", "Val Loss"])
    p.add_argument("--training-view", type=str, default="All seeds", choices=["All seeds", "Single seed"])
    p.add_argument("--training-seed", type=int, default=None, help="Seed (only used when --training-view='Single seed').")
    p.add_argument("--training-smoothing", type=int, default=3)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    export_report_html(
        Path(args.out),
        theme=args.theme,
        include_training_diagnostics=not bool(args.no_training),
        training_dataset=args.training_dataset,
        training_metric=args.training_metric,
        training_view_mode=args.training_view,
        training_seed=args.training_seed,
        training_smoothing=int(args.training_smoothing),
    )


if __name__ == "__main__":
    main()

