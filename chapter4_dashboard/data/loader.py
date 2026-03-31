from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.loaders import load_disk_data
from dashboard.constants import TRAINED_MODEL_ROOTS


def load_disk_v2(repo_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Load from repo-root `Trained Models v2/` using the existing (tested) loader.

    Note: `dashboard.loaders.load_disk_data("v2")` already implements:
      Trained Models v2/<dataset>/<variant>/seed_<seed>/metrics.json
      Trained Models v2/<dataset>/<variant>/seed_<seed>/logs/epochs.jsonl
    """
    # The existing loader resolves repo root internally; `repo_root` kept for API symmetry.
    try:
        df_runs, df_eff, df_curves, meta = load_disk_data("v2")
    except Exception as e:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"artifact_version": "v2", "error": str(e)},
        )
    return df_runs, df_eff, df_curves, meta


def load_disk_auto(repo_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Try to load trained-model artifacts from disk, preferring newer layouts.

    Checks (in order) for these repo-root folders:
      - Trained Models v3/
      - Trained Models v2/
      - Trained Models/        (treated as v2 layout)
      - Trained Models v1/
    """
    root = repo_root or Path(__file__).resolve().parents[2]

    # Map discovered folder to loader "version".
    folder_to_version = {
        "Trained Models v3": "v3",
        "Trained Models v2": "v2",
        "Trained Models": "v2_generic",
        "Trained Models v1": "v1",
    }

    errors: list[dict[str, Any]] = []
    for folder in TRAINED_MODEL_ROOTS:
        if not (root / folder).is_dir():
            continue
        version = folder_to_version.get(folder)
        if version is None:
            continue
        try:
            df_runs, df_eff, df_curves, meta = load_disk_data(version)
        except Exception as e:
            errors.append({"folder": folder, "version": version, "error": str(e)})
            continue
        if len(df_runs) and len(df_eff):
            meta = {**meta, "auto_selected_folder": folder, "auto_selected_version": version}
            return df_runs, df_eff, df_curves, meta

    return (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {"artifact_version": "auto", "tried_folders": TRAINED_MODEL_ROOTS, "errors": errors},
    )

