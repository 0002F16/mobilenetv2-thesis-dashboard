# Chapter 4 Dashboard (Standalone)

Streamlit dashboard for Chapter 4 (Results and Discussion).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

The app will try the following (in order):

- **On-disk trained-model artifacts** (optional): place one or more of these folders at the repo root:
  - `Trained Models v3/`
  - `Trained Models v2/`
  - `Trained Models v1/`
- **CSV upload**: use the sidebar to upload `df_runs.csv`, `df_efficiency.csv`, and `df_curves.csv`.
- **No fallback**: if no artifacts are found and no CSVs are uploaded, the dashboard shows an empty state.

## Repo layout

- `app.py`: Streamlit entrypoint
- `chapter4_dashboard/`: Chapter 4 dashboard package (tabs, figures, stats, data loaders)
- `dashboard/`: minimal shared utilities required by the dashboard loaders/colors

# mobilenetv2-thesis-dashboard
