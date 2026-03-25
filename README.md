# STAT-5243-Project-2-Team-2

Group members: Zeming Liang (`zl3688`), Yuhan Guo (`yg2695`), Baixuan Chen (`bc3212`), Cecilia Zang (`cz2957`)

## App Architecture

This project is a single local **Shiny for Python** app. The UI imports and calls local Python modules directly — no Flask, no REST API.

| File | Purpose |
|------|---------|
| `app.py` | Shiny app entrypoint, UI layout, and server logic |
| `eda.py` | Summary, filtering, plotting, regression, and correlation functions |
| `data_cleaning.py` | Data loading, cleaning, scaling, encoding, and outlier handling |
| `feature_engineering.py` | 11 feature transforms (log, square, cube, interaction, ratio, etc.) |
| `tests.py` | Integration smoke tests |
| `test_data/` | Built-in dataset (Sleep/Mobile/Stress, 15,000 rows) |
| `requirements.txt` | Python dependencies |
| `REPORT.md` | Final project report (also available as `report.pdf`) |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires **Python >= 3.10**.

### 2. Run the app

```bash
shiny run app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 3. Run smoke tests

```bash
python3 tests.py
```

## Features

- **Data Loading**: Upload CSV, Excel, JSON, and RDS files, or use 3 built-in datasets (Sleep/Mobile/Stress, Iris, Tips)
- **Data Cleaning**: Handle missing values (6 strategies), remove duplicates, scale (3 methods), encode (2 methods), handle outliers (IQR-based remove or cap) — all with preview-first workflow and before/after comparison charts
- **Feature Engineering**: 11 transforms with per-transform explanations, before/after visual feedback, and metadata tracking
- **EDA**: Summary tables, pandas query filtering, 1D/2D plots (7 kinds), polynomial/robust/LOWESS regression, multiline grouped plots, and Pearson/Spearman/Kendall correlation matrix
- **Export**: CSV download buttons on Load, Cleaning, and Feature Engineering tabs
- **UI**: Lux Bootstrap theme, sidebar layouts, tooltips, busy indicators, and dataset version history

## Troubleshooting

- **`ModuleNotFoundError`**: Run `pip install -r requirements.txt` to install all dependencies.
- **Port already in use**: Use `shiny run app.py --port 8765` to pick a different port.
- **seaborn cache error**: The app uses `sklearn.datasets.load_iris()` instead of seaborn for Iris to avoid cache directory issues in restricted environments.
- **RDS files**: Requires `pyreadr`. Install via `pip install pyreadr`.
