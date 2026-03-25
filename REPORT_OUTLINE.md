# STAT 5243 Project 2 — Interactive Data Workbench

## 1. App Purpose and Deployment

This application is an interactive data workbench built with **Shiny for Python**. It provides a code-free workflow for loading, cleaning, transforming, and exploring tabular datasets in the browser. All computation runs locally through direct Python module imports — no REST API or external backend.

[Deployment Link: _to be added upon deployment_]

**Run locally:** `pip install -r requirements.txt && shiny run --reload app.py`

**Supported formats:** CSV, Excel (.xlsx/.xls), JSON, and RDS. Three built-in datasets are included: Sleep/Mobile/Stress (15,000 rows), Iris (150 rows), and Tips (244 rows).

## 2. Main Functionalities

**Data Loading.** Upload files in 4 formats or select a built-in dataset. The app displays row/column counts, missing values, and duplicates, and tracks a full version history of every derived dataset.

**Data Cleaning (7 operations).** Handle missing values (6 strategies), remove duplicates (with inspection of duplicate rows), scale numeric columns (standard/min-max/robust), encode categoricals (label/one-hot), detect and handle outliers (IQR-based remove or cap), standardize text (whitespace/case), and coerce column types (string-to-numeric or vice versa). Every operation uses a preview-then-apply workflow with before/after distribution charts.

**Feature Engineering (11 transforms).** Log, square, cube, interaction, ratio, binning, one-hot encoding, standardize, normalize, fill NA, and drop NA. Each transform shows a contextual explanation, the formula applied (e.g., `log_age = log1p(age)`), and a before/after comparison chart.

**Exploratory Data Analysis.** Summary tables (head, describe, column types), free-text pandas query filtering, 1D plots (histograms/bar charts), 2D plots (scatter/line/bar/box/heatmap/2D-histogram with hue), polynomial/robust/LOWESS regression with Pearson r and p-value, multiline grouped plots, and a Pearson/Spearman/Kendall correlation matrix heatmap. All plots are interactive (Plotly).

**Export.** CSV download buttons on the Load, Cleaning, and Feature Engineering tabs.

## 3. How to Use

1. **Load** — Choose a built-in dataset or upload a CSV/Excel/JSON/RDS file in the Load tab.
2. **Inspect** — Review the summary card and version history.
3. **Clean** — Select an operation, configure parameters, Preview, then Apply.
4. **Engineer** — Pick a transform, read the explanation, Preview the formula and chart, then Apply.
5. **Explore** — Use the EDA tab for summaries, filtering, plots, regression, and correlation.
6. **Download** — Click any Download CSV button to export your data.

## 4. Team Contributions

| Team Member    | Contribution |
|----------------|-------------|
| Cecilia Zang   | Cleaning and preprocessing logic (`p2_divided.py`) |
| Baixuan Chen   | Feature engineering logic (`feature_engineering.py`) |
| Yuhan Guo      | EDA logic (`EDA.py`) |
| Zeming Liang   | Shiny UI, integration, deployment, and report (`app.py`) |

## 5. Limitations

- Dataset history is session-local (lost on restart).
- Single-user design; concurrent users would share state.
- Future work: hosted deployment, violin/KDE plots, Parquet support.
