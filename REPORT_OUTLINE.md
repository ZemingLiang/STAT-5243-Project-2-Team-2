# STAT 5243 Project 2 — Interactive Data Workbench

## 1. App Purpose and Deployment

This application is an interactive data workbench built with **Shiny for Python**. It allows
users to load, clean, transform, and explore datasets entirely in the browser without writing
code. The app runs as a single local process with no REST API or external backend — all
logic executes through direct Python module imports.

**How to run locally:**

```bash
cd "/Users/m2/Projects/STAT 5243/Project 2"
shiny run --reload app.py
```

Supported upload formats: CSV, Excel (.xlsx/.xls), and JSON. Two built-in datasets are
included for demonstration: *Sleep, Mobile and Stress* (15,000 rows) and *Iris* (150 rows).

## 2. Main Functionalities

**Data Loading.** Users can upload their own datasets in CSV, Excel, or JSON format, or
select a built-in dataset to get started immediately. Every loaded or derived dataset is
tracked in an in-memory version history with metadata (source, transform, timestamp).

**Data Cleaning and Preprocessing.** The Cleaning tab provides five categories of
operations: handle missing values (drop rows/columns, fill with mean/median/mode/constant),
remove duplicates, scale numeric columns (standard, min-max, robust), encode categorical
columns (label or one-hot), and detect/handle outliers (remove or cap via IQR). Users
always preview results before applying, and a before/after distribution chart shows the
impact of each operation visually.

**Feature Engineering.** Eleven transforms are available: log, square, cube, interaction,
ratio, binning, one-hot encoding, standardize, normalize, fill NA, and drop NA. Each
transform is backed by pure DataFrame-in/DataFrame-out functions. The preview workflow
shows both a data table and a side-by-side comparison chart so users can see the effect of
the transformation before committing it.

**Exploratory Data Analysis (EDA).** The EDA tab provides summary tables (head, describe,
column types), free-text pandas query filtering, and five visualization panels: 1D plots
(histograms and categorical bar charts), 2D plots (scatter, line, bar, box, heatmap, 2D
histogram), regression analysis (polynomial fit up to order 5, robust regression, LOWESS
smoothing, log-scale x), multiline grouped plots, and a full correlation matrix heatmap
(Pearson, Spearman, or Kendall).

**Export.** CSV download buttons are provided on the Load, Cleaning, and Feature Engineering
tabs so users can export the active dataset or any preview result at any time.

## 3. How to Use the App

1. **Load** — Go to the Load tab, choose a built-in dataset or upload a file, then click Load.
2. **Inspect** — Review the dataset summary (rows, columns, missing values, duplicates) and the version history table.
3. **Clean** — Switch to the Cleaning tab, select an operation and columns, click Preview to inspect the result, then click Apply to save.
4. **Engineer** — In the Feature Engineering tab, pick a transform and column(s), preview, then apply.
5. **Explore** — Use the EDA tab to view summaries, filter the data, and create interactive plots. Generate a correlation matrix for a high-level overview.
6. **Download** — Click any Download CSV button to export your current or previewed data.

## 4. Team Contributions

| Team Member    | Contribution                                                                                   |
|----------------|------------------------------------------------------------------------------------------------|
| Cecilia Zang   | Cleaning and preprocessing logic (`p2_divided.py`) — missing values, duplicates, scaling, encoding, outliers |
| Baixuan Chen   | Feature engineering logic (`feature_engineering.py`) — 11 transforms with metadata tracking     |
| Yuhan Guo      | EDA logic (`EDA.py`) — summaries, filtering, 1D/2D plots, regression, multiline, correlation   |
| Zeming Liang   | Shiny UI and integration (`app.py`) — app assembly, direct module wiring, deployment, report    |

## 5. Limitations and Next Steps

- **Session-local state.** Dataset history lives in memory and is lost when the app restarts.
- **No RDS support.** RDS loading is intentionally out of scope for this version.
- **Single-user.** The app is designed for one user at a time in a local browser session.
- **Future work.** Deploy to a hosted platform (e.g., Posit Connect, Hugging Face Spaces),
  add more plot types (violin, KDE density), and support Parquet/RDS file formats.
