# STAT 5243 Project 2 — Interactive Data Workbench

## 1. App Purpose and Deployment

This application is a full-featured, interactive data workbench built with **Shiny for Python**. It provides a complete, code-free workflow for loading, cleaning, transforming, and exploring tabular datasets directly in the browser. The entire application runs as a single local Python process with no REST API, external database, or cloud backend — all computation is performed through direct Python module imports, which keeps the architecture simple, auditable, and easy to deploy.

The workbench is designed for analysts, students, and data scientists who want to rapidly iterate on data preparation and exploratory analysis without switching between notebooks, scripts, and plotting libraries. Every action produces an immediate visual preview, and every transformation is tracked in a persistent version history so that users can retrace their steps.

[Deployment Link: _to be added upon deployment_]

**How to run locally:**

```bash
cd "/Users/m2/Projects/STAT 5243/Project 2"
pip install -r requirements.txt   # one-time setup
shiny run --reload app.py
```

The app supports five upload formats — **CSV**, **Excel (.xlsx / .xls)**, **JSON**, and **RDS** — and ships with three built-in demonstration datasets: **Sleep, Mobile and Stress** (15,000 rows), **Iris** (150 rows), and **Tips** (244 rows). These built-in datasets span a range of sizes and column types so that every feature of the workbench can be exercised without uploading a file.

## 2. Main Functionalities

### Data Loading

Users can upload their own datasets in any of the five supported formats (CSV, Excel `.xlsx`/`.xls`, JSON, RDS) or select one of the three built-in datasets to get started immediately. Upon loading, the app displays a concise summary card showing the number of rows, columns, missing values, and duplicate rows. Every loaded or derived dataset is registered in an **in-memory version history** that records the source, the transformation applied, and a timestamp, giving users a full audit trail of how their current data was produced.

### Data Cleaning and Preprocessing

The Cleaning tab exposes **seven** categories of operations, each configurable through dropdown menus and column selectors:

1. **Handle missing values** — drop rows or columns that contain nulls, or impute them with the mean, median, mode, or a user-specified constant.
2. **Remove duplicates** — deduplicate the dataset by all columns or a user-selected subset. A dedicated **duplicate inspection panel** lets users view the exact rows that would be removed before committing the operation.
3. **Scale numeric columns** — apply Standard scaling (z-score), Min-Max scaling (to [0, 1]), or Robust scaling (based on IQR, resistant to outliers).
4. **Encode categorical columns** — convert categorical variables via Label Encoding (integer codes) or One-Hot Encoding (binary indicator columns).
5. **Detect and handle outliers** — identify outliers using the IQR method and either remove the affected rows or cap (winsorize) extreme values to the fence boundaries.
6. **Standardize text** — normalize string columns by trimming whitespace, converting case, or applying other text-cleaning rules to reduce spurious category splits.
7. **Coerce column types** — explicitly cast columns to a target dtype (e.g., convert a string column of numbers to float, or a float column of codes to categorical).

Every operation follows a **preview-then-apply** workflow: users configure the operation, click Preview to inspect the result in a side-by-side data table, and only then click Apply to commit the change to the working dataset. A **before/after distribution chart** is rendered alongside each preview so the impact of the operation is immediately visible — for example, a histogram overlay showing the distribution before and after outlier capping.

### Feature Engineering

The Feature Engineering tab provides **eleven** column-level transforms:

| Transform | Description |
|---|---|
| **Log** | `log1p(x)` — compresses right-skewed distributions; safe for zeros |
| **Square** | `x^2` — captures quadratic relationships |
| **Cube** | `x^3` — captures cubic / asymmetric nonlinearities |
| **Interaction** | `x * y` — models the joint effect of two predictors |
| **Ratio** | `x / y` — with automatic zero-denominator protection |
| **Binning** | Equal-width discretization via `pd.cut` with configurable bin count |
| **One-Hot Encoding** | Dummy-variable expansion for categorical columns, with optional `drop_first` |
| **Standardize** | Z-score normalization (mean 0, std 1) |
| **Normalize** | Min-Max scaling to [0, 1] |
| **Fill NA** | Impute missing values (mean, median, mode, or constant) |
| **Drop NA** | Remove rows containing null values (globally or per-column) |

Each transform includes a **contextual explanation panel** that describes what the transform does, when it is appropriate to use it, and any assumptions or caveats (e.g., "log1p requires values >= -1"). After the user configures and previews a transform, the app displays a **formula line** (e.g., `output = log1p(column_name)`) together with a **before/after comparison chart** so the statistical effect of the transformation is clear before it is committed. All generated columns are appended to the working dataset with sensible default names (e.g., `column_squared`, `col1_div_col2`) that can be overridden by the user.

### Exploratory Data Analysis (EDA)

The EDA tab is organized into summary tables, a free-text query filter, and five visualization panels:

- **Summary tables** — head preview, `describe()` statistics, and a column-type inventory.
- **Free-text pandas query** — users can type arbitrary pandas query expressions (e.g., `age > 30 & salary < 100000`) to filter the working dataset before plotting.
- **1D plots** — histograms for numeric columns and categorical bar charts for non-numeric columns.
- **2D plots** — scatter, line, bar, box, heatmap, and 2D histogram, with configurable x and y axes and optional color grouping.
- **Regression analysis** — overlay a fitted curve on a scatter plot, with options for polynomial fit (up to order 5), robust regression, and LOWESS smoothing, plus a log-scale x-axis toggle. The panel reports the **Pearson r correlation coefficient** and its associated **p-value** for the selected pair of variables.
- **Multiline / grouped line plots** — plot multiple series on the same axes with automatic x-binning for dense data.
- **Correlation matrix heatmap** — compute and visualize a full pairwise correlation matrix using **Pearson**, **Spearman**, or **Kendall** methods, selectable from a dropdown.

All plots are rendered with Plotly for interactivity (zoom, pan, hover tooltips) and can be exported as images directly from the plot toolbar.

### Export

CSV download buttons are provided in **three locations** throughout the app — the **Load** tab, the **Cleaning** tab, and the **Feature Engineering** tab — so users can export the active dataset or any preview result at any stage of their workflow without navigating away from their current task.

## 3. How to Use the App

1. **Load a dataset.** Navigate to the Load tab. Choose one of the three built-in datasets (Sleep/Mobile/Stress, Iris, or Tips) from the dropdown, or click the file-upload widget to upload your own CSV, Excel, JSON, or RDS file. Click the Load button to ingest the data. The app will display a summary card with row count, column count, missing-value count, and duplicate-row count, along with a scrollable preview of the first rows.

2. **Inspect the data.** Review the summary card and the version history table at the bottom of the Load tab. The version history shows every dataset state in the current session — its source, the transformation that produced it, and a timestamp — so you can always trace how the current dataset was derived.

3. **Clean and preprocess.** Switch to the Cleaning tab. Select an operation category (e.g., Handle Missing Values), choose the target column(s), configure any parameters (e.g., fill strategy), and click Preview. Examine the before/after distribution chart and the preview data table. If the result looks correct, click Apply to commit the change; otherwise, adjust the parameters and preview again. Repeat for additional cleaning steps as needed.

4. **Engineer features.** In the Feature Engineering tab, select a transform from the dropdown (e.g., Log, Interaction, Binning). Read the contextual explanation panel to understand the transform's purpose and requirements. Choose the input column(s), optionally customize the output column name, and click Preview. Review the formula display and the before/after comparison chart, then click Apply to add the new column to the working dataset.

5. **Explore and visualize.** Use the EDA tab to examine summary statistics, apply free-text pandas query filters, and generate plots. Start with the 1D panel for univariate distributions, move to 2D plots for bivariate relationships, use the regression panel to quantify linear or nonlinear trends (with Pearson r and p-value), and generate a correlation matrix heatmap to identify the strongest pairwise associations across all numeric columns.

6. **Download results.** At any point, click the Download CSV button on the Load, Cleaning, or Feature Engineering tab to export your current working dataset (or the most recent preview) as a CSV file for use in downstream modeling, reporting, or sharing with collaborators.

## 4. Team Contributions

| Team Member    | Contribution                                                                                   |
|----------------|------------------------------------------------------------------------------------------------|
| Cecilia Zang   | Cleaning and preprocessing logic (`p2_divided.py`) — missing values, duplicates, scaling, encoding, outliers, text standardization, type coercion |
| Baixuan Chen   | Feature engineering logic (`feature_engineering.py`) — 11 transforms with metadata tracking     |
| Yuhan Guo      | EDA logic (`EDA.py`) — summaries, filtering, 1D/2D plots, regression, multiline, correlation   |
| Zeming Liang   | Shiny UI and integration (`app.py`) — app assembly, direct module wiring, deployment, report    |

## 5. Limitations and Next Steps

- **Session-local state.** The dataset version history lives in server memory and is lost when the app process restarts. There is currently no persistence layer (database or file-based session store) to survive restarts.
- **Single-user design.** The app is architected for one user at a time in a local browser session. Concurrent users would share the same reactive state, leading to conflicts. A production deployment would require per-session isolation or a multi-tenant framework.
- **Future work.** Deploy the app to a hosted platform (e.g., Posit Connect, Hugging Face Spaces, ShinyApps.io) for public access, add additional plot types (violin, KDE density), and support Parquet file format for large-dataset workflows.

## 6. Error Handling and Robustness

The application incorporates several layers of defensive programming to ensure a smooth user experience even when inputs are unexpected or malformed:

- **Input validation.** Every backend function validates its inputs before performing any computation. Column names are checked for existence in the DataFrame, required parameters are verified to be non-null, and method names are normalized and matched against an explicit allow-list. Invalid inputs produce clear, specific error messages rather than cryptic stack traces.
- **Graceful error messages.** All errors surfaced to the user are caught at the UI layer and displayed as human-readable notification banners. The app never crashes or shows a raw Python traceback in the browser; instead, it presents an actionable message (e.g., "Column 'age' must be numeric for this operation").
- **All-null column handling.** Operations that depend on statistical properties (mean, standard deviation, min/max) detect all-null or constant columns and raise descriptive errors rather than silently producing NaN or infinite results. For example, standardization checks that the standard deviation is nonzero before dividing.
- **Non-numeric column validation.** Transforms that require numeric input (log, square, cube, standardize, normalize, ratio, interaction) explicitly verify column dtypes before proceeding and report which columns failed the check, preventing confusing type-coercion errors downstream.
- **Busy indicators.** Long-running operations (e.g., loading a large file, computing a full correlation matrix) display visual busy indicators in the UI so the user knows the app is processing and has not frozen. This prevents impatient re-clicks that could queue duplicate operations.
