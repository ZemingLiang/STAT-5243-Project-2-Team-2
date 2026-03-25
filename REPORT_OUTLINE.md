# STAT 5243 Project 2 — Interactive Data Workbench

## 1. Overview and Deployment

Our application is an interactive data workbench built with **Shiny for Python**. It allows users to load, clean, transform, and explore tabular datasets entirely in the browser without writing any code. The app is organized into five tabs — **Guide**, **Load**, **Cleaning**, **Feature Engineering**, and **EDA** — accessible from a persistent top navigation bar. All computation runs locally via pure Python module imports; there is no REST API or external backend.

**Deployed application:** [Deployment Link: _to be added upon deployment_]

**To run locally:** `pip install -r requirements.txt` then `shiny run app.py` and open http://127.0.0.1:8000.

The app accepts uploads in **CSV, Excel (.xlsx/.xls), JSON, and RDS** formats and ships with three built-in demonstration datasets: **Sleep/Mobile/Stress** (15,000 rows, 13 columns), **Iris** (150 rows, 5 columns), and **Tips** (244 rows, 7 columns).

## 2. App Functionalities

**Data Loading.** In the **Load** tab, users choose a built-in dataset from the dropdown or click the file-upload widget to select a local file. After clicking "Load," the app displays a summary card showing row count, column count, missing-value count, and duplicate-row count. Every dataset loaded or derived is tracked in an in-memory **version history** table visible at the bottom of the tab, so users can switch between any previous version using the dataset picker dropdown.

**Data Cleaning and Preprocessing.** The **Cleaning** tab presents a sidebar with seven operations: handle missing values (six strategies including mean, median, mode, and constant fill), remove duplicates (with a dedicated panel showing which rows are duplicated), scale numeric columns (standard, min-max, or robust), encode categorical columns (label or one-hot), detect and handle outliers (IQR-based removal or capping with a configurable multiplier), standardize text (whitespace stripping and case normalization), and coerce column types (convert strings to numbers or vice versa). Users first click **Preview** to inspect a before/after comparison chart and preview table, then click **Apply** only when satisfied. The column selector automatically filters to show only relevant column types for each operation (e.g., numeric columns for scaling, categorical columns for encoding).

**Feature Engineering.** The **Feature Engineering** tab offers eleven column-level transforms: log, square, cube, interaction, ratio, binning, one-hot encoding, standardize, normalize, fill NA, and drop NA. When a method is selected, a contextual explanation appears below the dropdown describing what the transform does and when to use it. After clicking **Preview**, the app displays the applied formula (e.g., `log_age = log1p(age)`), before/after summary statistics (mean and standard deviation), and a side-by-side distribution comparison chart. Users can customize the output column name and adjust method-specific parameters (bin count, fill strategy, drop-first flag, etc.).

**Exploratory Data Analysis.** The **EDA** tab provides summary tables (head preview with adjustable row count, descriptive statistics, and column-type inventory), a free-text pandas query filter, and five interactive visualization panels: **1D plots** (histograms or categorical bar charts with log-scale and normalization toggles), **2D plots** (scatter, line, bar, box, heatmap, or 2D histogram with optional color grouping and log-scale axes), **regression analysis** (polynomial fit up to order 5, robust regression, or LOWESS smoothing, annotated with Pearson r, R-squared, and p-value), **multiline grouped plots**, and a **correlation matrix heatmap** (Pearson, Spearman, or Kendall). All plots are rendered with Plotly and support interactive zoom, pan, and hover tooltips. A persistent statistics panel below the 1D plot displays mean, median, standard deviation, skewness, and kurtosis.

**Export.** CSV download buttons appear on the Load, Cleaning, and Feature Engineering tabs so users can export the active dataset or any preview result at any stage.

## 3. How to Use the App

1. **Load a dataset.** Open the app and navigate to the **Load** tab (second tab in the top navbar). Either select a built-in dataset from the "Choose built-in dataset" dropdown and click **Load Built-in Dataset**, or click the file-upload area labeled "Upload CSV, Excel, JSON, or RDS," browse for your file, then click **Load Uploaded File**. The summary card on the right will populate with dataset statistics.

2. **Inspect and understand your data.** Scroll down in the Load tab to review the **Dataset History** table, which tracks every version of your data. Switch to the **EDA** tab and review the **Data Preview** card (first N rows), the **Describe** card (summary statistics), and the **Column Types** card to understand what you are working with.

3. **Clean and preprocess.** Navigate to the **Cleaning** tab. In the left sidebar, select an operation from the "Action" dropdown (e.g., "Handle missing values"). The column selector will automatically show only the relevant columns. Configure parameters (e.g., choose "Fill with mean" as the strategy), then click **Preview**. Examine the before/after chart on the right. If the result looks correct, click **Apply**; otherwise, adjust and preview again.

4. **Engineer features.** Navigate to the **Feature Engineering** tab. Select a transform (e.g., "Log transform") from the "Method" dropdown and read the explanation that appears. Choose a primary column, optionally set a custom output column name, then click **Preview**. Review the formula, before/after statistics, and comparison chart. Click **Apply** to add the new column to your working dataset.

5. **Explore with EDA.** In the **EDA** tab, use the filter text box to subset your data (e.g., `age > 30 and gender == "Female"`). Then render plots: select a column for the 1D panel and click **Render 1D Plot**; choose X, Y, and optional hue columns for the 2D panel and click **Render 2D Plot**. For trend analysis, use the Regression panel with polynomial order and fit-method toggles. Generate a correlation heatmap from the bottom panel.

6. **Download results.** At any point, click the **Download CSV** button on the Load, Cleaning, or Feature Engineering tab to save your current dataset or preview to a local CSV file.

## 4. Team Contributions

| Team Member    | Contribution |
|----------------|-------------|
| Cecilia Zang   | Data cleaning and preprocessing backend (`p2_divided.py`): missing-value handling, duplicate detection, scaling, encoding, outlier handling, text standardization, type coercion |
| Baixuan Chen   | Feature engineering backend (`feature_engineering.py`): 11 column-level transforms with metadata, formula tracking, and input validation |
| Yuhan Guo      | EDA backend (`EDA.py`): summary functions, filtering, 1D/2D plot generation, regression analysis, multiline plots, correlation matrix |
| Zeming Liang   | Shiny UI and integration (`app.py`): application assembly, reactive wiring, theme design, deployment, testing, and project report |

## 5. Limitations and Future Work

- **Session-local state.** Dataset version history is stored in memory and lost when the app restarts.
- **Single-user design.** Concurrent users would share reactive state; a production deployment would require per-session isolation.
- **Future enhancements.** Deploy to a hosted platform for public access, add violin and KDE density plots, and support the Parquet file format for large-dataset workflows.
