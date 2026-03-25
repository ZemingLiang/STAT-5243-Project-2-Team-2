# STAT 5243 Project 2 — Interactive Data Workbench

**Deployed App:** <https://019d23ea-1266-cada-1d21-45e5d97e6ea5.share.connect.posit.cloud/>

**Group Members:** Zeming Liang (`zl3688`), Yuhan Guo (`yg2695`), Baixuan Chen (`bc3212`), Cecilia Zang (`cz2957`)

---

## Overview

An interactive, code-free data workbench built with **Shiny for Python**. Users can load, clean, transform, and explore tabular datasets entirely in the browser. The app is organized into five tabs — **Guide**, **Load**, **Cleaning**, **Feature Engineering**, and **EDA** — with a polished Lux Bootstrap theme, 20+ tooltips, sidebar layouts, and a full dataset version history.

All computation runs locally through pure Python module imports. There is no Flask, no REST API, and no external backend — user data never leaves the machine.

---

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

All 7 tests should pass (imports, built-in loaders, cleaning, feature engineering, EDA summaries, EDA plots, correlation matrix).

---

## Repository Structure

| File | Purpose |
|------|---------|
| `app.py` | Shiny app entrypoint — UI layout (page_navbar, cards, sidebars, tooltips) and server logic (51 reactive elements, 19 output renderers) |
| `eda.py` | EDA backend — summary tables, pandas query filtering, 6 plot families (1D, 2D, regression, multiline, correlation), Pearson/Spearman/Kendall correlation matrix |
| `data_cleaning.py` | Cleaning backend — 9 operations: missing values (6 strategies), duplicates, scaling (3 methods), encoding (2 methods), outliers (IQR), text standardization, type coercion |
| `feature_engineering.py` | Feature engineering backend — 11 transforms (log, square, cube, interaction, ratio, binning, one-hot, standardize, normalize, fillna, dropna) with formula tracking and metadata |
| `tests.py` | Integration smoke tests — 7 test cases covering all modules |
| `test_data/` | Built-in dataset: Sleep, Mobile and Stress (15,000 rows, 13 columns) |
| `requirements.txt` | All Python dependencies (13 packages) |
| `REPORT.md` | Final project report (markdown source) |
| `report.pdf` | Final project report (2-page PDF for Courseworks submission) |
| `.gitignore` | Excludes `__pycache__/`, `.DS_Store`, and assignment PDFs |

---

## Features

### 1. Data Loading
- Upload **CSV, Excel (.xlsx/.xls), JSON, and RDS** files with robust error handling on every format
- **3 built-in datasets:** Sleep/Mobile/Stress (15,000 rows), Iris (150 rows), Tips (244 rows)
- Summary card displays row count, column count, missing values, and duplicates after loading
- Full **dataset version history** — every load, clean, or transform creates a new version; switch back to any previous version via the dataset picker

### 2. Data Cleaning and Preprocessing
- **9 operations:** handle missing values (drop rows/columns, fill with mean/median/mode/constant), remove duplicates (with inspection of duplicate rows), scale numeric columns (standard/min-max/robust), encode categorical columns (label/one-hot), detect and handle outliers (IQR-based removal or capping with adjustable multiplier), standardize text (whitespace/case), coerce column types (string-to-numeric or vice versa)
- **Smart column filtering** — dropdown auto-shows only numeric columns for scaling, only categorical for encoding
- **Preview-then-apply workflow** with before/after comparison charts (distribution histograms for value changes, row-count bars for row removals)
- Rich **outlier diagnostics** showing Q1, Q3, IQR, fence boundaries, and outlier count
- **Type safety warning** when applying text operations to numeric columns

### 3. Feature Engineering
- **11 transforms:** log (log1p), square, cube, interaction (col1 * col2), ratio (col1 / col2 with zero-denominator protection), binning, one-hot encoding (with cardinality guard at >50 unique values), standardize (z-score), normalize (min-max), fill NA, drop NA
- **Contextual explanation panel** for each transform (e.g., "Applies log(1+x). Reduces right-skew and compresses large values.")
- **Formula display** (e.g., `log_age = log1p(age)`) with before/after summary statistics (mean, std)
- **Before/after comparison chart** comparing input column vs. output column distributions
- Custom output column naming and method-specific parameter panels

### 4. Exploratory Data Analysis (EDA)
- **Summary tables:** data preview (adjustable row count), descriptive statistics, column types
- **Free-text pandas query filtering** with syntax hints and error guidance (e.g., `age > 30 and gender == "Female"`)
- **1D plots:** histograms and categorical bar charts with log-scale X/Y toggles, normalization, and a persistent statistics panel (mean, median, std, skewness, kurtosis)
- **2D plots:** scatter, line, bar, box, heatmap, and 2D histogram with optional color grouping (hue). Column selectors show type labels like `age (num)` and `gender (cat)` to guide valid selections
- **Regression analysis:** polynomial fit (order 1–5), robust regression, and LOWESS smoothing with **Pearson r, R-squared, and p-value** displayed directly on the plot
- **Multiline grouped plots** with shared bin edges for direct comparison across categories
- **Correlation matrix heatmap** with Pearson, Spearman, or Kendall methods, annotated values, and interactive hover
- All plots rendered with **Plotly** (zoom, pan, hover tooltips) and expandable to full screen

### 5. UI/UX
- **Lux Bootstrap theme** (shinyswatch) with custom CSS — gradient metric cards, hover shadow transitions, tip boxes
- **Custom Plotly template** with a 10-color coordinated palette and consistent typography
- **Guide tab** with welcome message, 6-step walkthrough, and 3 contextual tip boxes
- **20+ tooltips** on buttons, selectors, and controls throughout the app
- **Help notes** on numeric inputs (IQR formula, bin guidance, polynomial order explanation)
- **Sidebar layouts** in Cleaning and Feature Engineering tabs (collapsible on mobile)
- **11 full-screen expandable cards** for plots and tables
- **Busy indicators** during all reactive computations
- **Comprehensive error handling** — 13 try/except blocks, 57+ null checks, input validation on every operation

### 6. Export
- **CSV download buttons** on the Load, Cleaning, and Feature Engineering tabs
- Download the active dataset or any preview result at any stage

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Shiny for Python 1.0+ |
| Theme | shinyswatch (Lux Bootstrap) |
| Plotting | Plotly (with ScatterGL for large datasets) |
| Data | pandas, numpy |
| ML/Stats | scikit-learn (scalers, encoders), statsmodels (robust/LOWESS), scipy (pearsonr) |
| File I/O | openpyxl (Excel), pyreadr (RDS), seaborn (Tips dataset) |
| Testing | unittest (7 integration tests) |

---

## Branch History (Development Log)

The finalized submission lives on **`Main-Final-Deliverables`** (the default branch). Feature branches are preserved as a record of the team's development workflow — they are **not** intended to be merged further.

| Branch | Owner | Purpose |
|--------|-------|---------|
| `Main-Final-Deliverables` | Zeming Liang | Final integrated app — all modules merged, polished, and submission-ready |
| `Feature-Engineering` | Baixuan Chen | Development of the 11 feature transforms (`feature_engineering.py`) |
| `Data-Loading-Cleaning-Preprocessing` | Cecilia Zang | Development of the cleaning and preprocessing module (`data_cleaning.py`) |
| `Exploratory-Data-Analysis` | Yuhan Guo | Development of the EDA backend (`eda.py`) — filtering, plotting, regression, correlation |
| `UI-&-Web-App` | (initial scaffold) | Early project scaffold; superseded by the integrated `app.py` on the main branch |

---

## Team Contributions

| Team Member | Contribution |
|-------------|-------------|
| **Cecilia Zang** | Data cleaning and preprocessing backend (`data_cleaning.py`): 9 operations with validation, error handling, and pipeline support |
| **Baixuan Chen** | Feature engineering backend (`feature_engineering.py`): 11 transforms with formula tracking, metadata, and input validation |
| **Yuhan Guo** | EDA backend (`eda.py`): summary functions, filtering, 6 plot families, regression analysis, and correlation matrix |
| **Zeming Liang** | Shiny UI and integration (`app.py`): application assembly, reactive wiring, Lux theme, 20+ tooltips, deployment, testing, and report |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port already in use | Use `shiny run app.py --port 8765` |
| seaborn cache error | The app uses `sklearn.datasets.load_iris()` for Iris to avoid cache issues |
| RDS upload fails | Ensure `pyreadr` is installed: `pip install pyreadr` |
| Filter syntax error | Use `==` for equality, `&` for AND, `|` for OR, backticks for column names with spaces |
