# EDA Module — Backend Reference

This document covers the EDA (Exploratory Data Analysis) backend block for the interactive web application. It is intended for developers integrating, extending, or testing the code.

---

## Purpose

The EDA module provides backend-side logic for lightweight exploratory data analysis on user-uploaded tabular datasets.

Design goals:

- Keep **dataset storage / retrieval** (`dataset_store.py`) separate from **EDA computation logic** (`EDA.py`).
- Return **JSON-serializable Python dictionaries** from every public function — no matplotlib figures, no terminal output.
- Let the **frontend own all rendering** (tables, plots, messages).
- Support both **web integration** (via FastAPI / Flask) and **offline local testing** (`offline_test.py`).

---

## File overview

| File | Role |
|---|---|
| `dataset_store.py` | Dataset lifecycle: upload, validate, persist to disk, cache in RAM, retrieve by `dataset_id` |
| `EDA.py` | EDA logic operating on `pandas.DataFrame` |
| `offline_test.py` | Local test runner: exercises every EDA function, saves JSON + PNG outputs |

---

## Quick start

### Dependencies

```
pandas
numpy
scipy        # for KDE contour plots
statsmodels  # for LOWESS / robust regression
matplotlib   # for offline PNG rendering only
```

Install missing packages:

```bash
pip install statsmodels scipy matplotlib
```

### Run the offline test

```bash
python offline_test.py
```

**Input:** `test_data/sleep_mobile_stress_dataset_15000.csv`

**Output:** `offline_test_output/`
- JSON files for every EDA function call
- PNG renders of every plot type
- Text snapshots of table outputs

---

## Architecture

### Separation of concerns

`dataset_store.py` handles:
- CSV upload and validation
- Assigning a UUID-based `dataset_id`
- Optionally saving the raw CSV to `./uploaded_datasets/`
- RAM cache for fast repeated access
- Metadata storage (shape, columns, dtypes, timestamp, file size)

`EDA.py` handles:
- DataFrame viewing (`show_head`, `describe_dataframe`, `column_types`)
- Filtering (`filter_dataframe`)
- 1D and 2D plots
- Regression analysis

**EDA functions take a `pd.DataFrame` directly and have no knowledge of `dataset_id` or the API layer.**

### JSON-first return format

Every function returns a JSON-serializable dict. Three possible shapes:

```json
{ "status": "success", "data": { ... } }
{ "status": "warning", "message": "...", "data": { ... } }
{ "status": "error",   "message": "...", "details": { ... } }
```

NaN and infinity are converted to `null` for JSON compatibility.

---

## API reference — EDA.py

### Dataframe viewing

#### `show_head(df, n=5)`
Returns the first `n` rows. Equivalent to `df.head(n)`.

Response `data` fields: `view_type`, `n_requested`, `n_returned`, `shape`, `columns`, `rows`

#### `describe_dataframe(df)`
Returns summary statistics for all columns. Equivalent to `df.describe(include='all')`.

Response `data` fields: `view_type`, `columns`, `rows`

#### `column_types(df)`
Returns one row per column with `dtype`, `is_numeric`, and `is_categorical` flags.

Response `data` fields: `view_type`, `columns`, `rows`

---

### Filtering

#### `filter_dataframe(df, filter_expr)`

Filters the DataFrame using a pandas `query()`-compatible expression string.

**Examples:**
```
age > 30
city == "New York"
salary >= 50000 and department == "Physics"
`column with spaces` > 10
```

Response `data` fields: `filter_expr`, `n_rows_before`, `n_rows_after`, `columns`, `rows`

---

### 1D plots

#### `plot_categorical_1d(df, column, normalize=False, sort_order="desc")`

Returns a categorical count-bar specification.

Response `data` fields: `plot_type="categorical_count"`, `categories`, `counts`, `normalize`, `sort_order`, `logx_available`, `logy_available`

#### `plot_numeric_1d(df, column, bins=30, normalize=False)`

Returns a 1D histogram specification.

Response `data` fields: `plot_type="histogram"`, `bins` (edges), `counts`, `normalize`, `logx_available`, `logy_available`

---

### 2D plots — numeric × numeric

#### `plot_numeric_numeric(df, x, y, hue=None, kind="hist", bins=40, max_points=5000, gridsize=60)`

`kind` options:

| kind | plot_type in response | What is returned |
|---|---|---|
| `"hist"` | `hist2d` | `x_bins`, `y_bins`, `counts` (2D array) |
| `"scatter"` | `scatter` | `points` (list of records) |
| `"joint"` | `joint` | `points` + `x_marginal` + `y_marginal` histograms |
| `"contour"` | `contour` | `x_grid`, `y_grid`, `z_grid` (KDE density), `points_preview` |

All responses include `logx_available` and `logy_available`.

---

### 2D plots — numeric × categorical

#### `plot_numeric_categorical(df, x, y, hue=None, kind="box", max_points=5000)`

Convention: `x` is numeric, `y` is categorical. If `hue` is not provided, `y` is used as the hue.

`kind` options:

| kind | What is returned |
|---|---|
| `"bar"` | `summary` + `bars` (mean ± std per group) |
| `"box"` | `summary` (quartile stats) + `points` (raw sampled data) |
| `"violin"` | `summary` + `points` |
| `"swarm"` | `summary` + `points` |
| `"swarmonbox"` | `summary` + `points` |

`summary` rows contain: `count`, `mean`, `std`, `min`, `q1`, `median`, `q3`, `max`, `hue_value`

---

### 2D plots — categorical × categorical

#### `plot_categorical_categorical(df, x, y)`

Returns a contingency table for use as a heatmap.

Response `data` fields: `plot_type="heatmap"`, `x_categories`, `y_categories`, `values` (2D int array)

---

### Generic dispatcher

#### `plot_two_columns(df, x, y, hue=None, kind=None, **kwargs)`

Automatically routes to the correct specialized function based on column types:

| x dtype | y dtype | Routes to |
|---|---|---|
| numeric | numeric | `plot_numeric_numeric` (default kind: `"hist"`) |
| numeric | categorical | `plot_numeric_categorical` (default kind: `"box"`) |
| categorical | numeric | `plot_numeric_categorical` with x/y swapped |
| categorical | categorical | `plot_categorical_categorical` |

---

### Regression

#### `regression_analysis(df, x, y, order=1, logx=False, robust=False, lowess=False, max_points=5000, fit_points=200)`

Returns scatter points, a fitted curve, and Pearson correlation.

| Parameter | Effect |
|---|---|
| `order` | Polynomial degree for `np.polyfit` |
| `logx` | Fit against `log(x)` (requires all x > 0) |
| `robust` | Robust linear fit via `statsmodels.RLM` (order=1 only) |
| `lowess` | LOWESS smoothing via `statsmodels` (ignores `order`) |

Response `data` fields: `pearson_correlation`, `points`, `fit` (`fit_type`, `x_fit`, `y_fit`, `coefficients`)

---

### Log scale flags

All plot responses include:

```json
"logx_available": true,
"logy_available": false
```

**Rule:** a log scale is marked unavailable if the corresponding data contains any non-positive values. The frontend should use these flags to enable or disable log-scale toggles in the UI.

---

## Backend usage example

```python
from dataset_store import get_dataset_by_id
import EDA

df = get_dataset_by_id(dataset_id)

result = EDA.show_head(df, n=10)
# result["status"] == "success"
# result["data"]["rows"] is a list of dicts
```

---

## Extension guide

When adding a new EDA function:

1. Accept `df: pd.DataFrame` as the first argument.
2. Return a JSON-serializable dict using `_success(...)`, `_error(...)`, or `_warning(...)`.
3. Use `_replace_nan_with_none(...)` / `_json_ready(...)` before returning any numpy or pandas data.
4. Add a corresponding test case in `offline_test.py` (do not delete existing tests).

---

## Known limitations

- Dataset store is **in-process** and **single-node**: state is not shared across workers or restarts.
- EDA functions are **not optimized for very large datasets** (>1M rows); `max_points` sampling is used for point-based outputs.
- `contour` plots require `scipy`; LOWESS and robust regression require `statsmodels`. Both dependencies fail gracefully with an error response if absent.
- `filter_dataframe` uses `df.query()` with `engine="python"` and does a lightweight column-token check; it does not sanitize arbitrary expressions for production security.

---

## Summary

```
dataset_store.py  →  data lifecycle (upload, cache, retrieve)
EDA.py            →  analysis and plot computation
offline_test.py   →  local test + visual validation
frontend          →  all rendering (tables, charts, messages)
```
