# EDA Frontend Integration Guide

This document is for **frontend developers** integrating with the EDA backend (`EDA.py` + API layer).

It explains:
- What backend functions exist and what they return
- The exact shape of every JSON response
- What the frontend needs to render for each response type
- How to wire UI interactions to backend calls

---

## 1. High-level flow

```
User uploads CSV
    → backend returns dataset_id

User clicks action (e.g. "Show head", "Plot histogram")
    → frontend sends: dataset_id + action parameters
    → backend returns JSON
    → frontend checks status, renders accordingly
```

---

## 2. Standard response envelope

Every backend response follows one of three shapes:

### Success
```json
{
  "status": "success",
  "data": { ... }
}
```

### Warning
```json
{
  "status": "warning",
  "message": "...",
  "data": { ... }
}
```

### Error
```json
{
  "status": "error",
  "message": "...",
  "details": { ... }
}
```

### Frontend handling rules

| Status | Action |
|---|---|
| `"success"` | Render normally |
| `"warning"` | Render content + show warning banner |
| `"error"` | Show error message, do NOT attempt to render a plot |

> **Note:** `null` values in JSON (e.g. in statistics rows) indicate missing data (NaN) and should be displayed as "—" or empty cells rather than crashing.

---

## 3. Dataframe viewing

### 3.1 Show head

**Backend call:**
```
GET /head?dataset_id=...&n=8
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "view_type": "head",
    "n_requested": 8,
    "n_returned": 8,
    "shape": { "rows": 15000, "cols": 13 },
    "columns": ["col1", "col2", "col3"],
    "rows": [
      { "col1": 1, "col2": "Alice", "col3": 3.14 },
      ...
    ]
  }
}
```

**Frontend:** render `columns` as table header, `rows` as table body.

---

### 3.2 Describe

**Backend call:**
```
GET /describe?dataset_id=...
```

**Response:** same `columns` + `rows` structure as above, where rows contain summary stats (`count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`, plus `unique`/`top`/`freq` for categoricals).

**Frontend:** render as table. Cells may be `null` (e.g. `mean` for a string column) — display as "—".

---

### 3.3 Column types

**Backend call:**
```
GET /column_types?dataset_id=...
```

**Response:**
```json
{
  "data": {
    "rows": [
      { "column": "age",    "dtype": "int64",   "is_numeric": true,  "is_categorical": false },
      { "column": "gender", "dtype": "str",     "is_numeric": false, "is_categorical": true  }
    ]
  }
}
```

**Frontend:**
- Display as a table.
- Use `is_numeric` / `is_categorical` flags for UI logic: e.g. only allow numeric columns in histogram selectors, only allow categorical columns in bar-category selectors.

---

## 4. Filtering

Filtering uses a clean three-layer architecture:

| Layer | Function | Responsibility |
|---|---|---|
| EDA | `apply_filter(df, expr)` | Pure pandas query, raises ValueError |
| Dataset store | `register_dataframe(df, ...)` | Persist result, assign new dataset_id |
| **API (call this)** | `filter_and_save_dataset(dataset_id, expr)` | Orchestrate + return JSON |

**Backend call (frontend invokes this endpoint):**
```
POST /filter
{
  "dataset_id": "abc123",
  "filter_expr": "age > 30"
}
```

**Success response:**
```json
{
  "status": "success",
  "data": {
    "source_dataset_id": "abc123",
    "new_dataset_id": "def456",
    "n_rows_before": 15000,
    "n_rows_after": 8143
  }
}
```

The `new_dataset_id` is the id of the newly saved filtered dataset.  All
subsequent EDA calls for the filtered view should use `new_dataset_id`.

**Error response:**
```json
{ "status": "error", "message": "No valid column names found in filter expression: 'xyz > 0'." }
```

### Building the filter expression

The backend uses `pandas.DataFrame.query()` syntax. The frontend has two options:

**Option A — Free text input:** User types the expression directly.

**Option B — Structured UI:**
- Column dropdown → operator dropdown (`>`, `<`, `==`, `!=`, `>=`, `<=`) → value input
- Build string: `age > 30`, `city == "New York"`, `` `column with spaces` > 5 ``
- For columns with spaces in their names, wrap in backticks.
- For string values, wrap in double quotes inside the expression.

**Frontend responsibilities:**
- Display `n_rows_before` and `n_rows_after` to the user.
- Store `new_dataset_id` as the active dataset for subsequent EDA operations.
- On error, display `message` to the user.

---

## 5. 1D plots

### 5.1 Categorical count bar

**Request parameters:** `column`, `normalize` (bool), `sort_order` (`"asc"` / `"desc"` / `null`)

**Response:**
```json
{
  "data": {
    "plot_type": "categorical_count",
    "column": "gender",
    "categories": ["Male", "Female", "Non-binary"],
    "counts": [7234, 6891, 875],
    "normalize": false,
    "logy_available": true,
    "logx_available": false
  }
}
```

**Frontend:** render a bar chart. `categories` → x-axis labels, `counts` → bar heights. If `normalize=true`, label y-axis as "fraction". Offer log-y toggle only if `logy_available=true`.

---

### 5.2 Numeric histogram

**Request parameters:** `column`, `bins` (int), `normalize` (bool)

**Response:**
```json
{
  "data": {
    "plot_type": "histogram",
    "column": "age",
    "bins": [18.0, 22.3, 26.6, ...],
    "counts": [312, 445, ...],
    "normalize": false,
    "logx_available": true,
    "logy_available": false
  }
}
```

**Frontend:** render a histogram. `bins` is an array of **bin edges** (length = `len(counts) + 1`). Each bar spans `bins[i]` to `bins[i+1]`. Offer log-x/y toggles only if the corresponding `*_available` flag is `true`.

---

## 6. 2D plots

### 6.1 Numeric × Numeric

**Request parameters:** `x`, `y`, `hue` (optional), `kind`

#### kind = `"hist"` → 2D heatmap

```json
{
  "data": {
    "plot_type": "hist2d",
    "x": "age", "y": "sleep_hours",
    "x_bins": [...], "y_bins": [...],
    "counts": [[...], ...],
    "colorbar_label": "count"
  }
}
```

`counts` is a 2D array of shape `[n_x_bins, n_y_bins]`. Render as a color heatmap (e.g. Plotly heatmap, ECharts heatmap). Color encodes `counts`.

---

#### kind = `"scatter"` → scatter plot

```json
{
  "data": {
    "plot_type": "scatter",
    "x": "age", "y": "sleep_hours", "hue": "gender",
    "points": [
      { "age": 32, "sleep_hours": 6.5, "gender": "Female" },
      ...
    ]
  }
}
```

Render as a scatter plot. If `hue` is present in the records, color points by the hue column value.

---

#### kind = `"joint"` → scatter + marginals

```json
{
  "data": {
    "plot_type": "joint",
    "x": "age", "y": "sleep_hours",
    "points": [...],
    "x_marginal": { "bins": [...], "counts": [...] },
    "y_marginal": { "bins": [...], "counts": [...] }
  }
}
```

Render a central scatter plot with a histogram above (x-marginal) and one to the right (y-marginal).

---

#### kind = `"contour"` → KDE density contour

```json
{
  "data": {
    "plot_type": "contour",
    "x": "age", "y": "sleep_hours",
    "x_grid": [...], "y_grid": [...],
    "z_grid": [[...], ...],
    "points_preview": [...]
  }
}
```

`z_grid` is a 2D density array of shape `[gridsize, gridsize]`. Render as a contour or filled-contour plot. `points_preview` contains a sampled subset of the raw points (optional overlay).

---

#### kind = `"line"` → sorted line plot

```json
{
  "data": {
    "plot_type": "line",
    "x": "age", "y": "stress_level", "hue": "gender",
    "x_values": [18, 19, 20, ...],
    "y_values": [5.1, 6.3, 4.8, ...],
    "hue_values": ["Female", "Male", "Female", ...],
    "n_points": 2000
  }
}
```

Data is pre-sorted by `x`. Connect consecutive points to draw a single continuous line. `hue_values` is a parallel array present only when `hue` was specified; use it for per-point coloring or tooltip annotation. It does **not** split the data into multiple lines — use `plot_multiline` for that.

---

### 6.2 Numeric × Categorical

**Request parameters:** `x` (numeric), `y` (categorical), `hue` (optional), `kind`

All kinds return a `summary` array and most also return `points`.

**`summary` row structure:**
```json
{
  "gender": "Female",
  "hue_value": "Female",
  "count": 6891,
  "mean": 6.5,
  "std": 1.2,
  "min": 4.0,
  "q1": 5.5,
  "median": 6.5,
  "q3": 7.5,
  "max": 9.0
}
```

Fields can be `null` for empty groups.

| kind | Use `summary` for | Use `points` for | Suggested chart |
|---|---|---|---|
| `"bar"` | mean + error bars (use `bars` key, not `summary`) | — | grouped bar chart |
| `"box"` | quartile overlays | raw point jitter | box plot |
| `"violin"` | shape reference | distribution estimate | violin plot |
| `"swarm"` | reference | raw point swarm | swarm plot |
| `"swarmonbox"` | box quartiles | swarm overlay | swarm-on-box |

**`bars` key (bar kind only):**
```json
"bars": [
  { "gender": "Female", "value": 6.5, "count": 6891, "std": 1.2 },
  ...
]
```

---

### 6.3 Categorical × Categorical → Heatmap

```json
{
  "data": {
    "plot_type": "heatmap",
    "x": "gender", "y": "occupation",
    "x_categories": ["Female", "Male", "Non-binary"],
    "y_categories": ["Doctor", "Manager", "Teacher", ...],
    "values": [[231, 198, 22], [312, 289, 31], ...]
  }
}
```

`values[i][j]` is the count of rows where `y == y_categories[i]` and `x == x_categories[j]`. Render as a grid heatmap with color encoding count.

---

## 7. Generic plotting API

Use `plot_two_columns` when you want the backend to automatically decide the plot type:

**Request:**
```json
{
  "dataset_id": "...",
  "x": "age",
  "y": "gender",
  "hue": "occupation",
  "kind": "box"
}
```

The backend inspects column types and routes to the right specialized function. The response shape is identical to what the specialized function would return. The frontend can render by reading `plot_type` from the response.

---

## 8. Multi-line plots (`plot_multiline`)

Use this endpoint to overlay multiple lines on one canvas for group comparison.

### 8.1 Request

```
POST /multiline
{
  "dataset_id": "...",
  "column": "sleep_duration_hours",
  "x_column": "daily_screen_time_hours",   // omit for 1D mode
  "group_by": "gender",                    // OR use filter_strings — not both
  "normalize": false,
  "bins": 30,
  "sort_x": true,
  "max_points_per_line": 2000
}
```

**Grouping options — exactly one of:**

```jsonc
// Option A — split by categorical column
{ "group_by": "gender" }

// Option B — split by filter expressions
{
  "filter_strings": ["stress_level < 4", "stress_level > 8", "age > 999"],
  "filter_labels":  ["Low stress",       "High stress",      "Empty test"]
}
```

### 8.2 Response — 1D mode (`x_column` absent)

```json
{
  "status": "success",
  "data": {
    "plot_type": "multiline",
    "mode": "1d",
    "column": "sleep_duration_hours",
    "normalize": false,
    "bins": [4.0, 4.35, 4.7, "..."],
    "group_source": "categorical",
    "group_column": "gender",
    "n_lines": 3,
    "lines": [
      { "label": "Female",     "x": [4.17, 4.52, "..."], "y": [312, 445, "..."], "n_points": 6891 },
      { "label": "Male",       "x": [4.17, 4.52, "..."], "y": [290, 410, "..."], "n_points": 7234 },
      { "label": "Non-binary", "x": ["..."], "y": ["..."], "n_points": 875 }
    ],
    "logx_available": true,
    "logy_available": false,
    "warnings": []
  }
}
```

`x` in each line is **bin midpoints** (shared across all lines). `y` is counts, or fractions when `normalize=true`. All lines share the same `bins` edges — their x-axes are aligned for direct comparison.

### 8.3 Response — 2D mode (`x_column` provided)

```json
{
  "status": "success",
  "data": {
    "plot_type": "multiline",
    "mode": "2d",
    "x_column": "daily_screen_time_hours",
    "y_column": "sleep_duration_hours",
    "group_source": "filter",
    "group_column": null,
    "n_lines": 2,
    "lines": [
      { "label": "Low stress",  "x": [1.0, 1.2, "..."], "y": [7.5, 7.3, "..."], "n_points": 2000 },
      { "label": "High stress", "x": [1.0, 1.1, "..."], "y": [5.8, 6.0, "..."], "n_points": 2000 }
    ],
    "logx_available": true,
    "logy_available": true,
    "warnings": []
  }
}
```

Each line's `x` and `y` are pre-sorted by `x`. Connect them as a line chart.

### 8.4 Response — partial filter failure (`status="warning"`)

When some filters are skipped (empty selection, syntax error, or missing column), the response still delivers the valid lines with `status="warning"`:

```json
{
  "status": "warning",
  "message": "2 group(s) skipped; see data['warnings'] for details.",
  "data": {
    "plot_type": "multiline",
    "mode": "1d",
    "n_lines": 2,
    "lines": [ "..." ],
    "warnings": [
      "['Empty test'] Filter 'age > 999' produced an empty selection — skipped.",
      "['Invalid'] No valid column names found in filter 'xyz > 0' — skipped."
    ]
  }
}
```

### 8.5 Frontend responsibilities for multiline

- Iterate over `data.lines` — one line per entry.
- Use `line.label` as the legend entry.
- **1D mode:** draw a line connecting `(x[i], y[i])` for each group. Label y-axis as `"fraction"` if `normalize=true`, otherwise `"count"`.
- **2D mode:** draw a line connecting `(x[i], y[i])` per group. Points are pre-sorted by x.
- If `status == "warning"`: render the available lines AND display a warning banner; optionally show `data.warnings` in a tooltip or expandable sidebar.
- If `status == "error"`: display the error message; do not render.
- Respect `logx_available` / `logy_available` for scale toggles.

---

## 9. Regression

**Request parameters:** `x`, `y`, `order` (int), `logx` (bool), `robust` (bool), `lowess` (bool)

**Response:**
```json
{
  "data": {
    "plot_type": "regression",
    "x": "age", "y": "stress_level",
    "order": 1, "logx": false, "robust": false, "lowess": false,
    "pearson_correlation": 0.3421,
    "points": [
      { "age": 32, "stress_level": 6.5 },
      ...
    ],
    "fit": {
      "fit_type": "polynomial",
      "x_fit": [...],
      "y_fit": [...],
      "coefficients": [0.042, 4.1]
    }
  }
}
```

**Frontend:**
- Render scatter of `points`
- Overlay fit line from `fit.x_fit` / `fit.y_fit`
- Display `pearson_correlation` (may be `null` for degenerate cases)
- `fit_type` is one of `"polynomial"`, `"robust_linear"`, `"lowess"`

---

## 10. Log scale flags

Every plot response includes:

```json
"logx_available": true,
"logy_available": false
```

**Rule:** `false` means the corresponding axis contains non-positive values, so a log scale is mathematically invalid.

**Frontend:** show log-toggle UI elements only if the corresponding flag is `true`; otherwise hide or disable them with a tooltip.

---

## 11. UI action → backend function mapping

| UI action | Backend function | Notes |
|---|---|---|
| Upload CSV | `load_uploaded_csv` | Returns `dataset_id` |
| Show head | `show_head` | Table render |
| Describe | `describe_dataframe` | Table render |
| Column types | `column_types` | Use for UI logic |
| Filter | `api.filter_and_save_dataset` | Build query string in UI; store `new_dataset_id` |
| 1D histogram | `plot_numeric_1d` | |
| 1D bar | `plot_categorical_1d` | |
| 2D auto | `plot_two_columns` | Let backend dispatch |
| 2D num×num | `plot_numeric_numeric` | Explicit kind selection (hist/scatter/joint/contour/line) |
| 2D num×cat | `plot_numeric_categorical` | Explicit kind selection |
| 2D cat×cat | `plot_categorical_categorical` | |
| Multi-line | `plot_multiline` | Group by category or filter strings |
| Regression | `regression_analysis` | |

---

## 12. Frontend responsibilities checklist

### On dataset upload
- [ ] Send CSV file to upload endpoint
- [ ] Receive and store `dataset_id`
- [ ] Fetch `column_types` to populate column selectors and enable/disable plot options

### On table views
- [ ] Render `columns` + `rows` as a sortable table
- [ ] Display `null` values as "—"

### On plot requests
- [ ] Send `dataset_id`, column names, `kind`, and options
- [ ] Check `status` before rendering
- [ ] Dispatch rendering by `plot_type` value
- [ ] Respect `logx_available` / `logy_available` flags for toggle controls

### On errors
- [ ] Display `message` from the error response
- [ ] Do not attempt to render a plot

---

## 13. Important constraints

- **The backend does NOT return PNG images.** It returns data only.
- **The frontend must render everything** (charts, tables, messages).
- **NaN in JSON is represented as `null`.** Handle it in all table and chart code.
- **Sampled points:** scatter-based outputs cap returned points at `max_points` (default 5000). The full dataset is used for statistical summaries and histograms.

---

## 14. Recommended chart libraries

Any library that can consume arrays of data works. Common options:

| Library | Notes |
|---|---|
| [Plotly.js](https://plotly.com/javascript/) | Rich built-in types; good for heatmaps, contours, box plots |
| [ECharts](https://echarts.apache.org/) | Performant; good for large scatter and heatmap |
| [Chart.js](https://www.chartjs.org/) | Lightweight; best for bar / line / simple scatter |
| [Recharts](https://recharts.org/) | React-native; composable |
| [D3.js](https://d3js.org/) | Maximum control; higher implementation cost |

---

## 15. Debugging checklist

If something is not rendering correctly:

1. Log the full raw backend response in the browser console.
2. Check `response.status` first — if it is `"error"`, read `response.message`.
3. Check `response.data.plot_type` — make sure your renderer handles that type.
4. Check for `null` values in arrays that your chart library may not tolerate.
5. For regression, check that `fit.x_fit` and `fit.y_fit` are both non-null before drawing the line.
