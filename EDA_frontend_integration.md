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

**Backend call:**
```
POST /filter
{
  "dataset_id": "...",
  "filter_expr": "age > 30"
}
```

**Response:**
```json
{
  "data": {
    "filter_expr": "age > 30",
    "n_rows_before": 15000,
    "n_rows_after": 8143,
    "columns": [...],
    "rows": [...]
  }
}
```

### Building the filter expression

The backend uses `pandas.DataFrame.query()` syntax. The frontend has two options:

**Option A — Free text input:** User types the expression directly.

**Option B — Structured UI:**
- Column dropdown → operator dropdown (`>`, `<`, `==`, `!=`, `>=`, `<=`) → value input
- Build string: `age > 30`, `city == "New York"`, `` `column with spaces` > 5 ``
- For columns with spaces in their names, wrap in backticks.
- For string values, wrap in double quotes inside the expression.

**Frontend:** render filtered table, show row counts before/after.

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

## 8. Regression

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

## 9. Log scale flags

Every plot response includes:

```json
"logx_available": true,
"logy_available": false
```

**Rule:** `false` means the corresponding axis contains non-positive values, so a log scale is mathematically invalid.

**Frontend:** show log-toggle UI elements only if the corresponding flag is `true`; otherwise hide or disable them with a tooltip.

---

## 10. UI action → backend function mapping

| UI action | Backend function | Notes |
|---|---|---|
| Upload CSV | `load_uploaded_csv` | Returns `dataset_id` |
| Show head | `show_head` | Table render |
| Describe | `describe_dataframe` | Table render |
| Column types | `column_types` | Use for UI logic |
| Filter | `filter_dataframe` | Build query string in UI |
| 1D histogram | `plot_numeric_1d` | |
| 1D bar | `plot_categorical_1d` | |
| 2D auto | `plot_two_columns` | Let backend dispatch |
| 2D num×num | `plot_numeric_numeric` | Explicit kind selection |
| 2D num×cat | `plot_numeric_categorical` | Explicit kind selection |
| 2D cat×cat | `plot_categorical_categorical` | |
| Regression | `regression_analysis` | |

---

## 11. Frontend responsibilities checklist

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

## 12. Important constraints

- **The backend does NOT return PNG images.** It returns data only.
- **The frontend must render everything** (charts, tables, messages).
- **NaN in JSON is represented as `null`.** Handle it in all table and chart code.
- **Sampled points:** scatter-based outputs cap returned points at `max_points` (default 5000). The full dataset is used for statistical summaries and histograms.

---

## 13. Recommended chart libraries

Any library that can consume arrays of data works. Common options:

| Library | Notes |
|---|---|
| [Plotly.js](https://plotly.com/javascript/) | Rich built-in types; good for heatmaps, contours, box plots |
| [ECharts](https://echarts.apache.org/) | Performant; good for large scatter and heatmap |
| [Chart.js](https://www.chartjs.org/) | Lightweight; best for bar / line / simple scatter |
| [Recharts](https://recharts.org/) | React-native; composable |
| [D3.js](https://d3js.org/) | Maximum control; higher implementation cost |

---

## 14. Debugging checklist

If something is not rendering correctly:

1. Log the full raw backend response in the browser console.
2. Check `response.status` first — if it is `"error"`, read `response.message`.
3. Check `response.data.plot_type` — make sure your renderer handles that type.
4. Check for `null` values in arrays that your chart library may not tolerate.
5. For regression, check that `fit.x_fit` and `fit.y_fit` are both non-null before drawing the line.
