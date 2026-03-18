# EDA.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
except Exception:  # pragma: no cover
    sm_lowess = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


"""
Frontend integration note
-------------------------
1. For "column click + filter input" behavior, the frontend should construct a valid
   pandas-query-style filter string and call `filter_dataframe(...)`.

   Example:
       column = "age"
       operator = ">"
       value = 30
       frontend builds: "age > 30"

   For string values:
       column = "city"
       operator = "=="
       value = "New York"
       frontend builds: 'city == "New York"'

2. All public functions in this file return JSON-friendly Python dictionaries
   (serializable to JSON by FastAPI / Flask / similar frameworks).

3. Plotting functions return plot specifications and data payloads for frontend rendering.
   The backend here does NOT return PNGs.

4. For each 1D / 2D histogram-like output, `logx_available` / `logy_available` indicate
   whether that axis can safely be displayed on a log scale under the rule:
   "not available if there exist non-positive values".
"""


# ============================================================================
# Generic helpers
# ============================================================================

def _success(data: dict[str, Any], message: str | None = None) -> dict[str, Any]:
    """Wrap a successful payload in a consistent JSON-friendly response."""
    payload = {"status": "success", "data": data}
    if message:
        payload["message"] = message
    return payload


def _error(message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap an error payload in a consistent JSON-friendly response."""
    payload = {"status": "error", "message": message}
    if details:
        payload["details"] = details
    return payload


def _warning(message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap a warning payload in a consistent JSON-friendly response."""
    payload = {"status": "warning", "message": message}
    if data is not None:
        payload["data"] = data
    return payload


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> str | None:
    """Return None if all columns exist; otherwise return an error message."""
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        return f"Invalid column(s): {missing}"
    return None


def _is_numeric(series: pd.Series) -> bool:
    """Return True if a Series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    """Return True if a Series should be treated as categorical for plotting."""
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _replace_nan_with_none(obj: Any) -> Any:
    """Recursively replace NaN / inf-like values with None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: _replace_nan_with_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan_with_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_nan_with_none(v) for v in obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if pd.isna(obj):
        return None

    return obj


def _records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-friendly records."""
    out = df.where(df.notna(), None).to_dict(orient="records")
    return _replace_nan_with_none(out)


def _json_ready(value: Any) -> Any:
    """Convert nested pandas / numpy structures into JSON-friendly Python objects."""
    if isinstance(value, pd.DataFrame):
        return _records_from_df(value)
    if isinstance(value, pd.Series):
        return _replace_nan_with_none(value.to_dict())
    return _replace_nan_with_none(value)


def _series_log_available(series: pd.Series) -> bool:
    """Return whether a series can be shown on a log scale under the positive-only rule."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return False
    return bool((numeric > 0).all())


def _counts_log_available(counts: np.ndarray | pd.Series) -> bool:
    """Return whether histogram / count values are all strictly positive."""
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return False
    return bool(np.all(arr > 0))


def _maybe_normalize_counts(counts: np.ndarray, normalize: bool) -> np.ndarray:
    """Normalize counts to unity if requested and if the total is positive."""
    counts = counts.astype(float)
    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
    return counts


def _sample_df(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    """Return either the full DataFrame or a random sample capped at max_points."""
    if max_points is None or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=0)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric-only version of a column with NaNs dropped."""
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _sort_count_series(series: pd.Series, sort_order: str | None) -> pd.Series:
    """Sort a count series according to the requested order."""
    if sort_order is None:
        return series
    order = sort_order.lower()
    if order == "asc":
        return series.sort_values(ascending=True)
    if order == "desc":
        return series.sort_values(ascending=False)
    return series


def _category_box_stats(series: pd.Series) -> dict[str, Any]:
    """Compute box-plot-like summary statistics for one numeric group."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
        }

    return _replace_nan_with_none(
        {
            "count": int(clean.count()),
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if clean.count() > 1 else None,
            "min": float(clean.min()),
            "q1": float(clean.quantile(0.25)),
            "median": float(clean.quantile(0.50)),
            "q3": float(clean.quantile(0.75)),
            "max": float(clean.max()),
        }
    )


def _maybe_apply_hue_default(hue: str | None, fallback: str) -> str:
    """Use fallback hue when hue is empty or None."""
    return hue if hue else fallback


# ============================================================================
# Dataframe viewing
# ============================================================================

def show_head(df: pd.DataFrame, n: int = 5) -> dict[str, Any]:
    """
    Return the first n rows of a DataFrame in JSON-friendly table form.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    n:
        Number of rows to display, matching the spirit of pandas `df.head(n)`.

    Returns
    -------
    dict
        JSON-friendly response with columns, row records, and shape metadata.
    """
    out = df.head(n)
    return _success(
        {
            "view_type": "head",
            "n_requested": int(n),
            "n_returned": int(len(out)),
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": [str(c) for c in out.columns],
            "rows": _records_from_df(out),
        }
    )


def describe_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return DataFrame summary statistics similar to `df.describe(include='all')`.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    Returns
    -------
    dict
        JSON-friendly response with describe-table columns and rows.
    """
    desc = df.describe(include="all").transpose().reset_index()
    desc = desc.rename(columns={"index": "column"})
    return _success(
        {
            "view_type": "describe",
            "columns": [str(c) for c in desc.columns],
            "rows": _records_from_df(desc),
        }
    )


def column_types(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return DataFrame column dtypes similar to `df.dtypes()`.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    Returns
    -------
    dict
        JSON-friendly response with one row per column and its dtype.
    """
    out = pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [str(dtype) for dtype in df.dtypes],
            "is_numeric": [bool(_is_numeric(df[col])) for col in df.columns],
            "is_categorical": [bool(_is_categorical(df[col])) for col in df.columns],
        }
    )
    return _success(
        {
            "view_type": "dtypes",
            "columns": [str(c) for c in out.columns],
            "rows": _records_from_df(out),
        }
    )


# ============================================================================
# Filtering
# ============================================================================

def filter_dataframe(
    df: pd.DataFrame,
    filter_expr: str,
) -> dict[str, Any]:
    """
    Filter a DataFrame using a pandas-query-style expression.

    This function is intended to support both:
    1. free-form filter strings typed by the user, and
    2. frontend-built expressions from column-click interactions.

    Examples
    --------
    - "age > 30"
    - 'city == "New York"'
    - "salary >= 50000 and department == 'Physics'"

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    filter_expr:
        Query expression to be evaluated by `DataFrame.query()`.

    Returns
    -------
    dict
        JSON-friendly response containing the filtered DataFrame as table rows.
    """
    if not filter_expr or not filter_expr.strip():
        return _error("Filter expression is empty.")

    # Lightweight token check against existing columns:
    # This is intentionally simple and defensive, not a full parser.
    possible_columns = [
        col for col in df.columns
        if any(
            token in filter_expr
            for token in [col, f"`{col}`"]
        )
    ]
    if not possible_columns:
        return _warning(
            "No valid column names were detected in the filter string.",
            data={"filter_expr": filter_expr},
        )

    try:
        filtered = df.query(filter_expr, engine="python")
    except Exception as exc:
        return _error(
            "Failed to apply filter expression.",
            details={"filter_expr": filter_expr, "exception": str(exc)},
        )

    return _success(
        {
            "filter_expr": filter_expr,
            "n_rows_before": int(len(df)),
            "n_rows_after": int(len(filtered)),
            "columns": [str(c) for c in filtered.columns],
            "rows": _records_from_df(filtered),
        }
    )


# ============================================================================
# 1D plots
# ============================================================================

def plot_categorical_1d(
    df: pd.DataFrame,
    column: str,
    normalize: bool = False,
    sort_order: str | None = "desc",
) -> dict[str, Any]:
    """
    Return a categorical count-plot specification for one column.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    column:
        Categorical column to count.
    normalize:
        If True, normalize counts to unity.
    sort_order:
        "asc", "desc", or None.

    Returns
    -------
    dict
        JSON-friendly categorical plot spec for frontend rendering.
    """
    err = _validate_columns(df, [column])
    if err:
        return _error(err)

    series = df[column].fillna("<<MISSING>>").astype(str)
    counts = series.value_counts(dropna=False)
    counts = _sort_count_series(counts, sort_order)

    values = counts.to_numpy(dtype=float)
    values = _maybe_normalize_counts(values, normalize)

    payload = {
        "plot_family": "1d",
        "plot_type": "categorical_count",
        "column": column,
        "normalize": bool(normalize),
        "sort_order": sort_order,
        "categories": counts.index.tolist(),
        "counts": values.tolist(),
        "logx_available": False,
        "logy_available": _counts_log_available(values),
    }
    return _success(_json_ready(payload))


def plot_numeric_1d(
    df: pd.DataFrame,
    column: str,
    bins: int | str = 30,
    normalize: bool = False,
) -> dict[str, Any]:
    """
    Return a 1D histogram specification for one numerical column.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    column:
        Numeric column to histogram.
    bins:
        Histogram bin specification accepted by numpy / pandas, typically int or "auto".
    normalize:
        If True, normalize counts to unity.

    Returns
    -------
    dict
        JSON-friendly histogram spec with bin edges and counts.
    """
    err = _validate_columns(df, [column])
    if err:
        return _error(err)
    if not _is_numeric(df[column]):
        return _error(f"Column '{column}' is not numeric.")

    series = _numeric_series(df, column)
    if series.empty:
        return _error(f"Column '{column}' has no valid numeric values.")

    counts, edges = np.histogram(series.to_numpy(), bins=bins)
    counts = _maybe_normalize_counts(counts, normalize)

    payload = {
        "plot_family": "1d",
        "plot_type": "histogram",
        "column": column,
        "normalize": bool(normalize),
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "logx_available": _series_log_available(series),
        "logy_available": _counts_log_available(counts),
    }
    return _success(_json_ready(payload))


# ============================================================================
# Numerical x Numerical
# ============================================================================

def _numeric_numeric_hist2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    bins: int | tuple[int, int] = 40,
) -> dict[str, Any]:
    """Build a 2D histogram specification for two numeric columns."""
    plot_df = df[[x, y]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    counts, x_edges, y_edges = np.histogram2d(
        plot_df[x].to_numpy(),
        plot_df[y].to_numpy(),
        bins=bins,
    )

    payload = {
        "plot_family": "2d",
        "plot_type": "hist2d",
        "x": x,
        "y": y,
        "x_bins": x_edges.tolist(),
        "y_bins": y_edges.tolist(),
        "counts": counts.tolist(),
        "colorbar_label": "count",
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def _numeric_numeric_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    max_points: int = 5000,
) -> dict[str, Any]:
    """Build a scatter-style point payload for two numeric columns."""
    cols = [x, y] + ([hue] if hue else [])
    plot_df = df[cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    plot_df = _sample_df(plot_df, max_points)

    payload = {
        "plot_family": "2d",
        "plot_type": "scatter",
        "x": x,
        "y": y,
        "hue": hue,
        "points": _records_from_df(plot_df),
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def _numeric_numeric_joint(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    bins: int = 30,
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Build a 'jointplot-like' payload:
    scatter points + x-marginal histogram + y-marginal histogram.
    """
    cols = [x, y] + ([hue] if hue else [])
    plot_df = df[cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    scatter_df = _sample_df(plot_df, max_points)
    x_counts, x_edges = np.histogram(plot_df[x].to_numpy(), bins=bins)
    y_counts, y_edges = np.histogram(plot_df[y].to_numpy(), bins=bins)

    payload = {
        "plot_family": "2d",
        "plot_type": "joint",
        "x": x,
        "y": y,
        "hue": hue,
        "points": _records_from_df(scatter_df),
        "x_marginal": {
            "bins": x_edges.tolist(),
            "counts": x_counts.tolist(),
            "logx_available": _series_log_available(plot_df[x]),
            "logy_available": _counts_log_available(x_counts),
        },
        "y_marginal": {
            "bins": y_edges.tolist(),
            "counts": y_counts.tolist(),
            "logx_available": _series_log_available(plot_df[y]),
            "logy_available": _counts_log_available(y_counts),
        },
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def _numeric_numeric_contour(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    gridsize: int = 60,
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Build a KDE contour payload for two numeric columns.

    If scipy is unavailable, returns an error response.
    Hue is returned for frontend grouping / legend use, but the density grid here
    is computed only for x-y jointly, not separately per hue level.
    """
    if gaussian_kde is None:
        return _error("Contour / KDE output requires scipy.stats.gaussian_kde, which is unavailable.")

    cols = [x, y] + ([hue] if hue else [])
    plot_df = df[cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if len(plot_df) < 2:
        return _error("At least two valid points are required for contour / KDE output.")

    x_vals = plot_df[x].to_numpy(dtype=float)
    y_vals = plot_df[y].to_numpy(dtype=float)

    x_grid = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), gridsize)
    y_grid = np.linspace(float(np.min(y_vals)), float(np.max(y_vals)), gridsize)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_vals, y_vals])

    try:
        kde = gaussian_kde(values)
        density = np.reshape(kde(positions).T, xx.shape)
    except Exception as exc:
        return _error("Failed to compute KDE contour grid.", details={"exception": str(exc)})

    payload = {
        "plot_family": "2d",
        "plot_type": "contour",
        "x": x,
        "y": y,
        "hue": hue,
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "z_grid": density.tolist(),
        "points_preview": _records_from_df(_sample_df(plot_df, max_points)),
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))


def plot_numeric_numeric(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    kind: str = "hist",
    bins: int | tuple[int, int] = 40,
    max_points: int = 5000,
    gridsize: int = 60,
) -> dict[str, Any]:
    """
    Plot two numeric columns with one of: "hist", "joint", "scatter", "contour".

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Numeric columns.
    hue:
        Optional hue column for grouped display on the frontend.
    kind:
        One of "hist", "joint", "scatter", "contour".
    bins:
        Bin specification for histogram-based outputs.
    max_points:
        Maximum number of raw points returned for point-based rendering.
    gridsize:
        Grid resolution for contour / KDE output.

    Returns
    -------
    dict
        JSON-friendly 2D plot specification.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)
    if not _is_numeric(df[x]) or not _is_numeric(df[y]):
        return _error(f"Both '{x}' and '{y}' must be numeric columns.")

    kind = kind.lower()
    if kind == "hist":
        return _numeric_numeric_hist2d(df, x, y, bins=bins)
    if kind == "joint":
        return _numeric_numeric_joint(df, x, y, hue=hue, bins=int(bins) if not isinstance(bins, tuple) else 40, max_points=max_points)
    if kind == "scatter":
        return _numeric_numeric_scatter(df, x, y, hue=hue, max_points=max_points)
    if kind == "contour":
        return _numeric_numeric_contour(df, x, y, hue=hue, gridsize=gridsize, max_points=max_points)

    return _error("Invalid kind for numeric-numeric plot.", details={"allowed": ["hist", "joint", "scatter", "contour"]})


# ============================================================================
# Numeric x Categorical
# ============================================================================

def plot_numeric_categorical(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    kind: str = "box",
    max_points: int = 5000,
) -> dict[str, Any]:
    """
    Plot a numeric column against a categorical column with one of:
    "bar", "violin", "box", "swarm", "swarmonbox".

    Convention:
    - x is numeric
    - y is categorical

    If hue is not provided, y is also used as hue for frontend color grouping.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x:
        Numeric column.
    y:
        Categorical column.
    hue:
        Optional hue column. If absent, defaults to y.
    kind:
        One of "bar", "violin", "box", "swarm", "swarmonbox".
    max_points:
        Maximum number of raw points returned for point-based styles.

    Returns
    -------
    dict
        JSON-friendly grouped plot specification.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)
    if not _is_numeric(df[x]):
        return _error(f"Column '{x}' must be numeric.")
    if not _is_categorical(df[y]):
        return _error(f"Column '{y}' must be categorical.")

    effective_hue = _maybe_apply_hue_default(hue, y)
    # When hue resolves to the same column as y, avoid selecting / grouping on it twice.
    hue_same_as_y = (effective_hue == y)
    select_cols = [x, y] + ([] if hue_same_as_y else ([effective_hue] if effective_hue else []))
    plot_df = df[select_cols].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])

    if plot_df.empty:
        return _error("No valid rows remain after dropping NaNs in required columns.")

    kind = kind.lower()
    # Use a single key when hue and y are the same column to avoid the pandas
    # "not 1-dimensional" error that occurs when groupby receives duplicate keys.
    group_keys = [y] if (hue_same_as_y or not effective_hue) else [y, effective_hue]
    grouped = plot_df.groupby(group_keys, dropna=False)

    summary_rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        group_y = key[0]
        group_hue = group_y if hue_same_as_y else (key[1] if len(key) > 1 else None)

        stats = _category_box_stats(group[x])
        stats.update({y: group_y, "hue_value": group_hue})
        summary_rows.append(stats)

    payload: dict[str, Any] = {
        "plot_family": "2d",
        "plot_type": kind,
        "x": x,
        "y": y,
        "hue": effective_hue,
        "summary": _json_ready(summary_rows),
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": False,
    }

    if kind == "bar":
        bar_rows = (
            plot_df.groupby(group_keys, dropna=False)[x]
            .agg(["mean", "count", "std"])
            .reset_index()
            .rename(columns={"mean": "value"})
        )
        payload["bars"] = _records_from_df(bar_rows)
        return _success(_json_ready(payload))

    if kind in {"violin", "box", "swarm", "swarmonbox"}:
        payload["points"] = _records_from_df(_sample_df(plot_df, max_points))
        return _success(_json_ready(payload))

    return _error(
        "Invalid kind for numeric-categorical plot.",
        details={"allowed": ["bar", "violin", "box", "swarm", "swarmonbox"]},
    )


# ============================================================================
# Categorical x Categorical
# ============================================================================

def plot_categorical_categorical(
    df: pd.DataFrame,
    x: str,
    y: str,
) -> dict[str, Any]:
    """
    Plot two categorical columns against each other as a heatmap-like contingency table.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Categorical columns.

    Returns
    -------
    dict
        JSON-friendly heatmap specification using a contingency table.
    """
    err = _validate_columns(df, [x, y])
    if err:
        return _error(err)
    if not _is_categorical(df[x]) or not _is_categorical(df[y]):
        return _error(f"Both '{x}' and '{y}' must be categorical columns.")

    x_series = df[x].fillna("<<MISSING>>").astype(str)
    y_series = df[y].fillna("<<MISSING>>").astype(str)
    table = pd.crosstab(index=y_series, columns=x_series, dropna=False)

    payload = {
        "plot_family": "2d",
        "plot_type": "heatmap",
        "x": x,
        "y": y,
        "x_categories": [str(c) for c in table.columns],
        "y_categories": [str(i) for i in table.index],
        "values": table.to_numpy(dtype=int).tolist(),
        "logx_available": False,
        "logy_available": False,
    }
    return _success(_json_ready(payload))


# ============================================================================
# Generic 2-column dispatcher
# ============================================================================

def plot_two_columns(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    kind: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Dispatch to the appropriate 2-column plotting function based on column types.

    Rules
    -----
    - If both x and y are numeric:
        call `plot_numeric_numeric(...)`
    - If one is numeric and one is categorical:
        call `plot_numeric_categorical(...)`
        If x is categorical and y is numeric, swap them internally so the numeric
        variable remains on x and the categorical variable remains on y.
    - If both are categorical:
        ignore hue and call `plot_categorical_categorical(...)`

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Column names.
    hue:
        Optional hue column.
    kind:
        Plot-style option passed through to the chosen function.
    **kwargs:
        Additional keyword arguments passed through to the chosen function.

    Returns
    -------
    dict
        JSON-friendly plotting response from the selected specialized function.
    """
    err = _validate_columns(df, [x, y] + ([hue] if hue else []))
    if err:
        return _error(err)

    x_is_num = _is_numeric(df[x])
    y_is_num = _is_numeric(df[y])
    x_is_cat = _is_categorical(df[x])
    y_is_cat = _is_categorical(df[y])

    if x_is_num and y_is_num:
        return plot_numeric_numeric(df, x=x, y=y, hue=hue, kind=kind or "hist", **kwargs)

    if x_is_num and y_is_cat:
        return plot_numeric_categorical(df, x=x, y=y, hue=hue, kind=kind or "box", **kwargs)

    if x_is_cat and y_is_num:
        # Switch so numeric stays on x and categorical stays on y.
        return plot_numeric_categorical(df, x=y, y=x, hue=hue, kind=kind or "box", **kwargs)

    if x_is_cat and y_is_cat:
        return plot_categorical_categorical(df, x=x, y=y)

    return _error(
        "Could not determine valid plot dispatch from the input column types.",
        details={
            "x": x,
            "y": y,
            "x_dtype": str(df[x].dtype),
            "y_dtype": str(df[y].dtype),
        },
    )


# ============================================================================
# Regression
# ============================================================================

def regression_analysis(
    df: pd.DataFrame,
    x: str,
    y: str,
    order: int = 1,
    logx: bool = False,
    robust: bool = False,
    lowess: bool = False,
    max_points: int = 5000,
    fit_points: int = 200,
) -> dict[str, Any]:
    """
    Compute regression-oriented output for two numeric columns.

    This is intended to back a frontend option similar in spirit to seaborn.regplot:
    - polynomial fit via `order`
    - optional log-x fit
    - optional robust linear fit
    - optional LOWESS smoothing

    Notes
    -----
    - `lowess=True` requires statsmodels.
    - `robust=True` for linear fits requires statsmodels.
    - `logx=True` requires x > 0 for all used fit points.
    - Pearson correlation is returned as part of the JSON output.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    x, y:
        Numeric columns.
    order:
        Polynomial order for the fit. Ignored by lowess. For robust fits, only linear
        order=1 is supported here.
    logx:
        If True, fit against log(x) while still returning original x values for plotting.
    robust:
        If True, attempt a robust linear fit.
    lowess:
        If True, attempt LOWESS smoothing.
    max_points:
        Maximum number of raw points returned for scatter rendering.
    fit_points:
        Number of x-grid points used for returned fitted curves.

    Returns
    -------
    dict
        JSON-friendly regression output including point preview, fit curve, and
        Pearson correlation.
    """
    err = _validate_columns(df, [x, y])
    if err:
        return _error(err)
    if not _is_numeric(df[x]) or not _is_numeric(df[y]):
        return _error(
            "Regression requires both columns to be numeric.",
            details={"x_dtype": str(df[x].dtype), "y_dtype": str(df[y].dtype)},
        )

    plot_df = df[[x, y]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _error("No valid numeric pairs remain after dropping NaNs.")

    if logx and not _series_log_available(plot_df[x]):
        return _error(f"logx=True is invalid because column '{x}' contains non-positive values.")

    x_vals = plot_df[x].to_numpy(dtype=float)
    y_vals = plot_df[y].to_numpy(dtype=float)

    x_fit = np.log(x_vals) if logx else x_vals
    x_grid_original = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), fit_points)
    x_grid_fit = np.log(x_grid_original) if logx else x_grid_original

    pearson = float(plot_df[[x, y]].corr(method="pearson").iloc[0, 1])

    fit_payload: dict[str, Any] = {
        "fit_type": None,
        "x_fit": None,
        "y_fit": None,
        "coefficients": None,
        "warnings": [],
    }

    try:
        if lowess:
            if sm_lowess is None:
                return _error("LOWESS requested but statsmodels is unavailable.")
            fitted = sm_lowess(endog=y_vals, exog=x_fit, frac=0.66, return_sorted=True)
            x_fit_sorted = fitted[:, 0]
            y_fit_sorted = fitted[:, 1]

            if logx:
                x_fit_sorted = np.exp(x_fit_sorted)

            fit_payload.update(
                {
                    "fit_type": "lowess",
                    "x_fit": x_fit_sorted.tolist(),
                    "y_fit": y_fit_sorted.tolist(),
                }
            )

        elif robust:
            if order != 1:
                return _error("Robust fitting is only implemented here for order=1.")
            if sm is None:
                return _error("Robust fit requested but statsmodels is unavailable.")

            X = sm.add_constant(x_fit)
            model = sm.RLM(y_vals, X)
            result = model.fit()

            y_grid = result.predict(sm.add_constant(x_grid_fit))
            fit_payload.update(
                {
                    "fit_type": "robust_linear",
                    "x_fit": x_grid_original.tolist(),
                    "y_fit": y_grid.tolist(),
                    "coefficients": result.params.tolist(),
                }
            )

        else:
            coeffs = np.polyfit(x_fit, y_vals, deg=order)
            y_grid = np.polyval(coeffs, x_grid_fit)
            fit_payload.update(
                {
                    "fit_type": "polynomial",
                    "x_fit": x_grid_original.tolist(),
                    "y_fit": y_grid.tolist(),
                    "coefficients": coeffs.tolist(),
                }
            )

    except Exception as exc:
        return _error("Regression fit failed.", details={"exception": str(exc)})

    payload = {
        "plot_family": "2d",
        "plot_type": "regression",
        "x": x,
        "y": y,
        "order": int(order),
        "logx": bool(logx),
        "robust": bool(robust),
        "lowess": bool(lowess),
        "pearson_correlation": pearson,
        "points": _records_from_df(_sample_df(plot_df, max_points)),
        "fit": fit_payload,
        "logx_available": _series_log_available(plot_df[x]),
        "logy_available": _series_log_available(plot_df[y]),
    }
    return _success(_json_ready(payload))