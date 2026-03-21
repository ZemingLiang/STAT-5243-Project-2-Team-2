# offline_test.py

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from pprint import pformat
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import EDA

# Optional dataset store integration.
# This script will still work if dataset_store is absent.
try:
    import dataset_store
    HAS_DATASET_STORE = True
except Exception:
    HAS_DATASET_STORE = False


TEST_CSV_PATH = Path("test_data/sleep_mobile_stress_dataset_15000.csv")
OUTPUT_DIR = Path("offline_test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Generic helpers
# =============================================================================

def save_json(name: str, payload: dict[str, Any]) -> Path:
    """Save a JSON payload to disk."""
    path = OUTPUT_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def save_text(name: str, text: str) -> Path:
    """Save a text blob to disk."""
    path = OUTPUT_DIR / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def print_section(title: str) -> None:
    """Print a terminal section header."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def is_success(payload: dict[str, Any]) -> bool:
    """Check whether an EDA response payload represents success."""
    return payload.get("status") == "success"


def get_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the data field from a successful payload."""
    return payload.get("data", {})


def try_numeric_columns(df: pd.DataFrame, min_needed: int = 2) -> list[str]:
    """Return numeric columns."""
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < min_needed:
        raise RuntimeError(f"Need at least {min_needed} numeric columns, found {len(cols)}.")
    return cols


def try_categorical_columns(df: pd.DataFrame, min_needed: int = 2) -> list[str]:
    """Return categorical-like columns."""
    cols = [
        c for c in df.columns
        if (
            pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
        )
    ]
    if len(cols) < min_needed:
        raise RuntimeError(f"Need at least {min_needed} categorical columns, found {len(cols)}.")
    return cols


def _is_id_like_column(series: pd.Series, name: str) -> bool:
    """Return True if a column looks like a row identifier rather than a measurement."""
    lname = name.lower().strip()
    if (
        lname in {"id", "idx", "index", "row", "rowid"}
        or lname.endswith("_id")
        or lname.startswith("id_")
    ):
        return True
    # Also flag strictly-unique integer columns (every value distinct → sequential ID).
    if pd.api.types.is_integer_dtype(series) and series.nunique() == len(series):
        return True
    return False


def pick_test_columns(df: pd.DataFrame) -> dict[str, Any]:
    """
    Pick a reasonable set of columns for offline testing.

    ID-like columns (e.g. 'user_id', strictly unique integers) are pushed to
    the end of the numeric list so more informative measurement columns are
    selected first.
    """
    all_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Partition: real measurements first, ID-like columns last.
    numeric_cols = (
        [c for c in all_numeric if not _is_id_like_column(df[c], c)]
        + [c for c in all_numeric if _is_id_like_column(df[c], c)]
    )
    categorical_cols = [
        c for c in df.columns
        if (
            pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
        )
    ]

    if len(numeric_cols) < 2:
        # Try coercion-based fallback for "numeric-looking" object columns.
        for col in df.columns:
            if col in numeric_cols:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0.8 * len(df):
                numeric_cols.append(col)

    if len(numeric_cols) < 2:
        raise RuntimeError("Could not identify at least 2 usable numeric columns.")

    # If there are no obvious categorical columns, derive one from a numeric column for testing.
    if len(categorical_cols) < 1:
        derived = numeric_cols[0]
        tmp = f"{derived}_binned_cat"
        q = min(4, max(2, df[derived].nunique()))
        df[tmp] = pd.qcut(df[derived], q=q, duplicates="drop").astype(str)
        categorical_cols.append(tmp)

    if len(categorical_cols) < 2:
        derived = numeric_cols[1]
        tmp = f"{derived}_binned_cat"
        q = min(4, max(2, df[derived].nunique()))
        df[tmp] = pd.qcut(df[derived], q=q, duplicates="drop").astype(str)
        categorical_cols.append(tmp)

    result = {
        "numeric_1": numeric_cols[0],
        "numeric_2": numeric_cols[1],
        "numeric_3": numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0],
        "categorical_1": categorical_cols[0],
        "categorical_2": categorical_cols[1],
        "hue": categorical_cols[0] if categorical_cols else None,
    }
    return result


# =============================================================================
# Renderers from EDA JSON payloads
# =============================================================================

def render_plot_from_payload(name: str, payload: dict[str, Any]) -> Path | None:
    """
    Render a plot PNG from the JSON payload returned by EDA.py.

    Supported plot types:
    - categorical_count
    - histogram
    - hist2d
    - scatter
    - joint
    - contour
    - heatmap
    - bar
    - violin
    - box
    - swarm
    - swarmonbox
    - regression
    - line
    - multiline
    """
    status = payload.get("status")
    if status == "warning":
        print(f"[WARN] {name} has warning status: {payload.get('message', '')}")
    elif status != "success":
        print(f"[WARN] Not rendering {name}: payload status = {status}")
        return None

    data = get_data(payload)
    plot_type = data.get("plot_type")
    if not plot_type:
        print(f"[INFO] {name} has no plot_type; skipping PNG.")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    try:
        if plot_type == "categorical_count":
            _render_categorical_count(ax, data)

        elif plot_type == "histogram":
            _render_histogram(ax, data)

        elif plot_type == "hist2d":
            _render_hist2d(ax, fig, data)

        elif plot_type == "scatter":
            _render_scatter(ax, data)

        elif plot_type == "joint":
            plt.close(fig)
            fig = _render_joint(data)

        elif plot_type == "contour":
            _render_contour(ax, data)

        elif plot_type == "heatmap":
            _render_heatmap(ax, fig, data)

        elif plot_type == "bar":
            _render_numeric_categorical_bar(ax, data)

        elif plot_type in {"violin", "box", "swarm", "swarmonbox"}:
            _render_numeric_categorical_summary(ax, data)

        elif plot_type == "regression":
            _render_regression(ax, data)

        elif plot_type == "line":
            _render_line(ax, data)

        elif plot_type == "multiline":
            _render_multiline(ax, data)

        else:
            plt.close(fig)
            print(f"[INFO] Unsupported plot_type for rendering: {plot_type}")
            return None

        out = OUTPUT_DIR / f"{name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    except Exception as exc:
        plt.close(fig)
        print(f"[WARN] Failed to render {name}: {exc}")
        return None


def _render_categorical_count(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render a categorical count plot."""
    categories = [str(x) for x in data["categories"]]
    counts = np.asarray(data["counts"], dtype=float)

    ax.bar(range(len(categories)), counts)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_xlabel(data.get("column", "category"))
    ax.set_ylabel("fraction" if data.get("normalize") else "count")
    ax.set_title(f"Categorical count: {data.get('column')}")


def _render_histogram(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render a 1D histogram from bin edges and counts."""
    bins = np.asarray(data["bins"], dtype=float)
    counts = np.asarray(data["counts"], dtype=float)

    widths = np.diff(bins)
    ax.bar(bins[:-1], counts, width=widths, align="edge")
    ax.set_xlabel(data.get("column", "x"))
    ax.set_ylabel("fraction" if data.get("normalize") else "count")
    ax.set_title(f"Histogram: {data.get('column')}")

    if data.get("logx_available"):
        # Do not force log; just mention capability.
        ax.text(0.98, 0.95, "logx available", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
    if data.get("logy_available"):
        ax.text(0.98, 0.90, "logy available", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)


def _render_hist2d(ax: plt.Axes, fig: plt.Figure, data: dict[str, Any]) -> None:
    """Render a 2D histogram heatmap."""
    x_bins = np.asarray(data["x_bins"], dtype=float)
    y_bins = np.asarray(data["y_bins"], dtype=float)
    counts = np.asarray(data["counts"], dtype=float)

    mesh = ax.pcolormesh(x_bins, y_bins, counts.T, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(data.get("colorbar_label", "count"))
    ax.set_xlabel(data.get("x", "x"))
    ax.set_ylabel(data.get("y", "y"))
    ax.set_title(f"2D histogram: {data.get('x')} vs {data.get('y')}")


def _render_scatter(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render a scatter plot."""
    points = pd.DataFrame(data["points"])
    x = data["x"]
    y = data["y"]

    if data.get("hue") and data["hue"] in points.columns:
        for label, sub in points.groupby(data["hue"]):
            ax.scatter(sub[x], sub[y], label=str(label), alpha=0.7)
        ax.legend(title=data["hue"])
    else:
        ax.scatter(points[x], points[y], alpha=0.7)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Scatter: {x} vs {y}")


def _render_joint(data: dict[str, Any]) -> plt.Figure:
    """Render a simple joint-style figure with scatter and marginals."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], figure=fig)

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    points = pd.DataFrame(data["points"])
    x = data["x"]
    y = data["y"]

    ax_scatter.scatter(points[x], points[y], alpha=0.6)
    ax_scatter.set_xlabel(x)
    ax_scatter.set_ylabel(y)
    ax_scatter.set_title(f"Joint plot: {x} vs {y}")

    x_m = data["x_marginal"]
    xb = np.asarray(x_m["bins"], dtype=float)
    xc = np.asarray(x_m["counts"], dtype=float)
    ax_histx.bar(xb[:-1], xc, width=np.diff(xb), align="edge")
    ax_histx.set_ylabel("count")
    ax_histx.tick_params(labelbottom=False)

    y_m = data["y_marginal"]
    yb = np.asarray(y_m["bins"], dtype=float)
    yc = np.asarray(y_m["counts"], dtype=float)
    ax_histy.barh(yb[:-1], yc, height=np.diff(yb), align="edge")
    ax_histy.set_xlabel("count")
    ax_histy.tick_params(labelleft=False)

    return fig


def _render_contour(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render a contour plot from x_grid, y_grid, z_grid."""
    x_grid = np.asarray(data["x_grid"], dtype=float)
    y_grid = np.asarray(data["y_grid"], dtype=float)
    z_grid = np.asarray(data["z_grid"], dtype=float)

    xx, yy = np.meshgrid(x_grid, y_grid)
    ax.contourf(xx, yy, z_grid, levels=12)
    ax.set_xlabel(data.get("x", "x"))
    ax.set_ylabel(data.get("y", "y"))
    ax.set_title(f"Contour: {data.get('x')} vs {data.get('y')}")


def _render_heatmap(ax: plt.Axes, fig: plt.Figure, data: dict[str, Any]) -> None:
    """Render a categorical-categorical heatmap."""
    values = np.asarray(data["values"], dtype=float)
    im = ax.imshow(values, aspect="auto", origin="upper")
    fig.colorbar(im, ax=ax)

    x_categories = [str(v) for v in data["x_categories"]]
    y_categories = [str(v) for v in data["y_categories"]]

    ax.set_xticks(range(len(x_categories)))
    ax.set_xticklabels(x_categories, rotation=45, ha="right")
    ax.set_yticks(range(len(y_categories)))
    ax.set_yticklabels(y_categories)

    ax.set_xlabel(data.get("x", "x"))
    ax.set_ylabel(data.get("y", "y"))
    ax.set_title(f"Heatmap: {data.get('x')} vs {data.get('y')}")


def _render_numeric_categorical_bar(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render grouped bar-like output using summary rows."""
    rows = pd.DataFrame(data["bars"])
    x = data["x"]
    y = data["y"]

    # Aggregate by category for a simple offline rendering.
    summary = rows.groupby(y, dropna=False)["value"].mean().reset_index()
    ax.bar(summary[y].astype(str), summary["value"])
    ax.set_xlabel(y)
    ax.set_ylabel(f"mean({x})")
    ax.set_title(f"Bar summary: {x} vs {y}")
    ax.tick_params(axis="x", rotation=45)


def _render_numeric_categorical_summary(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render summary markers with IQR-style vertical lines."""
    rows = pd.DataFrame(data["summary"])
    ycol = data["y"]

    cats = rows[ycol].astype(str).tolist()
    xpos = np.arange(len(cats))

    med = pd.to_numeric(rows["median"], errors="coerce").to_numpy(dtype=float)
    q1 = pd.to_numeric(rows["q1"], errors="coerce").to_numpy(dtype=float)
    q3 = pd.to_numeric(rows["q3"], errors="coerce").to_numpy(dtype=float)
    ymin = pd.to_numeric(rows["min"], errors="coerce").to_numpy(dtype=float)
    ymax = pd.to_numeric(rows["max"], errors="coerce").to_numpy(dtype=float)

    for i in range(len(rows)):
        if np.isnan(ymin[i]) or np.isnan(ymax[i]):
            continue
        ax.vlines(xpos[i], ymin[i], ymax[i], linewidth=1.5)
        if not (np.isnan(q1[i]) or np.isnan(q3[i])):
            ax.vlines(xpos[i], q1[i], q3[i], linewidth=5)
        if not np.isnan(med[i]):
            ax.scatter([xpos[i]], [med[i]], zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_xlabel(ycol)
    ax.set_ylabel(data["x"])
    ax.set_title(f"{data['plot_type'].capitalize()} summary: {data['x']} vs {ycol}")


def _render_regression(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render regression scatter and fit."""
    points = pd.DataFrame(data["points"])
    x = data["x"]
    y = data["y"]

    ax.scatter(points[x], points[y], alpha=0.5, label="data")

    fit = data.get("fit", {})
    if fit.get("x_fit") is not None and fit.get("y_fit") is not None:
        ax.plot(fit["x_fit"], fit["y_fit"], linewidth=2, label=fit.get("fit_type", "fit"))

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    pearson = data.get("pearson_correlation")
    pearson_str = f"{pearson:.4f}" if pearson is not None else "N/A"
    ax.set_title(f"Regression: {x} vs {y} | Pearson={pearson_str}")
    ax.legend()


def _render_line(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render a single binned/sorted line plot from x_values / y_values."""
    x_vals = np.asarray(data["x_values"], dtype=float)
    y_vals = np.asarray(data["y_values"], dtype=float)

    ax.plot(x_vals, y_vals, alpha=0.85, linewidth=1.5)
    ax.set_xlabel(data.get("x", "x"))
    ax.set_ylabel(data.get("y", "y"))

    nbins = data.get("nbins")
    agg_note = f"  (mean per {nbins} x-bins)" if nbins else ""
    ax.set_title(f"Line: {data.get('y')} vs {data.get('x')}{agg_note}")


def _render_multiline(ax: plt.Axes, data: dict[str, Any]) -> None:
    """Render multiple lines on one axis from a plot_multiline payload."""
    mode = data.get("mode", "2d")
    lines = data.get("lines", [])
    warnings = data.get("warnings", [])

    for line in lines:
        x = np.asarray(line.get("x", []), dtype=float)
        y = np.asarray(line.get("y", []), dtype=float)
        ax.plot(x, y, label=str(line.get("label", "")), alpha=0.85)

    ax.legend(fontsize=8)

    if mode == "1d":
        column = data.get("column", "value")
        normalize = data.get("normalize", False)
        ax.set_xlabel(column)
        ax.set_ylabel("fraction" if normalize else "count")
        group_src = data.get("group_column") or data.get("group_source", "")
        ax.set_title(f"Multi-line distribution: {column}  (groups: {group_src})")
    else:
        x_col = data.get("x_column", "x")
        y_col = data.get("y_column", "y")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        group_src = data.get("group_column") or data.get("group_source", "")
        ax.set_title(f"Multi-line: {y_col} vs {x_col}  (groups: {group_src})")

    if warnings:
        ax.text(0.02, 0.98, f"({len(warnings)} group(s) skipped)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="darkorange")


# =============================================================================
# Offline presentation helpers
# =============================================================================

def present_table_payload(name: str, payload: dict[str, Any], max_rows: int = 10) -> None:
    """Pretty-print table-like payloads to terminal and save a text snapshot."""
    data = get_data(payload)

    text_lines = [f"{name}", "-" * len(name)]
    text_lines.append(pformat(payload, width=120))
    text_blob = "\n".join(text_lines)
    save_text(name, text_blob)

    print(f"[SAVED] {name}.txt")

    if "rows" in data and "columns" in data:
        df = pd.DataFrame(data["rows"])
        print(df.head(max_rows).to_string(index=False))
    else:
        print(pformat(payload, width=120))


def present_generic_payload(name: str, payload: dict[str, Any]) -> None:
    """Print a compact summary and save JSON."""
    out = save_json(name, payload)
    print(f"[SAVED] {out}")

    status = payload.get("status")
    print(f"status = {status}")
    if status not in {"success", "warning"}:
        print(pformat(payload, width=120))
        return

    if status == "warning":
        print(f"  warning: {payload.get('message', '')}")

    data = get_data(payload)
    keys = list(data.keys())
    print(f"data keys = {keys}")
    for w in data.get("warnings", []):
        print(f"  [SKIPPED] {w}")


# =============================================================================
# Main test workflow
# =============================================================================

def test_dataset_store(csv_path: Path) -> pd.DataFrame:
    """Test dataset_store if available; otherwise fall back to direct CSV load."""
    print_section("Dataset loading / dataset_store test")

    if HAS_DATASET_STORE:
        print("[INFO] dataset_store detected; testing load_uploaded_csv / get_dataset_by_id")

        with csv_path.open("rb") as f:
            meta = dataset_store.load_uploaded_csv(f, csv_path.name)

        save_json("dataset_store_metadata", meta)
        print("[SAVED] dataset_store_metadata.json")
        print("Loaded metadata:")
        print(pformat(meta, width=120))

        dataset_id = meta["dataset_id"]
        df = dataset_store.get_dataset_by_id(dataset_id)
        print(f"[INFO] Retrieved dataset by id = {dataset_id}")
        print(f"[INFO] Shape = {df.shape}")
        return df

    print("[INFO] dataset_store not available; loading CSV directly with pandas")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Shape = {df.shape}")
    return df


def run_dataframe_view_tests(df: pd.DataFrame) -> None:
    """Test head / describe / column types."""
    print_section("Dataframe viewing tests")

    payload = EDA.show_head(df, n=8)
    present_table_payload("show_head", payload)

    payload = EDA.describe_dataframe(df)
    present_table_payload("describe_dataframe", payload)

    payload = EDA.column_types(df)
    present_table_payload("column_types", payload)


def run_filter_tests(df: pd.DataFrame, cols: dict[str, Any]) -> pd.DataFrame:
    """Test dataframe filtering and return a filtered frame for later use."""
    print_section("Filtering tests")

    num_col = cols["numeric_1"]
    threshold = float(pd.to_numeric(df[num_col], errors="coerce").dropna().median())
    expr = f"`{num_col}` > {threshold}"

    payload = EDA.filter_dataframe(df, expr)
    present_table_payload("filter_dataframe", payload)

    if is_success(payload):
        filtered_rows = get_data(payload)["rows"]
        filtered_df = pd.DataFrame(filtered_rows)
        print(f"[INFO] Filtered shape reconstructed from JSON = {filtered_df.shape}")
        return filtered_df

    return df


def run_plot_tests(df: pd.DataFrame, cols: dict[str, Any]) -> None:
    """Test plotting-style EDA functions and render offline PNGs."""
    print_section("Plotting tests")

    # 1D categorical
    payload = EDA.plot_categorical_1d(
        df,
        column=cols["categorical_1"],
        normalize=False,
        sort_order="desc",
    )
    present_generic_payload("plot_categorical_1d", payload)
    render_plot_from_payload("plot_categorical_1d", payload)

    # 1D numeric
    payload = EDA.plot_numeric_1d(
        df,
        column=cols["numeric_1"],
        bins=30,
        normalize=False,
    )
    present_generic_payload("plot_numeric_1d", payload)
    render_plot_from_payload("plot_numeric_1d", payload)

    # numeric-numeric: hist
    payload = EDA.plot_numeric_numeric(
        df,
        x=cols["numeric_1"],
        y=cols["numeric_2"],
        hue=cols["hue"],
        kind="hist",
        bins=35,
    )
    present_generic_payload("plot_numeric_numeric_hist", payload)
    render_plot_from_payload("plot_numeric_numeric_hist", payload)

    # numeric-numeric: joint
    payload = EDA.plot_numeric_numeric(
        df,
        x=cols["numeric_1"],
        y=cols["numeric_2"],
        hue=cols["hue"],
        kind="joint",
        bins=25,
        max_points=3000,
    )
    present_generic_payload("plot_numeric_numeric_joint", payload)
    render_plot_from_payload("plot_numeric_numeric_joint", payload)

    # numeric-numeric: scatter
    payload = EDA.plot_numeric_numeric(
        df,
        x=cols["numeric_1"],
        y=cols["numeric_2"],
        hue=cols["hue"],
        kind="scatter",
        max_points=3000,
    )
    present_generic_payload("plot_numeric_numeric_scatter", payload)
    render_plot_from_payload("plot_numeric_numeric_scatter", payload)

    # numeric-numeric: contour
    payload = EDA.plot_numeric_numeric(
        df,
        x=cols["numeric_1"],
        y=cols["numeric_2"],
        hue=cols["hue"],
        kind="contour",
        gridsize=50,
        max_points=2000,
    )
    present_generic_payload("plot_numeric_numeric_contour", payload)
    render_plot_from_payload("plot_numeric_numeric_contour", payload)

    # numeric-categorical: bar
    payload = EDA.plot_numeric_categorical(
        df,
        x=cols["numeric_1"],
        y=cols["categorical_1"],
        hue=cols["hue"],
        kind="bar",
    )
    present_generic_payload("plot_numeric_categorical_bar", payload)
    render_plot_from_payload("plot_numeric_categorical_bar", payload)

    # numeric-categorical: box
    payload = EDA.plot_numeric_categorical(
        df,
        x=cols["numeric_1"],
        y=cols["categorical_1"],
        hue=cols["hue"],
        kind="box",
        max_points=3000,
    )
    present_generic_payload("plot_numeric_categorical_box", payload)
    render_plot_from_payload("plot_numeric_categorical_box", payload)

    # categorical-categorical heatmap
    payload = EDA.plot_categorical_categorical(
        df,
        x=cols["categorical_1"],
        y=cols["categorical_2"],
    )
    present_generic_payload("plot_categorical_categorical", payload)
    render_plot_from_payload("plot_categorical_categorical", payload)

    # generic dispatcher: numeric-numeric
    payload = EDA.plot_two_columns(
        df,
        x=cols["numeric_1"],
        y=cols["numeric_2"],
        hue=cols["hue"],
        kind="scatter",
        max_points=3000,
    )
    present_generic_payload("plot_two_columns_num_num", payload)
    render_plot_from_payload("plot_two_columns_num_num", payload)

    # generic dispatcher: numeric-categorical
    payload = EDA.plot_two_columns(
        df,
        x=cols["numeric_1"],
        y=cols["categorical_1"],
        hue=cols["hue"],
        kind="box",
        max_points=3000,
    )
    present_generic_payload("plot_two_columns_num_cat", payload)
    render_plot_from_payload("plot_two_columns_num_cat", payload)

    # generic dispatcher: categorical-categorical
    payload = EDA.plot_two_columns(
        df,
        x=cols["categorical_1"],
        y=cols["categorical_2"],
    )
    present_generic_payload("plot_two_columns_cat_cat", payload)
    render_plot_from_payload("plot_two_columns_cat_cat", payload)


def run_regression_tests(df: pd.DataFrame, cols: dict[str, Any]) -> None:
    """Test regression analysis."""
    print_section("Regression tests")

    payload = EDA.regression_analysis(
        df,
        x=cols["numeric_1"],
        y=cols["numeric_2"],
        order=1,
        logx=False,
        robust=False,
        lowess=False,
        max_points=3000,
        fit_points=200,
    )
    present_generic_payload("regression_analysis_linear", payload)
    render_plot_from_payload("regression_analysis_linear", payload)

    # Try logx only if valid.
    x_series = pd.to_numeric(df[cols["numeric_1"]], errors="coerce").dropna()
    if not x_series.empty and (x_series > 0).all():
        payload = EDA.regression_analysis(
            df,
            x=cols["numeric_1"],
            y=cols["numeric_2"],
            order=2,
            logx=True,
            robust=False,
            lowess=False,
            max_points=3000,
            fit_points=200,
        )
        present_generic_payload("regression_analysis_logx_poly2", payload)
        render_plot_from_payload("regression_analysis_logx_poly2", payload)
    else:
        print("[INFO] Skipping logx regression test: x contains non-positive values.")


def run_line_tests(df: pd.DataFrame, cols: dict[str, Any]) -> None:
    """Test kind='line' in plot_numeric_numeric."""
    print_section("Line plot tests  (plot_numeric_numeric  kind='line')")

    # Use non-ID numeric columns (pick_test_columns already deprioritises IDs).
    x_col = cols["numeric_1"]   # e.g. age
    y_col = cols["numeric_2"]   # e.g. daily_screen_time_hours
    cat_col = cols["hue"]       # e.g. gender

    # ── Single-line (no hue), default nbins=30 ───────────────────────────────
    payload = EDA.plot_numeric_numeric(
        df, x=x_col, y=y_col, kind="line",
    )
    assert is_success(payload), f"Expected success, got {payload.get('status')}"
    data = get_data(payload)
    assert data.get("plot_type") == "line"
    assert "x_values" in data and "y_values" in data
    assert len(data["x_values"]) == len(data["y_values"])
    assert data.get("aggregated") is True, "Expected binned aggregation by default"
    present_generic_payload("plot_numeric_numeric_line", payload)
    render_plot_from_payload("plot_numeric_numeric_line", payload)
    print("[PASS] kind='line' basic (binned, no hue)")

    # ── Single-line with explicit nbins ─────────────────────────────────────
    payload = EDA.plot_numeric_numeric(
        df, x=x_col, y=y_col, kind="line", nbins=15,
    )
    assert is_success(payload)
    assert get_data(payload).get("nbins") == 15
    present_generic_payload("plot_numeric_numeric_line_nbins15", payload)
    render_plot_from_payload("plot_numeric_numeric_line_nbins15", payload)
    print("[PASS] kind='line' with explicit nbins=15")

    # ── Multi-line via hue (returns multiline response) ──────────────────────
    payload = EDA.plot_numeric_numeric(
        df, x=x_col, y=y_col, hue=cat_col, kind="line",
    )
    assert is_success(payload), f"Expected success, got {payload.get('status')}"
    data = get_data(payload)
    # When hue is categorical, response is plot_type="multiline" with one line per group.
    assert data.get("plot_type") == "multiline", (
        f"Expected multiline response when hue is provided, got {data.get('plot_type')}"
    )
    assert data.get("mode") == "2d"
    assert data.get("group_column") == cat_col
    assert len(data.get("lines", [])) >= 1
    present_generic_payload("plot_numeric_numeric_line_hue", payload)
    render_plot_from_payload("plot_numeric_numeric_line_hue", payload)
    print("[PASS] kind='line' with hue → multiline response")


def run_multiline_tests(df: pd.DataFrame, cols: dict[str, Any]) -> None:
    """Test plot_multiline — both group modes and both draw modes."""
    print_section("Multi-line plot tests  (plot_multiline)")

    num_col  = cols["numeric_2"]   # e.g. age
    num_col2 = cols["numeric_3"]   # e.g. daily_screen_time_hours
    cat_col  = cols["categorical_1"]  # e.g. gender

    median_val = float(pd.to_numeric(df[num_col], errors="coerce").dropna().median())

    # ── 1D, categorical split ───────────────────────────────────────────────
    payload = EDA.plot_multiline(
        df,
        column=num_col,
        group_by=cat_col,
        normalize=True,
        nbins=20,
    )
    assert is_success(payload), f"Expected success, got {payload.get('status')}"
    data = get_data(payload)
    assert data["mode"] == "1d"
    assert len(data["lines"]) >= 1
    present_generic_payload("multiline_1d_categorical", payload)
    render_plot_from_payload("multiline_1d_categorical", payload)
    print("[PASS] multiline 1D categorical split")

    # ── 1D, filter strings with partial failures → warning ──────────────────
    payload = EDA.plot_multiline(
        df,
        column=num_col2,
        filter_strings=[
            f"`{num_col}` < {median_val}",          # valid
            f"`{num_col}` >= {median_val}",         # valid
            f"`{num_col}` > 9999",                  # valid syntax but empty → warning
            "nonexistent_col_xyz > 0",              # no column match → warning
        ],
        filter_labels=["Below median", "Above median", "Empty filter", "Invalid filter"],
        normalize=True,
        nbins=20,
    )
    assert payload.get("status") == "warning", (
        f"Expected warning (partial failures), got {payload.get('status')}"
    )
    data = get_data(payload)
    assert data["mode"] == "1d"
    assert len(data["lines"]) == 2, f"Expected 2 valid lines, got {len(data['lines'])}"
    assert len(data["warnings"]) == 2, f"Expected 2 warnings, got {len(data['warnings'])}"
    present_generic_payload("multiline_1d_filters_partial", payload)
    render_plot_from_payload("multiline_1d_filters_partial", payload)
    print("[PASS] multiline 1D filter-strings with partial failures → warning")

    # ── 2D, categorical split ───────────────────────────────────────────────
    payload = EDA.plot_multiline(
        df,
        column=num_col,
        x_column=num_col2,
        group_by=cat_col,
        sort_x=True,
        max_points_per_line=2000,
    )
    assert is_success(payload), f"Expected success, got {payload.get('status')}"
    data = get_data(payload)
    assert data["mode"] == "2d"
    assert len(data["lines"]) >= 1
    present_generic_payload("multiline_2d_categorical", payload)
    render_plot_from_payload("multiline_2d_categorical", payload)
    print("[PASS] multiline 2D categorical split")

    # ── 2D, filter strings (all valid) ──────────────────────────────────────
    payload = EDA.plot_multiline(
        df,
        column=num_col,
        x_column=num_col2,
        filter_strings=[
            f"`{num_col}` < {median_val}",
            f"`{num_col}` >= {median_val}",
        ],
        filter_labels=["Below median", "Above median"],
        sort_x=True,
        max_points_per_line=2000,
    )
    assert is_success(payload), f"Expected success, got {payload.get('status')}"
    data = get_data(payload)
    assert data["mode"] == "2d" and len(data["lines"]) == 2
    present_generic_payload("multiline_2d_filters", payload)
    render_plot_from_payload("multiline_2d_filters", payload)
    print("[PASS] multiline 2D filter-strings all valid")

    # ── Error: both group_by and filter_strings ──────────────────────────────
    payload = EDA.plot_multiline(
        df, column=num_col,
        group_by=cat_col, filter_strings=[f"`{num_col}` > 0"],
    )
    assert payload.get("status") == "error"
    print("[PASS] multiline error on both group_by + filter_strings")

    # ── Error: non-categorical group_by ──────────────────────────────────────
    payload = EDA.plot_multiline(df, column=num_col2, group_by=num_col)
    assert payload.get("status") == "error"
    print("[PASS] multiline error on non-categorical group_by")

    # ── Error: all filters fail ───────────────────────────────────────────────
    payload = EDA.plot_multiline(
        df,
        column=num_col,
        filter_strings=[f"`{num_col}` > 9999", "nonexistent_col_xyz > 0"],
        filter_labels=["Empty", "Invalid"],
    )
    assert payload.get("status") == "error"
    print("[PASS] multiline error when all filters fail")


def main() -> None:
    """Run the full offline test workflow for dataset_store.py and EDA.py."""
    print_section("Offline EDA test runner")

    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV_PATH}\n"
            f"Expected relative path from project root."
        )

    df = test_dataset_store(TEST_CSV_PATH)

    # Make a working copy, since we may add derived categorical columns.
    df = df.copy()

    print_section("Column auto-selection")
    cols = pick_test_columns(df)
    print(pformat(cols, width=120))
    save_json("picked_test_columns", cols)
    print("[SAVED] picked_test_columns.json")

    run_dataframe_view_tests(df)
    filtered_df = run_filter_tests(df, cols)

    # Use filtered_df only if non-empty, otherwise original.
    working_df = filtered_df if not filtered_df.empty else df
    if working_df.shape[0] < 10:
        working_df = df

    run_plot_tests(working_df, cols)
    run_regression_tests(working_df, cols)
    run_line_tests(working_df, cols)
    run_multiline_tests(working_df, cols)

    print_section("Done")
    print(f"Offline test outputs saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()