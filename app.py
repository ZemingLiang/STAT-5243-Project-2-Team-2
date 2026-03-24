from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris as sklearn_load_iris
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly
import shinyswatch

import EDA
import feature_engineering
import p2_divided as cleaning


# ---------------------------------------------------------------------------
# Plotly global template — consistent chart styling across the entire app
# ---------------------------------------------------------------------------
_app_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="'Nunito Sans', system-ui, sans-serif", color="#1e293b", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            "#1a1a2e", "#4361ee", "#7209b7", "#06d6a0", "#f77f00",
            "#d62828", "#0891b2", "#16a34a", "#e11d48", "#ca8a04",
        ],
        title=dict(font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zeroline=False),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zeroline=False),
        margin=dict(t=50, r=16, b=40, l=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)",
        ),
    )
)
pio.templates["app_theme"] = _app_template
pio.templates.default = "app_theme"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = BASE_DIR / "test_data" / "sleep_mobile_stress_dataset_15000.csv"


# ---------------------------------------------------------------------------
# Pure helper functions (NO changes from original)
# ---------------------------------------------------------------------------
def load_builtin_dataset(name: str) -> pd.DataFrame:
    if name == "sleep_health":
        return pd.read_csv(TEST_DATA_PATH)
    if name == "iris":
        return sklearn_load_iris(as_frame=True).frame
    if name == "tips":
        import seaborn as sns
        return sns.load_dataset("tips")
    raise ValueError(f"Unknown built-in dataset: {name}")


def load_uploaded_dataset(path: str, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return cleaning.load_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return cleaning.load_excel(path)
    if suffix == ".json":
        return cleaning.load_json(path)
    if suffix == ".rds":
        return cleaning.load_rds(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def next_dataset_key(datasets: OrderedDict[str, dict[str, Any]], prefix: str) -> str:
    if prefix == "original" and "original" not in datasets:
        return "original"
    index = 1
    while True:
        key = f"{prefix}_{index:02d}"
        if key not in datasets:
            return key
        index += 1


def register_dataset_version(
    datasets: OrderedDict[str, dict[str, Any]],
    df: pd.DataFrame,
    *,
    prefix: str,
    label: str,
    source_key: str | None = None,
    transform: str | None = None,
) -> tuple[OrderedDict[str, dict[str, Any]], str]:
    key = next_dataset_key(datasets, prefix)
    new_datasets = OrderedDict(datasets)
    new_datasets[key] = {
        "label": label,
        "df": df.copy(),
        "source_key": source_key,
        "transform": transform,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return new_datasets, key


def overwrite_dataset_version(
    datasets: OrderedDict[str, dict[str, Any]],
    key: str,
    df: pd.DataFrame,
    *,
    transform: str | None = None,
) -> OrderedDict[str, dict[str, Any]]:
    new_datasets = OrderedDict(datasets)
    record = dict(new_datasets[key])
    record["df"] = df.copy()
    record["transform"] = transform
    record["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_datasets[key] = record
    return new_datasets


def format_history_table(datasets: OrderedDict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, record in datasets.items():
        df = record["df"]
        rows.append(
            {
                "key": key,
                "label": record["label"],
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "source": record["source_key"] or "-",
                "transform": record["transform"] or "-",
                "created_at": record["created_at"],
            }
        )
    return pd.DataFrame(rows)


def coerce_text_value(value: str | None) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def dataframe_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data", {})
    columns = data.get("columns", [])
    rows = data.get("rows", [])
    if not rows and not columns:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=columns or None)


def current_overview(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None:
        return None
    return cleaning.get_overview(df)


def current_column_types(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["column", "dtype", "is_numeric", "is_categorical"])
    payload = EDA.column_types(df)
    return pd.DataFrame(payload["data"]["rows"])


def midpoints(edges: list[float]) -> list[float]:
    return [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(edges) - 1)]


def widths(edges: list[float]) -> list[float]:
    return [float(edges[i + 1]) - float(edges[i]) for i in range(len(edges) - 1)]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def empty_figure(title: str = "No plot yet.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title)
    return fig


def build_comparison_figure(
    before: pd.Series, after: pd.Series, col_name: str
) -> go.Figure:
    """Side-by-side before/after distribution chart."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Before", "After"])
    if pd.api.types.is_numeric_dtype(before):
        fig.add_histogram(
            x=before.dropna(), name="Before",
            marker_color="#94a3b8", row=1, col=1,
        )
        fig.add_histogram(
            x=after.dropna(), name="After",
            marker_color="#4361ee", row=1, col=2,
        )
    else:
        vc_before = before.value_counts().head(15)
        vc_after = after.value_counts().head(15)
        fig.add_bar(
            x=vc_before.index.astype(str), y=vc_before.values,
            name="Before", marker_color="#94a3b8", row=1, col=1,
        )
        fig.add_bar(
            x=vc_after.index.astype(str), y=vc_after.values,
            name="After", marker_color="#4361ee", row=1, col=2,
        )
    fig.update_layout(
        title=f"Before / After: {col_name}",
        showlegend=False,
        height=280,
        margin=dict(t=50, b=30, l=40, r=20),
    )
    return fig


def build_rowcount_figure(before_count: int, after_count: int, action: str) -> go.Figure:
    """Bar chart comparing row counts before vs after a row-removal operation."""
    fig = go.Figure()
    removed = before_count - after_count
    fig.add_bar(
        x=["Before", "After"],
        y=[before_count, after_count],
        marker_color=["#94a3b8", "#4361ee"],
        text=[f"{before_count:,}", f"{after_count:,}"],
        textposition="outside",
        width=0.5,
    )
    fig.update_layout(
        title=f"{action}: {removed:,} rows removed ({before_count:,} → {after_count:,})",
        yaxis_title="Row count",
        height=280,
        margin=dict(t=50, b=30, l=60, r=20),
    )
    return fig


def figure_from_payload(payload: dict[str, Any]) -> go.Figure:
    status = payload.get("status")
    if status == "error":
        return empty_figure(payload.get("message", "Unable to render plot."))

    data = payload.get("data", {})
    plot_type = data.get("plot_type")
    fig = go.Figure()

    if plot_type == "categorical_count":
        fig.add_bar(x=data["categories"], y=data["counts"], marker_color="#4361ee")
        fig.update_layout(xaxis_title=data["column"], yaxis_title="Count")

    elif plot_type == "histogram":
        fig.add_bar(
            x=midpoints(data["bins"]),
            y=data["counts"],
            width=widths(data["bins"]),
            marker_color="#1a1a2e",
        )
        fig.update_layout(xaxis_title=data["column"], yaxis_title="Count")

    elif plot_type == "scatter":
        points = pd.DataFrame(data["points"])
        if points.empty:
            return empty_figure("No scatter data available.")
        if data.get("hue"):
            for label, group in points.groupby(data["hue"]):
                fig.add_scattergl(
                    x=group[data["x"]],
                    y=group[data["y"]],
                    mode="markers",
                    name=str(label),
                    opacity=0.6,
                )
        else:
            fig.add_scattergl(
                x=points[data["x"]],
                y=points[data["y"]],
                mode="markers",
                name=f"{data['x']} vs {data['y']}",
                opacity=0.6,
            )
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "hist2d":
        fig.add_heatmap(
            x=midpoints(data["x_bins"]),
            y=midpoints(data["y_bins"]),
            z=data["counts"],
            colorscale="YlOrRd",
        )
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "bar":
        bars = pd.DataFrame(data["bars"])
        hue_col = "hue_value"
        if not bars.empty and hue_col in bars.columns:
            for label, group in bars.groupby(hue_col):
                fig.add_bar(
                    x=group[data["y"]],
                    y=group["value"],
                    name=str(label),
                )
        else:
            fig.add_bar(x=bars[data["y"]], y=bars["value"], name="Value")
        fig.update_layout(xaxis_title=data["y"], yaxis_title=data["x"], barmode="group")

    elif plot_type == "box":
        points = pd.DataFrame(data["points"])
        if points.empty:
            return empty_figure("No box-plot data available.")
        hue_col = data.get("hue") or data["y"]
        for label, group in points.groupby(hue_col):
            fig.add_box(
                x=group[data["y"]],
                y=group[data["x"]],
                name=str(label),
                boxpoints="outliers",
            )
        fig.update_layout(xaxis_title=data["y"], yaxis_title=data["x"])

    elif plot_type == "heatmap":
        fig.add_heatmap(
            x=data["x_categories"],
            y=data["y_categories"],
            z=data["values"],
            colorscale="Blues",
        )
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "regression":
        points = pd.DataFrame(data["points"])
        fig.add_scattergl(
            x=points[data["x"]],
            y=points[data["y"]],
            mode="markers",
            name="Points",
            opacity=0.55,
        )
        fit = data.get("fit") or {}
        if fit.get("x_fit") and fit.get("y_fit"):
            fig.add_scatter(
                x=fit["x_fit"],
                y=fit["y_fit"],
                mode="lines",
                name=fit.get("fit_type", "Fit"),
                line=dict(color="#d62828", width=3),
            )
        fig.update_layout(
            xaxis_title=data["x"],
            yaxis_title=data["y"],
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.12,
                    showarrow=False,
                    text=f"Pearson r: {round(float(data['pearson_correlation']), 4)}",
                )
            ],
        )

    elif plot_type == "multiline":
        for line in data["lines"]:
            fig.add_scatter(
                x=line["x"],
                y=line["y"],
                mode="lines",
                name=line["label"],
            )
        x_title = data["column"] if data["mode"] == "1d" else data.get("x_column", "x")
        y_title = "Count" if data["mode"] == "1d" else data["column"]
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    elif plot_type == "correlation_matrix":
        cols = data["columns"]
        fig.add_heatmap(
            x=cols,
            y=cols,
            z=data["values"],
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in data["values"]],
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        )
        fig.update_layout(
            title=f"Correlation Matrix ({data['method'].title()})",
            width=700,
            height=600,
        )

    else:
        return empty_figure(f"Unsupported plot type: {plot_type}")

    return fig


# ---------------------------------------------------------------------------
# CSS — minimal overrides on top of the Lux Bootstrap theme
# ---------------------------------------------------------------------------
APP_CSS = """
.tip-box {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-left: 4px solid #1a1a2e;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 12px;
  font-size: 0.93rem;
}
.small-note { color: #6c757d; font-size: 0.9rem; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 8px;
}
.metric {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 14px;
  text-align: center;
}
.metric .label {
  font-size: 0.72rem;
  color: #6c757d;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 600;
}
.metric .value {
  font-size: 1.3rem;
  font-weight: 700;
  color: #1a1a2e;
  margin-top: 2px;
}
.alert-stack { display: grid; gap: 6px; }
.bslib-sidebar-layout > .sidebar { border-right: 1px solid #dee2e6 !important; }
.card { transition: box-shadow 0.2s; }
.card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
"""


# ---------------------------------------------------------------------------
# UI — Lux-themed page_navbar with cards, sidebars, tooltips
# ---------------------------------------------------------------------------
app_ui = ui.page_navbar(
    # ── Guide Tab ──────────────────────────────────────────────────────────
    ui.nav_panel(
        "Guide",
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Welcome")),
                ui.p(
                    "This is an interactive data workbench built with Shiny for Python. "
                    "It lets you load, clean, transform, and explore datasets entirely in "
                    "the browser — no coding required. Every operation is backed by pure "
                    "Python functions that run locally."
                ),
            ),
            col_widths=[12],
        ),
        ui.card(
            ui.card_header(ui.strong("Step-by-Step Walkthrough")),
            ui.tags.ol(
                ui.tags.li(
                    ui.strong("Load a dataset "),
                    "— Pick a built-in dataset (Iris or Sleep/Mobile/Stress) or upload "
                    "your own CSV, Excel, JSON, or RDS file in the Load tab."
                ),
                ui.tags.li(
                    ui.strong("Inspect the data "),
                    "— The Load tab shows row/column counts, missing values, duplicates, "
                    "and a full version history of every dataset you create."
                ),
                ui.tags.li(
                    ui.strong("Clean and preprocess "),
                    "— In the Cleaning tab, handle missing values, remove duplicates, "
                    "scale numeric columns, encode categorical columns, or handle outliers. "
                    "Always preview before applying."
                ),
                ui.tags.li(
                    ui.strong("Engineer features "),
                    "— The Feature Engineering tab offers 11 transforms: log, square, cube, "
                    "interaction, ratio, binning, one-hot, standardize, normalize, fill NA, "
                    "and drop NA. Each shows a before/after comparison chart."
                ),
                ui.tags.li(
                    ui.strong("Explore with EDA "),
                    "— View summary tables, filter with pandas query expressions, "
                    "create 1D/2D plots, run regression analysis, plot multiline comparisons, "
                    "and generate a full correlation heatmap."
                ),
                ui.tags.li(
                    ui.strong("Download results "),
                    "— Use the CSV download buttons on the Load, Cleaning, and Feature "
                    "Engineering tabs to export your work."
                ),
            ),
        ),
        ui.card(
            ui.card_header(ui.strong("Tips")),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Dataset Picker: "),
                "Use the dropdown in the Load tab to switch between any version you have "
                "created (original, cleaned, feature-engineered, filtered).",
            ),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Filter Syntax: "),
                "Filtering uses pandas query expressions. Examples: ",
                ui.tags.code('age > 30 and gender == "Female"'),
                ", ",
                ui.tags.code("sepal_length > 5.0"),
                ".",
            ),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Preview First: "),
                "Both Cleaning and Feature Engineering have a Preview button. "
                "Always preview before applying to make sure the result looks correct.",
            ),
        ),
    ),
    # ── Load Tab ───────────────────────────────────────────────────────────
    ui.nav_panel(
        "Load",
        ui.layout_columns(
            ui.layout_columns(
                ui.card(
                    ui.card_header(ui.strong("Built-in Datasets")),
                    ui.input_select(
                        "builtin_dataset",
                        "Choose built-in dataset",
                        {
                            "sleep_health": "Sleep, Mobile and Stress",
                            "iris": "Iris",
                            "tips": "Tips (Restaurant)",
                        },
                    ),
                    ui.tooltip(
                        ui.input_action_button(
                            "load_builtin_btn", "Load Built-in Dataset",
                            class_="btn-dark w-100",
                        ),
                        "Load the selected built-in dataset into session memory",
                    ),
                ),
                ui.card(
                    ui.card_header(ui.strong("Upload Dataset")),
                    ui.input_file(
                        "upload_file",
                        "Upload CSV, Excel, or JSON",
                        accept=[".csv", ".xlsx", ".xls", ".json", ".rds"],
                    ),
                    ui.tooltip(
                        ui.input_action_button(
                            "load_upload_btn", "Load Uploaded File",
                            class_="btn-outline-dark w-100",
                        ),
                        "Parse and load the uploaded file",
                    ),
                ),
                col_widths=[12, 12],
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header(ui.strong("Active Dataset")),
                    ui.input_select("dataset_picker", "Active dataset version", {}),
                    ui.output_ui("active_dataset_summary"),
                    ui.download_button("download_active", "Download Active Dataset (CSV)",
                                       class_="btn-outline-dark btn-sm mt-2"),
                ),
                ui.card(
                    ui.card_header(ui.strong("Dataset History")),
                    ui.output_data_frame("history_table"),
                    full_screen=True,
                ),
                col_widths=[12, 12],
            ),
            col_widths=[4, 8],
        ),
    ),
    # ── Cleaning Tab ───────────────────────────────────────────────────────
    ui.nav_panel(
        "Cleaning",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Cleaning / Preprocessing", class_="text-uppercase fw-bold"),
                ui.hr(),
                ui.input_select(
                    "clean_action",
                    "Action",
                    {
                        "handle_missing": "Handle missing values",
                        "remove_duplicates": "Remove duplicates",
                        "scale_columns": "Scale numeric columns",
                        "encode_columns": "Encode categorical columns",
                        "handle_outliers": "Handle outliers",
                    },
                ),
                ui.input_selectize(
                    "clean_columns",
                    "Columns",
                    [],
                    multiple=True,
                ),
                ui.input_select("clean_single_column", "Single column", {}),
                ui.panel_conditional(
                    "input.clean_action === 'handle_missing'",
                    ui.input_select(
                        "clean_strategy",
                        "Missing-value strategy",
                        {
                            "drop_rows": "Drop rows",
                            "drop_cols": "Drop columns",
                            "mean": "Fill with mean",
                            "median": "Fill with median",
                            "mode": "Fill with mode",
                            "constant": "Fill with constant",
                        },
                    ),
                    ui.input_text("clean_constant_value", "Constant value", ""),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'scale_columns'",
                    ui.input_select(
                        "clean_scale_method",
                        "Scaling method",
                        {
                            "standard": "Standard",
                            "minmax": "Min-Max",
                            "robust": "Robust",
                        },
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'encode_columns'",
                    ui.input_select(
                        "clean_encode_method",
                        "Encoding method",
                        {"label": "Label encode", "onehot": "One-hot encode"},
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'handle_outliers'",
                    ui.input_select(
                        "clean_outlier_action",
                        "Outlier action",
                        {"remove": "Remove rows", "cap": "Cap values"},
                    ),
                    ui.input_numeric("clean_iqr", "IQR multiplier", 1.5, min=0.5, step=0.5),
                ),
                ui.input_radio_buttons(
                    "clean_save_mode",
                    "Apply mode",
                    {
                        "derived": "Save as derived version",
                        "current": "Apply to current version",
                    },
                    selected="derived",
                    inline=False,
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.tooltip(
                        ui.input_action_button("preview_clean_btn", "Preview",
                                               class_="btn-outline-dark w-100"),
                        "Preview the result without changing the active dataset",
                    ),
                    ui.tooltip(
                        ui.input_action_button("apply_clean_btn", "Apply",
                                               class_="btn-dark w-100"),
                        "Apply transformation and save the result",
                    ),
                    col_widths=[6, 6],
                ),
                ui.output_ui("cleaning_result_summary"),
                width="380px",
            ),
            ui.card(
                ui.card_header(ui.strong("Cleaning Preview")),
                ui.output_data_frame("cleaning_preview_table"),
                ui.download_button("download_cleaned", "Download Cleaned Preview (CSV)",
                                   class_="btn-outline-dark btn-sm mt-2"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Before / After Comparison")),
                output_widget("plot_clean_comparison", height="300px"),
            ),
        ),
    ),
    # ── Feature Engineering Tab ────────────────────────────────────────────
    ui.nav_panel(
        "Feature Engineering",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Feature Engineering", class_="text-uppercase fw-bold"),
                ui.hr(),
                ui.input_select(
                    "feature_method",
                    "Method",
                    {
                        "log": "Log transform",
                        "square": "Square",
                        "cube": "Cube",
                        "interaction": "Interaction",
                        "ratio": "Ratio",
                        "binning": "Binning",
                        "one_hot": "One-hot encoding",
                        "standardize": "Standardize",
                        "normalize": "Normalize",
                        "fillna": "Fill missing values",
                        "dropna": "Drop missing rows",
                    },
                ),
                ui.output_ui("feature_explanation"),
                ui.input_select("feature_col1", "Primary column", {}),
                ui.input_select("feature_col2", "Secondary column", {}),
                ui.input_text("feature_new_column", "New column name (optional)", ""),
                ui.panel_conditional(
                    "input.feature_method === 'binning'",
                    ui.input_numeric("feature_bins", "Number of bins", 4, min=2, step=1),
                    ui.input_checkbox("feature_labels", "Use interval labels", False),
                ),
                ui.panel_conditional(
                    "input.feature_method === 'one_hot'",
                    ui.input_text("feature_prefix", "Dummy prefix (optional)", ""),
                    ui.input_checkbox("feature_drop_first", "Drop first category", False),
                ),
                ui.panel_conditional(
                    "input.feature_method === 'fillna'",
                    ui.input_select(
                        "feature_fill_strategy",
                        "Fill strategy",
                        {
                            "mean": "Mean",
                            "median": "Median",
                            "mode": "Mode",
                            "constant": "Constant",
                        },
                    ),
                    ui.input_text("feature_fill_value", "Constant fill value", ""),
                ),
                ui.input_radio_buttons(
                    "feature_save_mode",
                    "Apply mode",
                    {
                        "derived": "Save as derived version",
                        "current": "Apply to current version",
                    },
                    selected="derived",
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.tooltip(
                        ui.input_action_button("preview_feature_btn", "Preview",
                                               class_="btn-outline-dark w-100"),
                        "Preview the transformation without saving",
                    ),
                    ui.tooltip(
                        ui.input_action_button("apply_feature_btn", "Apply",
                                               class_="btn-dark w-100"),
                        "Apply transformation and save the result",
                    ),
                    col_widths=[6, 6],
                ),
                ui.output_ui("feature_result_summary"),
                width="380px",
            ),
            ui.card(
                ui.card_header(ui.strong("Feature Preview")),
                ui.output_data_frame("feature_preview_table"),
                ui.download_button("download_featured", "Download Feature Preview (CSV)",
                                   class_="btn-outline-dark btn-sm mt-2"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Before / After Comparison")),
                output_widget("plot_feature_comparison", height="300px"),
            ),
        ),
    ),
    # ── EDA Tab ────────────────────────────────────────────────────────────
    ui.nav_panel(
        "EDA",
        # Filtering
        ui.card(
            ui.card_header(ui.strong("Filtering")),
            ui.layout_columns(
                ui.input_text_area(
                    "filter_expr",
                    "Pandas query expression",
                    placeholder='Example: age > 30 and gender == "Female"',
                    rows=2,
                ),
                ui.div(
                    ui.input_radio_buttons(
                        "filter_save_mode",
                        "Filter mode",
                        {
                            "derived": "Save filtered version",
                            "current": "Replace current version",
                        },
                        selected="derived",
                        inline=True,
                    ),
                    ui.input_action_button("apply_filter_btn", "Apply Filter",
                                           class_="btn-dark"),
                ),
                col_widths=[8, 4],
            ),
        ),
        # Summary tables
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Data Preview")),
                ui.input_numeric("head_rows", "Rows", 8, min=1, max=50, step=1),
                ui.output_data_frame("head_table"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Describe")),
                ui.output_data_frame("describe_table"),
                full_screen=True,
            ),
            col_widths=[5, 7],
        ),
        ui.card(
            ui.card_header(ui.strong("Column Types")),
            ui.output_data_frame("column_types_table"),
            full_screen=True,
        ),
        # 1D and 2D plots
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("1D Plot")),
                ui.input_select("plot1d_column", "Column", {}),
                ui.input_numeric("plot1d_bins", "Bins for numeric histogram", 30, min=5, max=100),
                ui.input_checkbox("plot1d_normalize", "Normalize counts", False),
                ui.input_action_button("render_1d_btn", "Render 1D Plot",
                                       class_="btn-dark btn-sm"),
                output_widget("plot_1d", height="380px"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("2D Plot")),
                ui.input_select("plot2d_x", "X column", {}),
                ui.input_select("plot2d_y", "Y column", {}),
                ui.input_select("plot2d_hue", "Hue (optional)", {"": "None"}),
                ui.input_select(
                    "plot2d_kind",
                    "2D plot kind",
                    {
                        "auto": "Auto",
                        "hist": "2D histogram",
                        "scatter": "Scatter",
                        "line": "Line",
                        "bar": "Bar",
                        "box": "Box",
                        "heatmap": "Heatmap",
                    },
                ),
                ui.input_action_button("render_2d_btn", "Render 2D Plot",
                                       class_="btn-dark btn-sm"),
                output_widget("plot_2d", height="380px"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),
        # Regression and Multiline
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Regression")),
                ui.layout_columns(
                    ui.input_select("regression_x", "X column", {}),
                    ui.input_select("regression_y", "Y column", {}),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.input_numeric("regression_order", "Polynomial order", 1, min=1, max=5),
                    ui.div(
                        ui.input_checkbox("regression_logx", "Log-scale x", False),
                        ui.input_checkbox("regression_robust", "Robust fit", False),
                        ui.input_checkbox("regression_lowess", "LOWESS fit", False),
                    ),
                    col_widths=[6, 6],
                ),
                ui.input_action_button("render_regression_btn", "Render Regression",
                                       class_="btn-dark btn-sm"),
                output_widget("plot_regression", height="380px"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Multiline")),
                ui.input_select("multiline_value", "Value column", {}),
                ui.input_select("multiline_group", "Group by", {}),
                ui.input_numeric("multiline_bins", "Histogram bins", 20, min=5, max=80),
                ui.input_checkbox("multiline_normalize", "Normalize counts", False),
                ui.input_action_button("render_multiline_btn", "Render Multiline",
                                       class_="btn-dark btn-sm"),
                output_widget("plot_multiline", height="380px"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),
        # Correlation
        ui.card(
            ui.card_header(ui.strong("Correlation Matrix")),
            ui.layout_columns(
                ui.input_select(
                    "corr_method",
                    "Method",
                    {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"},
                ),
                ui.input_action_button("render_corr_btn", "Render Correlation Matrix",
                                       class_="btn-dark btn-sm"),
                col_widths=[4, 4],
            ),
            output_widget("plot_correlation", height="520px"),
            full_screen=True,
        ),
    ),
    # ── Navbar configuration ───────────────────────────────────────────────
    title=ui.tags.span("STAT 5243 Data Workbench", style="font-weight:800; letter-spacing:0.5px;"),
    id="main_nav",
    theme=shinyswatch.theme.lux,
    fillable=False,
    header=ui.div(
        ui.busy_indicators.use(),
        ui.tags.style(APP_CSS),
        ui.output_ui("message_stack"),
    ),
)


# ---------------------------------------------------------------------------
# Server — ALL logic unchanged, only render.ui presentation updated
# ---------------------------------------------------------------------------
def server(input, output, session):
    datasets_state = reactive.value(OrderedDict())
    active_key_state = reactive.value(None)
    messages_state = reactive.value([])

    cleaning_preview_df = reactive.value(pd.DataFrame())
    cleaning_preview_meta = reactive.value("")
    feature_preview_df = reactive.value(pd.DataFrame())
    feature_preview_meta = reactive.value("")

    plot1d_payload = reactive.value(None)
    plot2d_payload = reactive.value(None)
    regression_payload = reactive.value(None)
    multiline_payload = reactive.value(None)
    corr_payload = reactive.value(None)
    clean_comparison_fig = reactive.value(None)
    feature_comparison_fig = reactive.value(None)

    def push_message(level: str, text: str) -> None:
        items = list(messages_state.get())
        items.insert(0, {"level": level, "text": text})
        messages_state.set(items[:8])

    def clear_previews() -> None:
        cleaning_preview_df.set(pd.DataFrame())
        cleaning_preview_meta.set("")
        feature_preview_df.set(pd.DataFrame())
        feature_preview_meta.set("")

    @reactive.calc
    def current_record() -> dict[str, Any] | None:
        datasets = datasets_state.get()
        active_key = active_key_state.get()
        if active_key is None:
            return None
        return datasets.get(active_key)

    @reactive.calc
    def current_df() -> pd.DataFrame | None:
        record = current_record()
        return None if record is None else record["df"]

    @reactive.calc
    def column_type_frame() -> pd.DataFrame:
        return current_column_types(current_df())

    @reactive.effect
    def _sync_dataset_picker() -> None:
        datasets = datasets_state.get()
        choices = {key: f"{record['label']} ({key})" for key, record in datasets.items()}
        ui.update_select(
            "dataset_picker",
            choices=choices,
            selected=active_key_state.get(),
            session=session,
        )

    @reactive.effect
    @reactive.event(input.dataset_picker)
    def _activate_from_picker() -> None:
        key = input.dataset_picker()
        if key and key in datasets_state.get():
            active_key_state.set(key)
            clear_previews()

    @reactive.effect
    def _sync_column_inputs() -> None:
        df = current_df()
        if df is None:
            empty_choices: dict[str, str] = {}
            ui.update_selectize("clean_columns", choices=empty_choices, selected=[], session=session)
            for widget in [
                "clean_single_column",
                "feature_col1",
                "feature_col2",
                "plot1d_column",
                "plot2d_x",
                "plot2d_y",
                "regression_x",
                "regression_y",
                "multiline_value",
                "multiline_group",
            ]:
                ui.update_select(widget, choices=empty_choices, session=session)
            ui.update_select("plot2d_hue", choices={"": "None"}, selected="", session=session)
            return

        all_cols = [str(col) for col in df.columns]
        numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in all_cols if not pd.api.types.is_numeric_dtype(df[col])]

        ui.update_selectize("clean_columns", choices={col: col for col in all_cols}, selected=[], session=session)
        ui.update_select("clean_single_column", choices={col: col for col in numeric_cols or all_cols}, session=session)
        ui.update_select("feature_col1", choices={col: col for col in all_cols}, session=session)
        ui.update_select("feature_col2", choices={"": "None", **{col: col for col in all_cols}}, selected="", session=session)
        ui.update_select("plot1d_column", choices={col: col for col in all_cols}, session=session)
        ui.update_select("plot2d_x", choices={col: col for col in all_cols}, session=session)
        ui.update_select("plot2d_y", choices={col: col for col in all_cols}, session=session)
        ui.update_select("plot2d_hue", choices={"": "None", **{col: col for col in all_cols}}, selected="", session=session)
        ui.update_select("regression_x", choices={col: col for col in numeric_cols}, session=session)
        ui.update_select("regression_y", choices={col: col for col in numeric_cols}, session=session)
        ui.update_select("multiline_value", choices={col: col for col in numeric_cols}, session=session)
        ui.update_select("multiline_group", choices={col: col for col in categorical_cols}, session=session)

    @reactive.effect
    @reactive.event(input.load_builtin_btn)
    def _load_builtin() -> None:
        name = input.builtin_dataset()
        try:
            df = load_builtin_dataset(name)
            prefix = "original" if not datasets_state.get() else "loaded"
            datasets, key = register_dataset_version(
                datasets_state.get(),
                df,
                prefix=prefix,
                label=f"Built-in: {name}",
                transform="built-in load",
            )
            datasets_state.set(datasets)
            active_key_state.set(key)
            clear_previews()
            push_message("success", f"Loaded built-in dataset '{name}' as {key}.")
        except Exception as exc:
            push_message("error", f"Failed to load built-in dataset: {exc}")

    @reactive.effect
    @reactive.event(input.load_upload_btn)
    def _load_uploaded() -> None:
        files = input.upload_file()
        if not files:
            push_message("warning", "Choose a file before loading.")
            return
        info = files[0]
        try:
            df = load_uploaded_dataset(info["datapath"], info["name"])
            prefix = "original" if not datasets_state.get() else "loaded"
            datasets, key = register_dataset_version(
                datasets_state.get(),
                df,
                prefix=prefix,
                label=f"Upload: {info['name']}",
                transform=f"uploaded {Path(info['name']).suffix.lower()}",
            )
            datasets_state.set(datasets)
            active_key_state.set(key)
            clear_previews()
            push_message("success", f"Loaded uploaded file '{info['name']}' as {key}.")
        except Exception as exc:
            push_message("error", f"Failed to load uploaded file: {exc}")

    def compute_cleaning_result() -> tuple[pd.DataFrame, str]:
        df = current_df()
        if df is None:
            raise ValueError("Load a dataset first.")

        action = input.clean_action()
        if action == "handle_missing":
            transformed = cleaning.handle_missing(
                df,
                columns=list(input.clean_columns() or []),
                strategy=input.clean_strategy(),
                constant_value=coerce_text_value(input.clean_constant_value()),
            )
        elif action == "remove_duplicates":
            transformed = cleaning.remove_duplicates(df)
        elif action == "scale_columns":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more numeric columns to scale.")
            transformed = cleaning.scale_columns(df, columns=columns, method=input.clean_scale_method())
        elif action == "encode_columns":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more categorical columns to encode.")
            transformed = cleaning.encode_columns(df, columns=columns, method=input.clean_encode_method())
        elif action == "handle_outliers":
            column = input.clean_single_column()
            diagnostics = cleaning.detect_outliers(
                df,
                column=column,
                iqr_multiplier=float(input.clean_iqr()),
            )
            transformed = cleaning.handle_outliers(
                df,
                column=column,
                action=input.clean_outlier_action(),
                iqr_multiplier=float(input.clean_iqr()),
            )
            return transformed, (
                f"Outlier handling on {column}: {diagnostics['n_outliers']} outliers "
                f"identified with IQR multiplier {input.clean_iqr()}."
            )
        else:
            raise ValueError(f"Unsupported cleaning action: {action}")

        return transformed, f"Cleaning action '{action}' produced shape {transformed.shape}."

    def apply_transformed_result(
        transformed: pd.DataFrame,
        *,
        mode: str,
        prefix: str,
        label: str,
        transform: str,
    ) -> str:
        datasets = datasets_state.get()
        active_key = active_key_state.get()
        if active_key is None:
            raise ValueError("No active dataset to update.")
        if mode == "current":
            datasets_state.set(overwrite_dataset_version(datasets, active_key, transformed, transform=transform))
            active_key_state.set(active_key)
            return active_key
        datasets, key = register_dataset_version(
            datasets,
            transformed,
            prefix=prefix,
            label=label,
            source_key=active_key,
            transform=transform,
        )
        datasets_state.set(datasets)
        active_key_state.set(key)
        return key

    @reactive.effect
    @reactive.event(input.preview_clean_btn)
    def _preview_cleaning() -> None:
        try:
            transformed, summary = compute_cleaning_result()
            cleaning_preview_df.set(transformed.head(20))
            cleaning_preview_meta.set(summary)
            # Build before/after comparison chart
            df = current_df()
            action = input.clean_action()

            # Detect row-removal operations — show row count comparison
            is_row_removal = (
                action == "remove_duplicates"
                or (action == "handle_missing" and input.clean_strategy() in ("drop_rows", "drop_cols"))
                or (action == "handle_outliers" and input.clean_outlier_action() == "remove")
            )

            if is_row_removal and df is not None:
                clean_comparison_fig.set(
                    build_rowcount_figure(len(df), len(transformed), action.replace("_", " ").title())
                )
            else:
                # Value-changing operation — show distribution comparison
                if action == "handle_outliers":
                    col = input.clean_single_column()
                else:
                    cols = list(input.clean_columns() or [])
                    col = cols[0] if cols else None
                if col and df is not None and col in df.columns and col in transformed.columns:
                    clean_comparison_fig.set(build_comparison_figure(df[col], transformed[col], col))
                else:
                    clean_comparison_fig.set(None)
            push_message("info", "Cleaning preview updated.")
        except Exception as exc:
            push_message("error", f"Cleaning preview failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_clean_btn)
    def _apply_cleaning() -> None:
        try:
            transformed, summary = compute_cleaning_result()
            target_key = apply_transformed_result(
                transformed,
                mode=input.clean_save_mode(),
                prefix="cleaned",
                label=f"Cleaned from {active_key_state.get()}",
                transform=summary,
            )
            cleaning_preview_df.set(transformed.head(20))
            cleaning_preview_meta.set(summary)
            push_message("success", f"Cleaning applied to {target_key}.")
        except Exception as exc:
            push_message("error", f"Cleaning apply failed: {exc}")

    def compute_feature_result() -> tuple[pd.DataFrame, str, dict]:
        df = current_df()
        if df is None:
            raise ValueError("Load a dataset first.")

        method = input.feature_method()
        col2 = input.feature_col2() or None
        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            df,
            method,
            input.feature_col1() or None,
            col2=col2,
            bins=int(input.feature_bins()),
            new_column=input.feature_new_column() or None,
            labels=bool(input.feature_labels()),
            prefix=input.feature_prefix() or None,
            drop_first=bool(input.feature_drop_first()),
            strategy=input.feature_fill_strategy(),
            fill_value=coerce_text_value(input.feature_fill_value()),
        )
        return transformed, f"{meta['feature_type']} created/updated columns: {meta['output_columns']}", meta

    @reactive.effect
    @reactive.event(input.preview_feature_btn)
    def _preview_feature() -> None:
        try:
            transformed, summary, meta = compute_feature_result()
            feature_preview_df.set(transformed.head(20))
            feature_preview_meta.set(summary)
            # Build before/after comparison chart using correct columns
            df = current_df()
            input_col = (meta.get("input_columns") or [None])[0]
            output_cols = meta.get("output_columns", [])
            out_col = output_cols[0] if output_cols else None
            if (input_col and out_col and df is not None
                    and input_col in df.columns and out_col in transformed.columns):
                feature_comparison_fig.set(
                    build_comparison_figure(
                        df[input_col], transformed[out_col],
                        f"{input_col} -> {out_col}",
                    )
                )
            else:
                feature_comparison_fig.set(None)
            push_message("info", "Feature-engineering preview updated.")
        except Exception as exc:
            push_message("error", f"Feature preview failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_feature_btn)
    def _apply_feature() -> None:
        try:
            transformed, summary, _meta = compute_feature_result()
            target_key = apply_transformed_result(
                transformed,
                mode=input.feature_save_mode(),
                prefix="feature",
                label=f"Feature engineered from {active_key_state.get()}",
                transform=summary,
            )
            feature_preview_df.set(transformed.head(20))
            feature_preview_meta.set(summary)
            push_message("success", f"Feature engineering applied to {target_key}.")
        except Exception as exc:
            push_message("error", f"Feature apply failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_filter_btn)
    def _apply_filter() -> None:
        df = current_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        expr = input.filter_expr().strip()
        if not expr:
            push_message("warning", "Enter a pandas query expression first.")
            return
        try:
            filtered = EDA.apply_filter(df, expr)
            target_key = apply_transformed_result(
                filtered,
                mode=input.filter_save_mode(),
                prefix="filtered",
                label=f"Filtered from {active_key_state.get()}",
                transform=f"filter: {expr}",
            )
            push_message(
                "success",
                f"Filter applied to {target_key}. Rows: {len(df)} -> {len(filtered)}.",
            )
        except Exception as exc:
            push_message("error", f"Filter failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_1d_btn)
    def _render_1d() -> None:
        df = current_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        column = input.plot1d_column()
        if not column:
            push_message("warning", "Choose a column for the 1D plot.")
            return
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                payload = EDA.plot_numeric_1d(
                    df,
                    column=column,
                    bins=int(input.plot1d_bins()),
                    normalize=bool(input.plot1d_normalize()),
                )
            else:
                payload = EDA.plot_categorical_1d(
                    df,
                    column=column,
                    normalize=bool(input.plot1d_normalize()),
                )
            plot1d_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "1D plot rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "1D plot failed."))
            else:
                push_message("success", "1D plot rendered.")
                col_data = df[column].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    stats_msg = (f"Stats for {column}: mean={col_data.mean():.3f}, "
                                 f"median={col_data.median():.3f}, std={col_data.std():.3f}, "
                                 f"skew={col_data.skew():.3f}")
                    push_message("info", stats_msg)
        except Exception as exc:
            push_message("error", f"1D plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_2d_btn)
    def _render_2d() -> None:
        df = current_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            kind = input.plot2d_kind()
            payload = EDA.plot_two_columns(
                df,
                x=input.plot2d_x(),
                y=input.plot2d_y(),
                hue=input.plot2d_hue() or None,
                kind=None if kind == "auto" else kind,
            )
            plot2d_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "2D plot rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "2D plot failed."))
            else:
                push_message("success", "2D plot rendered.")
        except Exception as exc:
            push_message("error", f"2D plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_regression_btn)
    def _render_regression() -> None:
        df = current_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.regression_analysis(
                df,
                x=input.regression_x(),
                y=input.regression_y(),
                order=int(input.regression_order()),
                logx=bool(input.regression_logx()),
                robust=bool(input.regression_robust()),
                lowess=bool(input.regression_lowess()),
            )
            regression_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "Regression rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "Regression failed."))
            else:
                push_message("success", "Regression rendered.")
                # Add p-value info
                from scipy.stats import pearsonr
                x_data = df[input.regression_x()].dropna()
                y_data = df[input.regression_y()].dropna()
                common = df[[input.regression_x(), input.regression_y()]].dropna()
                if len(common) > 2:
                    r, p = pearsonr(common.iloc[:, 0], common.iloc[:, 1])
                    push_message("info", f"Pearson r = {r:.4f}, p-value = {p:.2e} (n={len(common)})")
        except Exception as exc:
            push_message("error", f"Regression failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_multiline_btn)
    def _render_multiline() -> None:
        df = current_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.plot_multiline(
                df,
                column=input.multiline_value(),
                group_by=input.multiline_group() or None,
                normalize=bool(input.multiline_normalize()),
                nbins=int(input.multiline_bins()),
            )
            multiline_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "Multiline rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "Multiline failed."))
            else:
                push_message("success", "Multiline plot rendered.")
        except Exception as exc:
            push_message("error", f"Multiline plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_corr_btn)
    def _render_correlation() -> None:
        df = current_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.correlation_matrix(df, method=input.corr_method())
            corr_payload.set(payload)
            if payload.get("status") == "error":
                push_message("error", payload.get("message", "Correlation matrix failed."))
            else:
                push_message("success", "Correlation matrix rendered.")
        except Exception as exc:
            push_message("error", f"Correlation matrix failed: {exc}")

    @render.download(filename="active_dataset.csv")
    def download_active():
        df = current_df()
        if df is not None:
            yield df.to_csv(index=False)

    @render.download(filename="cleaned_preview.csv")
    def download_cleaned():
        df = cleaning_preview_df.get()
        if not df.empty:
            yield df.to_csv(index=False)

    @render.download(filename="feature_preview.csv")
    def download_featured():
        df = feature_preview_df.get()
        if not df.empty:
            yield df.to_csv(index=False)

    # ── Render outputs ─────────────────────────────────────────────────────

    @output
    @render.ui
    def message_stack():
        messages = messages_state.get()
        if not messages:
            return ui.div()
        _level_map = {
            "info": "alert-info",
            "success": "alert-success",
            "warning": "alert-warning",
            "error": "alert-danger",
        }
        return ui.div(
            {"class": "alert-stack", "style": "padding: 0 12px;"},
            *[
                ui.div(
                    {"class": f"alert {_level_map.get(item['level'], 'alert-secondary')} py-2 mb-1",
                     "role": "alert"},
                    item["text"],
                )
                for item in messages
            ],
        )

    @output
    @render.ui
    def active_dataset_summary():
        df = current_df()
        record = current_record()
        if df is None or record is None:
            return ui.div({"class": "small-note"}, "Load a dataset to begin.")
        overview = current_overview(df)
        return ui.div(
            ui.p(ui.strong("Label: "), record["label"]),
            ui.p(ui.strong("Key: "), active_key_state.get()),
            ui.div(
                {"class": "metric-grid"},
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Rows"), ui.div({"class": "value"}, str(overview["n_rows"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Columns"), ui.div({"class": "value"}, str(overview["n_cols"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Missing"), ui.div({"class": "value"}, str(overview["n_missing"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Duplicates"), ui.div({"class": "value"}, str(overview["n_duplicates"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Source"), ui.div({"class": "value"}, record["source_key"] or "-")),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Transform"), ui.div({"class": "value"}, record["transform"] or "-")),
            ),
        )

    @output
    @render.data_frame
    def history_table():
        history = format_history_table(datasets_state.get())
        return render.DataGrid(history)

    @output
    @render.data_frame
    def head_table():
        df = current_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        payload = EDA.show_head(df, n=int(input.head_rows()))
        return render.DataGrid(dataframe_from_payload(payload))

    @output
    @render.data_frame
    def describe_table():
        df = current_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        payload = EDA.describe_dataframe(df)
        return render.DataGrid(dataframe_from_payload(payload))

    @output
    @render.data_frame
    def column_types_table():
        return render.DataGrid(column_type_frame())

    @output
    @render.ui
    def cleaning_result_summary():
        text = cleaning_preview_meta.get()
        return ui.div({"class": "small-note"}, text or "Preview a cleaning action to see a summary.")

    @output
    @render.data_frame
    def cleaning_preview_table():
        return render.DataGrid(cleaning_preview_df.get())

    @output
    @render.ui
    def feature_result_summary():
        text = feature_preview_meta.get()
        return ui.div({"class": "small-note"}, text or "Preview a feature step to see a summary.")

    @output
    @render.ui
    def feature_explanation():
        method = input.feature_method()
        explanations = {
            "log": "Applies log(1+x). Reduces right-skew and compresses large values.",
            "square": "Squares values (x^2). Amplifies differences between large and small values.",
            "cube": "Cubes values (x^3). Captures cubic relationships, preserves sign.",
            "interaction": "Multiplies two columns (x * y). Captures combined/synergistic effects.",
            "ratio": "Divides col1 by col2. Useful for per-unit metrics (e.g., price per sqft).",
            "binning": "Groups continuous values into discrete bins using equal-width intervals.",
            "one_hot": "Creates binary 0/1 columns for each category. Required by most ML models.",
            "standardize": "Z-score normalization: (x - mean) / std. Centers data at 0 with unit variance.",
            "normalize": "Min-Max scaling to [0, 1]. Preserves shape, bounds values.",
            "fillna": "Replaces missing values with a computed or constant value.",
            "dropna": "Removes rows containing missing values in the selected column.",
        }
        text = explanations.get(method, "")
        if not text:
            return ui.div()
        return ui.div(
            {"class": "tip-box", "style": "margin-top: 8px;"},
            ui.tags.small(text),
        )

    @output
    @render.data_frame
    def feature_preview_table():
        return render.DataGrid(feature_preview_df.get())

    @output
    @render_plotly
    def plot_1d():
        payload = plot1d_payload.get()
        if payload is None:
            return empty_figure("Render a 1D plot to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_2d():
        payload = plot2d_payload.get()
        if payload is None:
            return empty_figure("Render a 2D plot to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_regression():
        payload = regression_payload.get()
        if payload is None:
            return empty_figure("Render a regression plot to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_multiline():
        payload = multiline_payload.get()
        if payload is None:
            return empty_figure("Render a multiline plot to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_correlation():
        payload = corr_payload.get()
        if payload is None:
            return empty_figure("Render a correlation matrix to see output here.")
        return figure_from_payload(payload)

    @output
    @render_plotly
    def plot_clean_comparison():
        fig = clean_comparison_fig.get()
        return fig if fig else empty_figure("Preview a cleaning action to see comparison.")

    @output
    @render_plotly
    def plot_feature_comparison():
        fig = feature_comparison_fig.get()
        return fig if fig else empty_figure("Preview a feature to see comparison.")


app = App(app_ui, server)
