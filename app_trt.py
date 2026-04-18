from __future__ import annotations

"""
STAT 5243 Project 2 — Interactive Data Workbench (Shiny for Python).
"""

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import random
import re
import uuid
from urllib.parse import parse_qs

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris as sklearn_load_iris
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly
import shinyswatch

import eda as EDA
import feature_engineering
import data_cleaning as cleaning


# ---------------------------------------------------------------------------
# Plotly global template
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
AB_LOG_PATH = BASE_DIR / "ab_test_events.csv"
# Team-only gate for downloading the event log. Not security-critical; just
# keeps random visitors from grabbing the file.
AB_ADMIN_PASSWORD = "team21-cleaning-ab"

# Dramatic visual makeover applied only to the Cleaning tab for Group B.
# The experimental intervention is still the guided-cleaning UX; this CSS
# is the *full* treatment package — dark gradient theme, glass-morphism
# cards, pill-shaped gradient buttons, custom typography. Applied as an
# injected <style> tag rendered only for sessions with ab_group == "B".
GROUP_B_DRAMATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

#cleaning-tab-content {
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 45%, #581c87 100%);
  color: #f5f3ff;
  padding: 28px;
  border-radius: 20px;
  margin-top: 8px;
  min-height: 85vh;
  font-family: 'Space Grotesk', 'Inter', system-ui, -apple-system, sans-serif;
  box-shadow: 0 20px 60px rgba(15, 23, 42, 0.45);
  border: 1px solid rgba(168, 85, 247, 0.25);
}
#cleaning-tab-content .bslib-sidebar-layout,
#cleaning-tab-content .bslib-sidebar-layout > * {
  background: transparent !important;
}
#cleaning-tab-content aside,
#cleaning-tab-content .sidebar,
#cleaning-tab-content [class*="sidebar"] > div:first-child {
  background: rgba(255, 255, 255, 0.04) !important;
  backdrop-filter: blur(16px);
  border: 1px solid rgba(168, 85, 247, 0.2) !important;
  border-radius: 16px !important;
  padding: 18px !important;
}
#cleaning-tab-content .card {
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(168, 85, 247, 0.22) !important;
  color: #f5f3ff !important;
  border-radius: 14px !important;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
  overflow: hidden;
}
#cleaning-tab-content .card-header {
  background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%) !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  font-size: 1.02rem !important;
  letter-spacing: 0.04em !important;
  padding: 14px 20px !important;
  border-bottom: none !important;
  text-transform: uppercase;
}
#cleaning-tab-content h6 {
  color: #fbbf24 !important;
  font-weight: 900 !important;
  letter-spacing: 0.1em !important;
  font-size: 0.88rem !important;
  text-shadow: 0 0 12px rgba(251, 191, 36, 0.25);
}
#cleaning-tab-content label,
#cleaning-tab-content .form-label,
#cleaning-tab-content .control-label {
  color: #e0e7ff !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.03em !important;
  margin-bottom: 6px !important;
}
#cleaning-tab-content select,
#cleaning-tab-content .form-select,
#cleaning-tab-content .form-control,
#cleaning-tab-content input[type="text"],
#cleaning-tab-content input[type="number"],
#cleaning-tab-content input[type="password"],
#cleaning-tab-content textarea,
#cleaning-tab-content .selectize-input {
  background: rgba(15, 23, 42, 0.6) !important;
  color: #f5f3ff !important;
  border: 1.5px solid rgba(168, 85, 247, 0.4) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
}
#cleaning-tab-content select:focus,
#cleaning-tab-content .form-select:focus,
#cleaning-tab-content .form-control:focus,
#cleaning-tab-content input:focus,
#cleaning-tab-content textarea:focus,
#cleaning-tab-content .selectize-input.focus {
  background: rgba(15, 23, 42, 0.9) !important;
  border-color: #f97316 !important;
  box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.25) !important;
  outline: none !important;
}
#cleaning-tab-content .selectize-dropdown,
#cleaning-tab-content .selectize-dropdown-content {
  background: #1e1b4b !important;
  color: #f5f3ff !important;
  border: 1px solid rgba(168, 85, 247, 0.4) !important;
}
#cleaning-tab-content .selectize-dropdown .option:hover,
#cleaning-tab-content .selectize-dropdown .active {
  background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
  color: #ffffff !important;
}
#cleaning-tab-content .selectize-input > .item,
#cleaning-tab-content .selectize-input > div[data-value] {
  background: linear-gradient(135deg, #f97316, #ec4899) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border-radius: 999px !important;
  padding: 2px 10px !important;
}
#cleaning-tab-content .radio label,
#cleaning-tab-content .checkbox label,
#cleaning-tab-content .form-check-label {
  color: #e0e7ff !important;
  font-weight: 500 !important;
}
#cleaning-tab-content .form-check-input {
  background-color: rgba(15, 23, 42, 0.6) !important;
  border: 2px solid rgba(168, 85, 247, 0.6) !important;
}
#cleaning-tab-content .form-check-input:checked {
  background-color: #f97316 !important;
  border-color: #f97316 !important;
}
#cleaning-tab-content .btn {
  border-radius: 999px !important;
  font-weight: 700 !important;
  padding: 11px 26px !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  font-size: 0.82rem !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  border: none !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
}
#cleaning-tab-content .btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.4) !important;
}
#cleaning-tab-content .btn-primary,
#cleaning-tab-content .btn-dark {
  background: linear-gradient(135deg, #f97316 0%, #ec4899 100%) !important;
  color: #ffffff !important;
}
#cleaning-tab-content .btn-primary:hover,
#cleaning-tab-content .btn-dark:hover {
  background: linear-gradient(135deg, #fb923c 0%, #f472b6 100%) !important;
  box-shadow: 0 10px 28px rgba(236, 72, 153, 0.5) !important;
}
#cleaning-tab-content .btn-success {
  background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%) !important;
  color: #ffffff !important;
}
#cleaning-tab-content .btn-success:hover {
  background: linear-gradient(135deg, #14b8a6 0%, #0ea5e9 100%) !important;
  box-shadow: 0 10px 28px rgba(6, 182, 212, 0.55) !important;
}
#cleaning-tab-content .btn-outline-dark {
  background: transparent !important;
  color: #fbbf24 !important;
  border: 2px solid #fbbf24 !important;
  box-shadow: 0 0 0 transparent !important;
}
#cleaning-tab-content .btn-outline-dark:hover {
  background: rgba(251, 191, 36, 0.12) !important;
  color: #fef3c7 !important;
  box-shadow: 0 0 22px rgba(251, 191, 36, 0.45) !important;
}
#cleaning-tab-content hr {
  border-color: rgba(168, 85, 247, 0.3) !important;
  margin: 16px 0 !important;
}
#cleaning-tab-content .clean-step-grid {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
  gap: 16px !important;
}
#cleaning-tab-content .clean-step-card {
  background: linear-gradient(135deg, rgba(249, 115, 22, 0.18), rgba(236, 72, 153, 0.18)) !important;
  border: 1px solid rgba(251, 191, 36, 0.35) !important;
  color: #fef3c7 !important;
  padding: 18px !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.3) !important;
  min-height: 140px !important;
  transition: transform 0.2s ease !important;
}
#cleaning-tab-content .clean-step-card:hover {
  transform: translateY(-3px);
}
#cleaning-tab-content .clean-step-card .step-label {
  color: #fb923c !important;
  font-weight: 900 !important;
  letter-spacing: 0.12em !important;
  text-shadow: 0 0 14px rgba(251, 146, 60, 0.55);
}
#cleaning-tab-content .clean-step-card .step-title {
  color: #fef3c7 !important;
  font-weight: 800 !important;
  font-size: 1.1rem !important;
}
#cleaning-tab-content .clean-cta-box {
  background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%) !important;
  border: none !important;
  padding: 22px !important;
  border-radius: 16px !important;
  box-shadow: 0 14px 44px rgba(124, 58, 237, 0.55) !important;
  margin-top: 14px !important;
}
#cleaning-tab-content .clean-cta-title {
  color: #ffffff !important;
  font-size: 1.28rem !important;
  font-weight: 900 !important;
  letter-spacing: 0.02em !important;
  margin-bottom: 6px !important;
}
#cleaning-tab-content .clean-cta-note {
  color: rgba(255, 255, 255, 0.92) !important;
  font-size: 0.94rem !important;
  margin-bottom: 16px !important;
}
#cleaning-tab-content .clean-version-panel {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(6, 182, 212, 0.18)) !important;
  border: 1px solid rgba(16, 185, 129, 0.35) !important;
  color: #d1fae5 !important;
  border-radius: 12px !important;
  padding: 14px 16px !important;
}
#cleaning-tab-content .instr-box,
#cleaning-tab-content .tip-box {
  background: linear-gradient(135deg, rgba(251, 146, 60, 0.15), rgba(251, 191, 36, 0.1)) !important;
  border-left: 4px solid #fb923c !important;
  color: #fef3c7 !important;
  border-radius: 10px !important;
  padding: 14px 18px !important;
  font-size: 0.95rem !important;
}
#cleaning-tab-content .small-note {
  color: #e0e7ff !important;
  font-size: 0.9rem !important;
}
#cleaning-tab-content table {
  color: #f5f3ff !important;
  background: transparent !important;
}
#cleaning-tab-content thead th,
#cleaning-tab-content table thead tr th {
  background: linear-gradient(90deg, rgba(124, 58, 237, 0.4), rgba(236, 72, 153, 0.4)) !important;
  color: #fbbf24 !important;
  font-weight: 800 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  font-size: 0.75rem !important;
  padding: 12px !important;
  border: none !important;
}
#cleaning-tab-content tbody td,
#cleaning-tab-content table tbody tr td {
  border-top: 1px solid rgba(168, 85, 247, 0.18) !important;
  padding: 10px 12px !important;
  color: #e0e7ff !important;
  background: transparent !important;
}
#cleaning-tab-content tbody tr:hover,
#cleaning-tab-content table tbody tr:hover {
  background: rgba(124, 58, 237, 0.2) !important;
}
#cleaning-tab-content .datatable,
#cleaning-tab-content .shiny-output-frame {
  background: rgba(15, 23, 42, 0.3) !important;
  border-radius: 10px !important;
  color: #f5f3ff !important;
}
#cleaning-tab-content p,
#cleaning-tab-content div,
#cleaning-tab-content span {
  color: inherit;
}
#cleaning-tab-content .card-body p,
#cleaning-tab-content .card-body span,
#cleaning-tab-content .card-body div {
  color: #f5f3ff;
}

/* Dramatic top banner (rendered only for Group B, above the tab body) */
.cleaning-b-banner {
  background: linear-gradient(135deg, #1e1b4b 0%, #7c3aed 45%, #f97316 100%);
  color: #ffffff;
  padding: 30px 36px;
  border-radius: 20px;
  margin: 12px 0 18px 0;
  box-shadow: 0 20px 60px rgba(124, 58, 237, 0.32);
  position: relative;
  overflow: hidden;
  font-family: 'Space Grotesk', 'Inter', system-ui, -apple-system, sans-serif;
}
.cleaning-b-banner::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.12), transparent 60%),
              radial-gradient(circle at 80% 70%, rgba(251, 191, 36, 0.18), transparent 55%);
  pointer-events: none;
}
.cleaning-b-banner-inner { position: relative; z-index: 1; }
.cleaning-b-banner-title {
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  margin-bottom: 6px;
  text-shadow: 0 2px 14px rgba(0, 0, 0, 0.3);
}
.cleaning-b-banner-subtitle {
  font-size: 1rem;
  opacity: 0.92;
  margin-bottom: 22px;
  font-weight: 500;
}
.cleaning-b-banner-steps {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.cb-step {
  display: flex;
  align-items: center;
  gap: 10px;
  background: rgba(255, 255, 255, 0.16);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.28);
  border-radius: 999px;
  padding: 7px 16px 7px 8px;
  font-weight: 700;
}
.cb-step-num {
  background: rgba(255, 255, 255, 0.35);
  border-radius: 999px;
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  font-size: 0.95rem;
  flex-shrink: 0;
}
.cb-step-label { font-size: 0.9rem; letter-spacing: 0.02em; }
.cb-step-arrow { color: rgba(255, 255, 255, 0.6); font-weight: 700; font-size: 1.1rem; }
"""


# ---------------------------------------------------------------------------
# Pure helper functions
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
        rows.append({
            "key": key,
            "label": record["label"],
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "source": record["source_key"] or "-",
            "transform": record["transform"] or "-",
            "created_at": record["created_at"],
        })
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
# Descriptive key generation
# ---------------------------------------------------------------------------
def _sanitize_col(col: str, max_len: int = 8) -> str:
    """Shorten a column name to a filesystem-safe token."""
    s = re.sub(r"[^a-zA-Z0-9]", "_", str(col))
    return s[:max_len]


def _cols_token(cols: list[str] | str | None, max_cols: int = 2) -> str:
    if not cols:
        return ""
    if isinstance(cols, str):
        cols = [cols]
    return "_".join(_sanitize_col(c) for c in list(cols)[:max_cols])


def _filter_expr_to_slug(expr: str) -> str:
    """Convert a filter expression to a compact slug for dataset naming.

    Examples
    --------
    ``age >= 5``               → ``age_geq_5``
    ``type == "race"``         → ``type_eq_race``
    ``(age >= 5) & (x != 0)`` → ``age_geq_5_x_neq_0``
    """
    s = expr
    # Replace multi-char operators before single-char ones
    for op, abbr in [(">=", "_geq_"), ("<=", "_leq_"), ("!=", "_neq_"),
                     ("==", "_eq_"), (">", "_gt_"), ("<", "_lt_")]:
        s = s.replace(op, abbr)
    # Strip quotes and parentheses
    s = re.sub(r'["\']', '', s)
    s = re.sub(r'[()&|~]', '_', s)
    # Collapse whitespace → underscore
    s = re.sub(r'\s+', '_', s.strip())
    # Remove any characters that aren't alphanumeric or underscore
    s = re.sub(r'[^a-zA-Z0-9_]', '', s)
    # Collapse repeated underscores and strip leading/trailing ones
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:35]


def generate_descriptive_key(
    action: str,
    columns: list[str] | str | None = None,
    method: str | None = None,
    expr: str | None = None,
) -> str:
    """Return a short but meaningful dataset key prefix encoding the operation."""
    cc = _cols_token(columns)

    if action == "handle_missing":
        smap = {
            "knn": "knn", "drop_rows": "dropr", "drop_cols": "dropc",
            "mean": "fill_mn", "median": "fill_med", "mode": "fill_mo",
            "constant": "fill_c",
        }
        pfx = smap.get(method or "", "miss")
        raw = f"{pfx}_{cc}" if cc else pfx

    elif action == "remove_duplicates":
        raw = "dedup"

    elif action == "scale_columns":
        mmap = {"standard": "std", "minmax": "mm", "robust": "rob"}
        abbr = mmap.get(method or "", method or "scl")
        raw = f"scl_{abbr}_{cc}" if cc else f"scl_{abbr}"

    elif action == "encode_columns":
        mmap = {"label": "lbl", "onehot": "ohe"}
        abbr = mmap.get(method or "", method or "enc")
        raw = f"enc_{abbr}_{cc}" if cc else f"enc_{abbr}"

    elif action == "handle_outliers":
        mmap = {"remove": "rm", "cap": "cap"}
        abbr = mmap.get(method or "", method or "out")
        raw = f"out_{abbr}_{cc}" if cc else f"out_{abbr}"

    elif action == "standardize_text":
        mmap = {"lower": "lo", "upper": "up", "title": "ti", "none": ""}
        abbr = mmap.get(method or "", "")
        raw = f"txt_{abbr}_{cc}".rstrip("_") if cc else f"txt_{abbr}".rstrip("_")

    elif action == "coerce_types":
        mmap = {"numeric": "num", "string": "str"}
        abbr = mmap.get(method or "", method or "crc")
        raw = f"coerce_{abbr}_{cc}" if cc else f"coerce_{abbr}"

    elif action in {
        "log", "square", "cube", "interaction", "ratio",
        "binning", "one_hot", "standardize", "normalize",
        "fillna", "dropna",
    }:
        fe_abbr = {
            "log": "log", "square": "sq", "cube": "cu",
            "interaction": "int", "ratio": "rat", "binning": "bin",
            "one_hot": "ohe", "standardize": "zscore", "normalize": "norm",
            "fillna": "fill", "dropna": "dropna",
        }
        pfx = fe_abbr.get(action, action)
        raw = f"{pfx}_{cc}" if cc else pfx

    elif action == "custom_expr":
        raw = f"expr_{cc}" if cc else "expr"

    elif action == "filter":
        slug = _filter_expr_to_slug(expr) if expr else ""
        raw = f"filt_{slug}" if slug else "filt"

    else:
        raw = re.sub(r"[^a-zA-Z0-9_]", "_", action)[:20]

    # Truncate to 40 chars to keep keys readable
    return raw[:40]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def empty_figure(title: str = "No plot yet.") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title)
    return fig


def build_comparison_figure(before: pd.Series, after: pd.Series, col_name: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Before", "After"])
    if pd.api.types.is_numeric_dtype(before):
        fig.add_histogram(x=before.dropna(), name="Before", marker_color="#94a3b8", row=1, col=1)
        fig.add_histogram(x=after.dropna(), name="After", marker_color="#4361ee", row=1, col=2)
    else:
        vc_before = before.value_counts().head(15)
        vc_after = after.value_counts().head(15)
        fig.add_bar(x=vc_before.index.astype(str), y=vc_before.values, name="Before",
                    marker_color="#94a3b8", row=1, col=1)
        fig.add_bar(x=vc_after.index.astype(str), y=vc_after.values, name="After",
                    marker_color="#4361ee", row=1, col=2)
    fig.update_layout(title=f"Before / After: {col_name}", showlegend=False,
                      height=280, margin=dict(t=50, b=30, l=40, r=20))
    return fig


def build_rowcount_figure(before_count: int, after_count: int, action: str) -> go.Figure:
    fig = go.Figure()
    removed = before_count - after_count
    fig.add_bar(
        x=["Before", "After"], y=[before_count, after_count],
        marker_color=["#94a3b8", "#4361ee"],
        text=[f"{before_count:,}", f"{after_count:,}"],
        textposition="outside", width=0.5,
    )
    fig.update_layout(
        title=f"{action}: {removed:,} rows removed ({before_count:,} → {after_count:,})",
        yaxis_title="Row count", height=280, margin=dict(t=50, b=30, l=60, r=20),
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
        fig.add_bar(x=midpoints(data["bins"]), y=data["counts"], width=widths(data["bins"]),
                    marker_color="#1a1a2e")
        fig.update_layout(xaxis_title=data["column"], yaxis_title="Count")

    elif plot_type == "scatter":
        points = pd.DataFrame(data["points"])
        if points.empty:
            return empty_figure("No scatter data available.")
        if data.get("hue"):
            for label, group in points.groupby(data["hue"]):
                fig.add_scattergl(x=group[data["x"]], y=group[data["y"]], mode="markers",
                                  name=str(label), opacity=0.6)
        else:
            fig.add_scattergl(x=points[data["x"]], y=points[data["y"]], mode="markers",
                              name=f"{data['x']} vs {data['y']}", opacity=0.6)
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "hist2d":
        fig.add_heatmap(x=midpoints(data["x_bins"]), y=midpoints(data["y_bins"]),
                        z=data["counts"], colorscale="YlOrRd")
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "bar":
        bars = pd.DataFrame(data["bars"])
        hue_col = "hue_value"
        if not bars.empty and hue_col in bars.columns:
            for label, group in bars.groupby(hue_col):
                fig.add_bar(x=group[data["y"]], y=group["value"], name=str(label))
        else:
            fig.add_bar(x=bars[data["y"]], y=bars["value"], name="Value")
        fig.update_layout(xaxis_title=data["y"], yaxis_title=data["x"], barmode="group")

    elif plot_type == "box":
        points = pd.DataFrame(data["points"])
        if points.empty:
            return empty_figure("No box-plot data available.")
        hue_col = data.get("hue") or data["y"]
        for label, group in points.groupby(hue_col):
            fig.add_box(x=group[data["y"]], y=group[data["x"]], name=str(label),
                        boxpoints="outliers")
        fig.update_layout(xaxis_title=data["y"], yaxis_title=data["x"])

    elif plot_type == "heatmap":
        fig.add_heatmap(x=data["x_categories"], y=data["y_categories"], z=data["values"],
                        colorscale="Blues")
        fig.update_layout(xaxis_title=data["x"], yaxis_title=data["y"])

    elif plot_type == "regression":
        points = pd.DataFrame(data["points"])
        fig.add_scattergl(x=points[data["x"]], y=points[data["y"]], mode="markers",
                          name="Points", opacity=0.55)
        fit = data.get("fit") or {}
        if fit.get("x_fit") and fit.get("y_fit"):
            fig.add_scatter(x=fit["x_fit"], y=fit["y_fit"], mode="lines",
                            name=fit.get("fit_type", "Fit"),
                            line=dict(color="#d62828", width=3))
        fig.update_layout(
            xaxis_title=data["x"], yaxis_title=data["y"],
            annotations=[dict(
                xref="paper", yref="paper", x=0, y=1.12, showarrow=False,
                text=(f"Pearson r: {round(float(data['pearson_correlation']), 4)}"
                      f"  |  R\u00b2: {round(float(data['pearson_correlation'])**2, 4)}"),
            )],
        )

    elif plot_type == "multiline":
        for line in data["lines"]:
            fig.add_scatter(x=line["x"], y=line["y"], mode="lines", name=line["label"])
        x_title = data["column"] if data["mode"] == "1d" else data.get("x_column", "x")
        y_title = "Count" if data["mode"] == "1d" else data["column"]
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

    elif plot_type == "correlation_matrix":
        cols = data["columns"]
        fig.add_heatmap(
            x=cols, y=cols, z=data["values"], colorscale="RdBu_r", zmid=0,
            text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in data["values"]],
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        )
        fig.update_layout(title=f"Correlation Matrix ({data['method'].title()})",
                          width=700, height=600)

    else:
        return empty_figure(f"Unsupported plot type: {plot_type}")

    return fig


# ---------------------------------------------------------------------------
# CSS
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
.instr-box {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border-left: 4px solid #4361ee;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 14px;
  font-size: 0.92rem;
}
.ab-badge {
  display: inline-block;
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  background: #eef2ff;
  color: #3730a3;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}
.clean-step-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.clean-step-card {
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
  border: 1px solid #dbeafe;
  border-radius: 10px;
  padding: 14px;
  min-height: 118px;
}
.clean-step-card .step-label {
  color: #4361ee;
  font-size: 0.76rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 6px;
}
.clean-step-card .step-title {
  color: #0f172a;
  font-size: 1rem;
  font-weight: 700;
  margin-bottom: 6px;
}
.clean-cta-box {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border: 1px solid #bfdbfe;
  border-radius: 12px;
  padding: 14px;
  margin-top: 10px;
}
.clean-cta-title {
  font-weight: 800;
  color: #1e3a8a;
  margin-bottom: 4px;
}
.clean-cta-note {
  color: #475569;
  font-size: 0.9rem;
  margin-bottom: 10px;
}
.clean-primary-btn .btn, .clean-secondary-btn .btn {
  font-weight: 700;
  min-height: 48px;
}
.clean-primary-btn .btn {
  background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
  border-color: #1d4ed8;
}
.clean-secondary-btn .btn {
  background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
  border-color: #0f172a;
}
.clean-version-panel {
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
"""


def cleaning_instruction_card_a():
    return ui.card(
        ui.card_header(ui.strong("How to Use Cleaning")),
        ui.div(
            {"class": "instr-box"},
            ui.tags.ul(
                ui.tags.li(
                    "Select a ",
                    ui.strong("dataset to clean"),
                    " from the sidebar picker, then choose an ",
                    ui.strong("Action"),
                    " and the relevant columns.",
                ),
                ui.tags.li(
                    "Always click ",
                    ui.strong("Preview"),
                    " first to inspect the before/after comparison chart and the row count impact before committing.",
                ),
                ui.tags.li(
                    "For missing-value decisions, use the ",
                    ui.strong("Overview tab"),
                    " to see which columns have NAs and their percentages.",
                ),
                ui.tags.li(
                    "For outlier decisions, use ",
                    ui.strong("EDA → 1D Plot"),
                    " (histogram/box) and ",
                    ui.strong("Scale Review"),
                    " in Overview to quantify column ranges.",
                ),
                ui.tags.li(
                    ui.strong("k-NN imputation"),
                    " is the default for missing values. It uses other numeric columns to estimate missing values; the description in the sidebar explains the feature-selection logic.",
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# UI
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
                    "Load, clean, transform, and explore datasets in the browser — no coding "
                    "required. All computation runs locally through pure Python modules."
                ),
                ui.p(ui.strong("Group Members: "),
                     "Bohong Zheng (bz2575), Zeming Liang (zl3688), "
                     "Zuer Weng (zw3118), Maya Rubin (mr4459)"),
                ui.tags.p(
                    {"class": "small-note",
                     "style": "margin-top: 8px; color: #64748b; font-style: italic;"},
                    "This app is part of a Columbia STAT 5243 class research project. "
                    "Anonymous session interaction events (no personal data, no uploaded file content) "
                    "are logged for the analysis. By using the app you consent to this "
                    "research-purposes logging.",
                ),
            ),
            col_widths=[12],
        ),
        ui.card(
            ui.card_header(ui.strong("Tab Overview")),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Important: "),
                "The tabs are ",
                ui.strong("not strictly sequential."),
                " You can (and should) jump to EDA or Overview at any point to assist "
                "decision-making for Cleaning or Feature Engineering. Think of the tabs as "
                "a toolbox, not a rigid pipeline.",
            ),
            ui.tags.ol(
                ui.tags.li(
                    ui.strong("Load "),
                    "— Upload a CSV/Excel/JSON/RDS file or load a built-in dataset. "
                    "If you load more than one dataset you will be asked to choose one to proceed with."
                ),
                ui.tags.li(
                    ui.strong("Overview "),
                    "— Quick decision-support summary: missing-value counts per column, "
                    "duplicate-row analysis, and numeric scale ranges (min/max/mean). "
                    "Use this before Cleaning and Feature Engineering to identify what "
                    "needs attention.",
                ),
                ui.tags.li(
                    ui.strong("Cleaning "),
                    "— Handle missing values (including k-NN imputation), remove duplicates, "
                    "scale or encode columns, handle outliers, standardize text, and coerce types. "
                    "Each operation offers a preview before applying.",
                ),
                ui.tags.li(
                    ui.strong("Feature Engineering "),
                    "— Apply 12 transforms (log, square, cube, interaction, ratio, binning, "
                    "one-hot, standardize, normalize, fill NA, drop NA, and custom algebraic "
                    "expressions). Before/after comparison charts are shown.",
                ),
                ui.tags.li(
                    ui.strong("EDA "),
                    "— In-depth statistical exploration: filter the data, inspect summary "
                    "statistics split by type (numeric vs. categorical), create 1D/2D plots, "
                    "run regression analysis, compare multiline distributions, and generate a "
                    "correlation matrix. EDA is equally useful before, during, and after "
                    "cleaning and feature engineering.",
                ),
            ),
        ),
        ui.card(
            ui.card_header(ui.strong("Overview vs. EDA")),
            ui.p(
                "Overview and EDA have ", ui.strong("partially overlapping"),
                " functionality. Overview is designed as a ",
                ui.em("quick, pre-cleaning decision guide"),
                " — a snapshot to inform what operations to apply. EDA provides ",
                ui.em("detailed, in-depth statistical studies"),
                " with interactive plots, regression, and correlation analysis. "
                "When you need richer context for a cleaning or feature-engineering decision, "
                "switch to EDA at any time.",
            ),
        ),
        ui.card(
            ui.card_header(ui.strong("Tips")),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Dataset Picker (per tab): "),
                "Every tab (Cleaning, Feature Engineering, EDA) has its own dataset picker "
                "so you can apply operations to any saved version independently.",
            ),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Descriptive Version Names: "),
                "Saved derived datasets are named to encode what was done — e.g., ",
                ui.tags.code("knn_Age_01"),
                " for a k-NN imputed version of the Age column. "
                "The full history is always visible in the Load tab.",
            ),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Filter Syntax: "),
                "Use pandas query expressions. Wrap each condition in parentheses before "
                "combining: ",
                ui.tags.code('(col_cat == "sex") & (col_num >= 5)'),
                ". Use ",
                ui.tags.code("=="),
                " for equality, ",
                ui.tags.code("&"),
                " / ",
                ui.tags.code("|"),
                " for AND/OR, and backticks for column names with spaces.",
            ),
            ui.div(
                {"class": "tip-box"},
                ui.strong("Preview First: "),
                "Cleaning and Feature Engineering both have a Preview button. "
                "Always preview before applying.",
            ),
        ),
        ui.accordion(
            ui.accordion_panel(
                "Team only — download A/B event log",
                ui.tags.p(
                    {"class": "small-note"},
                    "Team members: enter the team password to reveal the download button for ",
                    ui.tags.code("ab_test_events.csv"),
                    ". Others can ignore this section.",
                ),
                ui.input_password("admin_pwd", "Team password", placeholder="•••••••"),
                ui.output_ui("admin_download_area"),
            ),
            id="guide_admin",
            open=False,
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
                        "builtin_dataset", "Choose built-in dataset",
                        {"sleep_health": "Sleep, Mobile and Stress",
                         "iris": "Iris", "tips": "Tips (Restaurant)"},
                    ),
                    ui.tooltip(
                        ui.input_action_button("load_builtin_btn", "Load Built-in Dataset",
                                               class_="btn-dark w-100"),
                        "Load the selected built-in dataset into session memory",
                    ),
                ),
                ui.card(
                    ui.card_header(ui.strong("Upload Dataset")),
                    ui.input_file("upload_file", "Upload CSV, Excel, JSON, or RDS",
                                  accept=[".csv", ".xlsx", ".xls", ".json", ".rds"]),
                    ui.tooltip(
                        ui.input_action_button("load_upload_btn", "Load Uploaded File",
                                               class_="btn-outline-dark w-100"),
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

    # ── Overview Tab ───────────────────────────────────────────────────────
    ui.nav_panel(
        "Overview",
        ui.card(
            ui.card_header(ui.strong("How to Use Overview")),
            ui.div(
                {"class": "instr-box"},
                ui.tags.ul(
                    ui.tags.li(
                        "Overview provides a ",
                        ui.strong("quick, at-a-glance summary"),
                        " of the active dataset to guide Cleaning and Feature Engineering decisions.",
                    ),
                    ui.tags.li(
                        "For ",
                        ui.strong("in-depth statistical analysis"),
                        " — interactive plots, regression, correlation — use the ",
                        ui.strong("EDA tab"),
                        ". EDA can be opened at any time to support decision-making.",
                    ),
                    ui.tags.li(
                        "Switch the active dataset in the ",
                        ui.strong("Load tab"),
                        " to compare Overview summaries for different versions.",
                    ),
                ),
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.strong("Missing Value Overview")),
                ui.output_ui("overview_missing_content"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Duplicate Overview")),
                ui.output_ui("overview_duplicate_content"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header(ui.strong("Scale Review (Numeric Columns)")),
            ui.output_ui("overview_scale_content"),
            full_screen=True,
        ),
    ),

    # ── Cleaning Tab ───────────────────────────────────────────────────────
    ui.nav_panel(
        "Cleaning",
        # Group-B-only <style> injection (empty div for Group A). Kept inside
        # the Cleaning tab so CSS only loads when users are on this tab.
        ui.output_ui("cleaning_b_theme_style"),
        # Group-B-only top banner. Empty div for Group A.
        ui.output_ui("cleaning_b_top_banner"),
        # Themeable wrapper — Group B's CSS targets this id for descendants.
        ui.div(
            {"id": "cleaning-tab-content"},
            ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Cleaning / Preprocessing", class_="text-uppercase fw-bold"),
                ui.output_ui("cleaning_sidebar_intro"),
                ui.hr(),
                ui.input_select("clean_df_picker", "Dataset to clean", {}),
                ui.hr(),
                ui.tooltip(
                    ui.input_select(
                        "clean_action", "Action",
                        {
                            "handle_missing": "Handle missing values",
                            "remove_duplicates": "Remove duplicates",
                            "scale_columns": "Scale numeric columns",
                            "encode_columns": "Encode categorical columns",
                            "handle_outliers": "Handle outliers",
                            "standardize_text": "Standardize text (whitespace & case)",
                            "coerce_types": "Coerce column types",
                        },
                    ),
                    "Choose a preprocessing operation to apply to the selected dataset",
                ),
                ui.input_selectize("clean_columns", "Columns", [], multiple=True),
                ui.panel_conditional(
                    "input.clean_action === 'handle_outliers'",
                    ui.input_select("clean_single_column", "Single column", {"": "— select a column —"}),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'handle_missing'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_strategy", "Missing-value strategy",
                            {
                                "knn": "k-NN imputation (recommended)",
                                "drop_rows": "Drop rows",
                                "drop_cols": "Drop columns",
                                "mean": "Fill with mean",
                                "median": "Fill with median",
                                "mode": "Fill with mode",
                                "constant": "Fill with constant",
                            },
                            selected="knn",
                        ),
                        "How to handle missing values in selected columns",
                    ),
                    ui.input_text("clean_constant_value", "Constant value", "",
                                  placeholder="e.g., 0 or unknown"),
                    ui.panel_conditional(
                        "input.clean_strategy === 'knn'",
                        ui.input_numeric("clean_knn_k", "k (neighbours)", 5, min=1, max=100),
                        ui.div(
                            {"class": "tip-box", "style": "margin-top:8px;"},
                            ui.tags.small(
                                ui.strong("k-NN Imputation: "),
                                "Fills missing values by finding k similar rows using other "
                                "numeric columns as features, then averaging those neighbours' "
                                "values. Only numeric columns with ≥ 80 % valid values in the "
                                "rows that need imputation are used as features. Features are "
                                "automatically scaled to unit variance before distance "
                                "computation so no single column dominates. "
                                "Rows with no valid features are dropped. "
                                "Uses a KD-tree internally so it is efficient even for large "
                                "datasets (tens of thousands of rows).",
                            ),
                        ),
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'scale_columns'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_scale_method", "Scaling method",
                            {"standard": "Standard", "minmax": "Min-Max", "robust": "Robust"},
                        ),
                        "Algorithm for rescaling numeric values to a standard range",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'encode_columns'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_encode_method", "Encoding method",
                            {"label": "Label encode", "onehot": "One-hot encode"},
                        ),
                        "Method for converting categorical values to numbers",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'handle_outliers'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_outlier_action", "Outlier action",
                            {"remove": "Remove rows", "cap": "Cap values"},
                        ),
                        "How to treat values outside the IQR fence boundaries",
                    ),
                    ui.input_numeric("clean_iqr", "IQR multiplier", 1.5, min=0.5, step=0.5),
                    ui.tags.small(
                        {"class": "small-note"},
                        "IQR = Q3 - Q1. Outliers lie beyond Q1 - k*IQR or Q3 + k*IQR. "
                        "Use 1.5 for mild, 3.0 for extreme.",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'standardize_text'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_text_case", "Case transform",
                            {"lower": "Lowercase", "upper": "Uppercase",
                             "title": "Title Case", "none": "No change"},
                        ),
                        "Letter case normalization to apply to string columns",
                    ),
                ),
                ui.panel_conditional(
                    "input.clean_action === 'coerce_types'",
                    ui.tooltip(
                        ui.input_select(
                            "clean_coerce_target", "Target type",
                            {"numeric": "Numeric (non-convertible → NaN)", "string": "String"},
                        ),
                        "Target data type — non-convertible values become NaN for numeric",
                    ),
                ),
                ui.input_radio_buttons(
                    "clean_save_mode", "Apply mode",
                    {"derived": "Save as derived version", "current": "Apply to current version"},
                    selected="current", inline=False,
                ),
                ui.hr(),
                ui.output_ui("cleaning_action_buttons"),
                ui.output_ui("cleaning_result_summary"),
                width="380px",
            ),
            ui.output_ui("cleaning_instructions_card"),
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
    ),

    # ── Feature Engineering Tab ────────────────────────────────────────────
    ui.nav_panel(
        "Feature Engineering",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Feature Engineering", class_="text-uppercase fw-bold"),
                ui.hr(),
                ui.input_select("feature_df_picker", "Dataset to transform", {}),
                ui.hr(),
                ui.tooltip(
                    ui.input_select(
                        "feature_method", "Method",
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
                            "custom_expr": "Custom New Column",
                        },
                    ),
                    "Type of feature transformation — see explanation below",
                ),
                ui.output_ui("feature_explanation"),
                ui.panel_conditional(
                    "input.feature_method !== 'custom_expr'",
                    ui.tooltip(
                        ui.input_select("feature_col1", "Primary column", {}),
                        "Column to transform (required for all methods except Custom)",
                    ),
                    ui.tooltip(
                        ui.input_select("feature_col2", "Secondary column", {}),
                        "Second column — only used for Interaction and Ratio transforms",
                    ),
                ),
                ui.panel_conditional(
                    "input.feature_method === 'custom_expr'",
                    ui.input_text(
                        "feature_custom_expr", "Algebraic expression",
                        placeholder="e.g., col_a * 2 + col_b",
                    ),
                    ui.div(
                        {"class": "tip-box", "style": "margin-top:8px;"},
                        ui.tags.small(
                            ui.strong("Custom New Column: "),
                            "Enter a pandas-eval expression referencing existing column names "
                            "(e.g., ",
                            ui.tags.code("(price - cost) / price"),
                            "). The result is added as a new column. If a column name "
                            "contains spaces, wrap it in backticks. Errors for non-existent "
                            "columns or invalid syntax are reported immediately.",
                        ),
                        ui.output_ui("feature_custom_expr_columns"),
                    ),
                ),
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
                        "feature_fill_strategy", "Fill strategy",
                        {"mean": "Mean", "median": "Median", "mode": "Mode", "constant": "Constant"},
                    ),
                    ui.input_text("feature_fill_value", "Constant fill value", "",
                                  placeholder="e.g., 0 or missing"),
                ),
                ui.input_radio_buttons(
                    "feature_save_mode", "Apply mode",
                    {"derived": "Save as derived version", "current": "Apply to current version"},
                    selected="current",
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
                ui.card_header(ui.strong("How to Use Feature Engineering")),
                ui.div(
                    {"class": "instr-box"},
                    ui.tags.ul(
                        ui.tags.li(
                            "Select a ",
                            ui.strong("dataset to transform"),
                            " from the sidebar picker, choose a ",
                            ui.strong("Method"),
                            ", and configure the parameters.",
                        ),
                        ui.tags.li(
                            "Preview the transformation before applying — the before/after "
                            "comparison chart shows the distribution change.",
                        ),
                        ui.tags.li(
                            "Use ",
                            ui.strong("EDA → 1D Plot"),
                            " to examine column distributions and decide on appropriate "
                            "transforms (e.g., check skewness before applying log transform).",
                        ),
                        ui.tags.li(
                            "Use ",
                            ui.strong("Custom New Column"),
                            " for any algebraic combination not covered by the built-in "
                            "methods — enter any pandas-eval expression.",
                        ),
                    ),
                ),
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
        ui.card(
            ui.card_header(ui.strong("How to Use EDA")),
            ui.div(
                {"class": "instr-box"},
                ui.tags.ul(
                    ui.tags.li(
                        "Select a dataset from the picker below. EDA is not only for final "
                        "analysis — it is a ",
                        ui.strong("powerful aid for cleaning and feature-engineering decisions"),
                        " at any stage.",
                    ),
                    ui.tags.li(
                        ui.strong("Filter"),
                        ": apply a pandas query to subset rows. Wrap each condition in "
                        "parentheses before combining: ",
                        ui.tags.code('(col_cat == "val") & (col_num >= 5)'),
                        ".",
                    ),
                    ui.tags.li(
                        ui.strong("Describe"),
                        ": numeric and categorical statistics shown in separate tables.",
                    ),
                    ui.tags.li(
                        ui.strong("1D / 2D Plot"),
                        ": visualize distributions and relationships; "
                        "useful for outlier detection and transform decisions.",
                    ),
                    ui.tags.li(
                        ui.strong("Regression"),
                        ": fit polynomial, robust, or LOWESS curves; see Pearson r and p-value.",
                    ),
                    ui.tags.li(
                        ui.strong("Multiline"),
                        ": compare distributions of a numeric column across categories.",
                    ),
                    ui.tags.li(
                        ui.strong("Correlation Matrix"),
                        ": Pearson, Spearman, or Kendall; useful before feature selection.",
                    ),
                ),
            ),
        ),
        ui.card(
            ui.card_header(ui.strong("Dataset")),
            ui.input_select("eda_df_picker", "Dataset to explore", {}),
        ),
        # Filtering
        ui.card(
            ui.card_header(ui.strong("Filtering")),
            ui.layout_columns(
                ui.input_text_area(
                    "filter_expr", "Pandas query expression",
                    placeholder=(
                        'Example: (age > 30) & (gender == "Female")  '
                        '|  Use `backticks` for column names with spaces'
                    ),
                    rows=2,
                ),
                ui.div(
                    ui.input_radio_buttons(
                        "filter_save_mode", "Filter mode",
                        {"derived": "Save filtered version",
                         "current": "Replace current version"},
                        selected="derived", inline=True,
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
                ui.card_header(ui.strong("Describe — Numeric")),
                ui.output_data_frame("describe_num_table"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("Describe — Categorical")),
                ui.output_data_frame("describe_cat_table"),
                full_screen=True,
            ),
            col_widths=[4, 4, 4],
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
                ui.tags.small({"class": "small-note"}, "More bins = finer detail; fewer = smoother shape"),
                ui.input_checkbox("plot1d_normalize", "Normalize counts", False),
                ui.input_checkbox("plot1d_logx", "Log-scale X", False),
                ui.input_checkbox("plot1d_logy", "Log-scale Y", False),
                ui.input_action_button("render_1d_btn", "Render 1D Plot", class_="btn-dark btn-sm"),
                output_widget("plot_1d", height="380px"),
                ui.output_ui("plot1d_stats"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(ui.strong("2D Plot")),
                ui.input_select("plot2d_x", "X column", {}),
                ui.input_select("plot2d_y", "Y column", {}),
                ui.input_select("plot2d_hue", "Hue (optional)", {"": "None"}),
                ui.tooltip(
                    ui.input_select(
                        "plot2d_kind", "2D plot kind",
                        {"auto": "Auto", "hist": "2D histogram", "scatter": "Scatter",
                         "line": "Line", "bar": "Bar", "box": "Box", "heatmap": "Heatmap"},
                    ),
                    "Chart type — Auto detects based on column types",
                ),
                ui.input_checkbox("plot2d_logx", "Log-scale X", False),
                ui.input_checkbox("plot2d_logy", "Log-scale Y", False),
                ui.input_action_button("render_2d_btn", "Render 2D Plot", class_="btn-dark btn-sm"),
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
                    ui.tooltip(
                        ui.input_numeric("regression_order", "Polynomial order", 1, min=1, max=5),
                        "Degree of the polynomial fit: 1=linear, 2=quadratic, 3=cubic, etc.",
                    ),
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
                ui.tags.small({"class": "small-note"},
                              "Number of equal-width bins for grouping the distribution"),
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
                ui.tooltip(
                    ui.input_select(
                        "corr_method", "Method",
                        {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"},
                    ),
                    "Pearson measures linear correlation; Spearman/Kendall measure monotonic association",
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
    title=ui.tags.span("STAT 5243 Data Workbench",
                       style="font-weight:800; letter-spacing:0.5px;"),
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
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):
    datasets_state = reactive.value(OrderedDict())
    active_key_state = reactive.value(None)
    messages_state = reactive.value([])
    ab_group_state = reactive.value(random.choice(["A", "B"]))
    session_id = str(uuid.uuid4())
    cleaning_entry_time = reactive.value(datetime.now())

    # Team-only override: visiting the app with ?force_group=A or
    # ?force_group=B in the URL pins that session to the specified arm.
    # Useful when team members want to screenshot or verify a specific
    # layout without refreshing incognito windows until the coin lands.
    # Real users won't know this parameter exists (it's not surfaced in
    # the UI or sharing copy), so blinding is preserved.
    @reactive.effect
    def _honor_force_group_url_param():
        try:
            query_str = input[".clientdata_url_search"]() or ""
        except Exception:
            return
        params = parse_qs(query_str.lstrip("?"))
        forced = (params.get("force_group", [""])[0] or "").strip().upper()
        if forced in {"A", "B"} and ab_group_state.get() != forced:
            ab_group_state.set(forced)

    cleaning_preview_df = reactive.value(pd.DataFrame())
    cleaning_preview_meta = reactive.value("")
    feature_preview_df = reactive.value(pd.DataFrame())
    feature_preview_meta = reactive.value("")

    plot1d_payload = reactive.value(None)
    plot1d_stats_text = reactive.value("")
    plot2d_payload = reactive.value(None)
    regression_payload = reactive.value(None)
    multiline_payload = reactive.value(None)
    corr_payload = reactive.value(None)
    clean_comparison_fig = reactive.value(None)
    feature_comparison_fig = reactive.value(None)

    # Tracks whether a multi-source conflict modal is pending
    _conflict_pending = reactive.value(False)

    def log_ab_event(event_type: str, *, success: bool | None = None,
                     details: str = "") -> None:
        # Degrade silently on any I/O error: the user's action must not crash
        # if the log file is unwritable (read-only filesystem, disk full, etc.).
        try:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": session_id,
                "ab_group": ab_group_state.get(),
                "event_type": event_type,
                "clean_action": (input.clean_action() if hasattr(input, "clean_action") else None) or "",
                "dataset_key": (input.clean_df_picker() if hasattr(input, "clean_df_picker") and input.clean_df_picker() else ""),
                "columns_count": len(list(input.clean_columns() or [])) if hasattr(input, "clean_columns") else 0,
                "success": "" if success is None else str(bool(success)),
                "seconds_since_session_start": round((datetime.now() - cleaning_entry_time.get()).total_seconds(), 3),
                "details": details,
            }
            file_exists = AB_LOG_PATH.exists()
            with AB_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            # Swallow — a logging failure should never block the experiment's
            # user-facing flow. No push_message here either, so users aren't
            # alerted that the experiment exists.
            pass

    def push_message(level: str, text: str) -> None:
        items = list(messages_state.get())
        items.insert(0, {"level": level, "text": text})
        messages_state.set(items[:8])

    def clear_previews() -> None:
        cleaning_preview_df.set(pd.DataFrame())
        cleaning_preview_meta.set("")
        feature_preview_df.set(pd.DataFrame())
        feature_preview_meta.set("")

    def cleaning_action_hint(action: str | None) -> str:
        hints = {
            "handle_missing": "Step 3 focuses on missing data. Select the columns with NA values, then choose whether to impute, fill, or drop them.",
            "remove_duplicates": "Step 3 is simple here. Preview first so users can see how many duplicate rows will be removed before saving.",
            "scale_columns": "Choose numeric columns only. Previewing first helps users verify that the rescaled values still make sense.",
            "encode_columns": "Choose categorical columns to convert into numeric form. Previewing helps users confirm the new encoded structure.",
            "handle_outliers": "Pick one numeric column, then decide whether to cap extreme values or remove the affected rows.",
            "standardize_text": "Use this when text values differ only by case or spacing. It is useful before grouping or encoding text categories.",
            "coerce_types": "Choose the columns to convert, then preview the result to check whether non-convertible values become missing as expected.",
        }
        return hints.get(action or "handle_missing", "Choose an action, configure its inputs, preview the result, and apply only when the output looks correct.")

    @output
    @render.ui
    def cleaning_b_theme_style():
        # Group B only: inject a <style> block that applies the dramatic
        # dark-gradient theme to #cleaning-tab-content. Group A gets nothing.
        if ab_group_state.get() != "B":
            return ui.div()
        return ui.tags.style(GROUP_B_DRAMATIC_CSS)

    @output
    @render.ui
    def cleaning_b_top_banner():
        # Group B only: prominent gradient banner + 4-step progress indicator
        # above the tab body. Group A gets nothing (keeps the familiar layout).
        if ab_group_state.get() != "B":
            return ui.div()
        return ui.div(
            {"class": "cleaning-b-banner"},
            ui.div(
                {"class": "cleaning-b-banner-inner"},
                ui.div({"class": "cleaning-b-banner-title"},
                       "Data Cleaning Studio"),
                ui.div({"class": "cleaning-b-banner-subtitle"},
                       "Guided pipeline — choose a dataset, pick an action, preview the impact, then save."),
                ui.div(
                    {"class": "cleaning-b-banner-steps"},
                    ui.div({"class": "cb-step"},
                           ui.div({"class": "cb-step-num"}, "1"),
                           ui.div({"class": "cb-step-label"}, "Dataset")),
                    ui.div({"class": "cb-step-arrow"}, "→"),
                    ui.div({"class": "cb-step"},
                           ui.div({"class": "cb-step-num"}, "2"),
                           ui.div({"class": "cb-step-label"}, "Action")),
                    ui.div({"class": "cb-step-arrow"}, "→"),
                    ui.div({"class": "cb-step"},
                           ui.div({"class": "cb-step-num"}, "3"),
                           ui.div({"class": "cb-step-label"}, "Parameters")),
                    ui.div({"class": "cb-step-arrow"}, "→"),
                    ui.div({"class": "cb-step"},
                           ui.div({"class": "cb-step-num"}, "4"),
                           ui.div({"class": "cb-step-label"}, "Preview & Save")),
                ),
            ),
        )

    @output
    @render.ui
    def cleaning_sidebar_intro():
        # Blinded: do not label the user's arm. The treatment effect in B comes
        # from the layout + CTA box below (plus the dramatic theme), not from
        # telling the user they are in a "Version B / Guided" condition.
        group = ab_group_state.get()
        tip_text = "Tip: use the Preview button to see the before/after impact before applying changes."
        if group == "A":
            return ui.div({"class": "small-note"}, tip_text)
        return ui.div(
            {"class": "clean-version-panel"},
            ui.tags.p({"class": "small-note", "style": "margin: 0;"}, tip_text),
        )

    @output
    @render.ui
    def cleaning_action_buttons():
        group = ab_group_state.get()
        if group == "A":
            return ui.layout_columns(
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
            )
        return ui.div(
            {"class": "clean-cta-box"},
            ui.div({"class": "clean-cta-title"}, "Step 4 — Preview, then save"),
            ui.div(
                {"class": "clean-cta-note"},
                "Use Preview Changes first. If the preview looks correct, finish the task with Apply and Save.",
            ),
            ui.layout_columns(
                ui.div(
                    {"class": "clean-primary-btn"},
                    ui.tooltip(
                        ui.input_action_button("preview_clean_btn", "Preview Changes",
                                               class_="btn-primary w-100"),
                        "Preview the result without changing the active dataset",
                    ),
                ),
                ui.div(
                    {"class": "clean-secondary-btn"},
                    ui.tooltip(
                        ui.input_action_button("apply_clean_btn", "Apply and Save",
                                               class_="btn-success w-100"),
                        "Apply transformation and save the result",
                    ),
                ),
                col_widths=[6, 6],
            ),
        )

    @output
    @render.ui
    def cleaning_instructions_card():
        group = ab_group_state.get()
        if group == "A":
            return cleaning_instruction_card_a()
        return ui.TagList(
            ui.card(
                ui.card_header(ui.strong("Guided Cleaning Workflow")),
                ui.div(
                    {"class": "clean-step-grid"},
                    ui.div(
                        {"class": "clean-step-card"},
                        ui.div({"class": "step-label"}, "Step 1"),
                        ui.div({"class": "step-title"}, "Choose a dataset"),
                        ui.div("Pick the dataset version you want to clean from the sidebar dataset selector."),
                    ),
                    ui.div(
                        {"class": "clean-step-card"},
                        ui.div({"class": "step-label"}, "Step 2"),
                        ui.div({"class": "step-title"}, "Choose a cleaning action"),
                        ui.div("Select the operation you want to perform, such as filling missing values or scaling columns."),
                    ),
                    ui.div(
                        {"class": "clean-step-card"},
                        ui.div({"class": "step-label"}, "Step 3"),
                        ui.div({"class": "step-title"}, "Set the parameters"),
                        ui.div(cleaning_action_hint(input.clean_action() if hasattr(input, "clean_action") else None)),
                    ),
                    ui.div(
                        {"class": "clean-step-card"},
                        ui.div({"class": "step-label"}, "Step 4"),
                        ui.div({"class": "step-title"}, "Preview before applying"),
                        ui.div("Check the preview table and the before/after chart first, then save only when the result looks right."),
                    ),
                ),
            ),
            ui.card(
                ui.card_header(ui.strong("Current Task Hint")),
                ui.div(
                    {"class": "instr-box"},
                    # Blinded: removed the explicit "Assigned version: B
                    # (guided treatment)" line. The guided-workflow cards above
                    # are the intervention; users do not need to be told.
                    ui.tags.p(cleaning_action_hint(input.clean_action() if hasattr(input, "clean_action") else None)),
                    ui.tags.p(
                        {"class": "small-note mb-0"},
                        "For missing-value and outlier decisions, you can still use Overview and EDA as supporting tabs before returning here.",
                    ),
                ),
            ),
        )

    # ── Core reactive calcs ─────────────────────────────────────────────

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
    def clean_active_df() -> pd.DataFrame | None:
        datasets = datasets_state.get()
        key = input.clean_df_picker()
        if key and key in datasets:
            return datasets[key]["df"]
        return current_df()

    @reactive.calc
    def feature_active_df() -> pd.DataFrame | None:
        datasets = datasets_state.get()
        key = input.feature_df_picker()
        if key and key in datasets:
            return datasets[key]["df"]
        return current_df()

    @reactive.calc
    def eda_active_df() -> pd.DataFrame | None:
        datasets = datasets_state.get()
        key = input.eda_df_picker()
        if key and key in datasets:
            return datasets[key]["df"]
        return current_df()

    def _tab_picker_key(picker_input: str) -> str | None:
        """Return the key currently selected in a tab picker (or active_key)."""
        datasets = datasets_state.get()
        try:
            val = getattr(input, picker_input)()
        except Exception:
            val = None
        if val and val in datasets:
            return val
        return active_key_state.get()

    # ── Sync dataset picker dropdowns ───────────────────────────────────

    @reactive.effect
    def _sync_dataset_picker() -> None:
        datasets = datasets_state.get()
        choices = {key: f"{record['label']} [{key}]"
                   for key, record in datasets.items()}
        ui.update_select("dataset_picker", choices=choices,
                         selected=active_key_state.get(), session=session)

    @reactive.effect
    def _sync_tab_pickers() -> None:
        datasets = datasets_state.get()
        active_key = active_key_state.get()
        choices = {key: f"{record['label']} [{key}]"
                   for key, record in datasets.items()}
        for picker in ("clean_df_picker", "feature_df_picker", "eda_df_picker"):
            try:
                current_val = getattr(input, picker)()
            except Exception:
                current_val = None
            selected = current_val if (current_val and current_val in datasets) else active_key
            ui.update_select(picker, choices=choices, selected=selected, session=session)

    @reactive.effect
    @reactive.event(input.dataset_picker)
    def _activate_from_picker() -> None:
        key = input.dataset_picker()
        if key and key in datasets_state.get():
            active_key_state.set(key)
            clear_previews()

    # ── Column input sync (split by tab) ────────────────────────────────

    @reactive.effect
    def _sync_clean_col_inputs() -> None:
        df = clean_active_df()
        action = input.clean_action()
        if df is None:
            empty: dict[str, str] = {}
            ui.update_selectize("clean_columns", choices=empty, selected=[], session=session)
            ui.update_select("clean_single_column",
                             choices={"": "— select a column —"},
                             selected="", session=session)
            return

        all_cols = [str(c) for c in df.columns]
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

        if action == "scale_columns":
            col_choices = {c: c for c in numeric_cols}
        elif action in ("encode_columns", "standardize_text"):
            col_choices = {c: c for c in categorical_cols}
        else:
            col_choices = {c: c for c in all_cols}

        ui.update_selectize("clean_columns", choices=col_choices, selected=[], session=session)

        single_base = numeric_cols if numeric_cols else all_cols
        ui.update_select(
            "clean_single_column",
            choices={"": "— select a column —", **{c: c for c in single_base}},
            selected="",
            session=session,
        )

    @reactive.effect
    def _sync_feature_col_inputs() -> None:
        df = feature_active_df()
        if df is None:
            empty: dict[str, str] = {}
            for w in ("feature_col1", "feature_col2"):
                ui.update_select(w, choices=empty, session=session)
            return
        all_cols = [str(c) for c in df.columns]
        ui.update_select("feature_col1", choices={c: c for c in all_cols}, session=session)
        ui.update_select(
            "feature_col2",
            choices={"": "None", **{c: c for c in all_cols}},
            selected="",
            session=session,
        )

    @reactive.effect
    def _sync_eda_col_inputs() -> None:
        df = eda_active_df()
        if df is None:
            empty: dict[str, str] = {}
            for w in ("plot1d_column", "plot2d_x", "plot2d_y",
                      "regression_x", "regression_y",
                      "multiline_value", "multiline_group"):
                ui.update_select(w, choices=empty, session=session)
            ui.update_select("plot2d_hue", choices={"": "None"}, selected="", session=session)
            return

        all_cols = [str(c) for c in df.columns]
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]
        typed_cols = {c: f"{c} (num)" if c in numeric_cols else f"{c} (cat)" for c in all_cols}

        ui.update_select("plot1d_column", choices={c: c for c in all_cols}, session=session)
        ui.update_select("plot2d_x", choices=typed_cols, session=session)
        ui.update_select("plot2d_y", choices=typed_cols, session=session)
        ui.update_select("plot2d_hue", choices={"": "None", **typed_cols},
                         selected="", session=session)
        ui.update_select("regression_x", choices={c: c for c in numeric_cols}, session=session)
        ui.update_select("regression_y", choices={c: c for c in numeric_cols}, session=session)
        ui.update_select("multiline_value", choices={c: c for c in numeric_cols}, session=session)
        ui.update_select("multiline_group", choices={c: c for c in categorical_cols},
                         session=session)

    # ── Dataset loading ─────────────────────────────────────────────────

    def _check_and_prompt_conflict(new_datasets: OrderedDict, new_key: str) -> bool:
        """Show a conflict modal if ≥2 source datasets exist. Returns True if conflict."""
        source_keys = [k for k in new_datasets.keys()
                       if k == "original" or k.startswith("loaded_")]
        if len(source_keys) >= 2:
            choices = {k: f"{new_datasets[k]['label']} [{k}]" for k in source_keys}
            ui.modal_show(
                ui.modal(
                    ui.p(
                        "You have loaded more than one source dataset. "
                        "Please choose one to keep. All derived versions "
                        "(cleaned, feature-engineered, filtered) will be removed "
                        "and you will start fresh from the chosen dataset."
                    ),
                    ui.input_radio_buttons(
                        "modal_dataset_choice",
                        "Keep this dataset:",
                        choices=choices,
                        selected=new_key,
                    ),
                    title="Multiple Datasets Loaded",
                    footer=ui.div(
                        ui.input_action_button(
                            "modal_confirm_btn", "Confirm", class_="btn-dark me-2"
                        ),
                        ui.modal_button("Cancel"),
                    ),
                    easy_close=False,
                )
            )
            _conflict_pending.set(True)
            return True
        return False

    @reactive.effect
    @reactive.event(input.load_builtin_btn)
    def _load_builtin() -> None:
        name = input.builtin_dataset()
        try:
            df = load_builtin_dataset(name)
            prefix = "original" if not datasets_state.get() else "loaded"
            datasets, key = register_dataset_version(
                datasets_state.get(), df,
                prefix=prefix, label=f"Built-in: {name}", transform="built-in load",
            )
            datasets_state.set(datasets)
            active_key_state.set(key)
            clear_previews()
            if not _check_and_prompt_conflict(datasets, key):
                push_message("success", f"Loaded built-in dataset '{name}' as [{key}].")
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
                datasets_state.get(), df,
                prefix=prefix, label=f"Upload: {info['name']}",
                transform=f"uploaded {Path(info['name']).suffix.lower()}",
            )
            datasets_state.set(datasets)
            active_key_state.set(key)
            clear_previews()
            if not _check_and_prompt_conflict(datasets, key):
                push_message("success", f"Loaded '{info['name']}' as [{key}].")
        except Exception as exc:
            push_message("error", f"Failed to load uploaded file: {exc}")

    @reactive.effect
    @reactive.event(input.modal_confirm_btn)
    def _modal_confirm() -> None:
        chosen_key = input.modal_dataset_choice()
        datasets = datasets_state.get()
        if not chosen_key or chosen_key not in datasets:
            return
        # Keep only the chosen source dataset
        new_datasets: OrderedDict[str, dict[str, Any]] = OrderedDict()
        new_datasets[chosen_key] = datasets[chosen_key]
        datasets_state.set(new_datasets)
        active_key_state.set(chosen_key)
        _conflict_pending.set(False)
        clear_previews()
        ui.modal_remove()
        push_message("success",
                     f"Kept dataset [{chosen_key}]. All other versions removed. "
                     "Starting fresh from this dataset.")

    # ── Cleaning operations ─────────────────────────────────────────────

    def compute_cleaning_result() -> tuple[pd.DataFrame, str, list[tuple[str, str]]]:
        """Run the selected cleaning op. Returns (transformed_df, summary, extra_messages)."""
        df = clean_active_df()
        if df is None:
            raise ValueError("Load a dataset first.")

        action = input.clean_action()
        extra_msgs: list[tuple[str, str]] = []

        if action == "handle_missing":
            columns = list(input.clean_columns() or [])
            strategy = input.clean_strategy()

            if strategy in ("drop_rows", "drop_cols") and not columns:
                raise ValueError(
                    f"Select at least one column before using '{strategy}'. "
                    "Without a column selection the operation would scan ALL columns and "
                    "drop far more rows than expected."
                )

            if strategy == "knn":
                if not columns:
                    raise ValueError("Select at least one column for k-NN imputation.")
                k = int(input.clean_knn_k())
                transformed, warning_msg = cleaning.knn_impute(df, columns, k=k)
                n_dropped = len(df) - len(transformed)
                summary = (
                    f"k-NN imputation (k={k}) on {columns}: "
                    f"{n_dropped} rows dropped, {len(transformed)}/{len(df)} rows retained."
                )
                if warning_msg:
                    extra_msgs.append(("warning", warning_msg))
            else:
                transformed = cleaning.handle_missing(
                    df, columns=columns or None,
                    strategy=strategy,
                    constant_value=coerce_text_value(input.clean_constant_value()),
                )
                summary = f"handle_missing strategy='{strategy}' → shape {transformed.shape}."

        elif action == "remove_duplicates":
            transformed = cleaning.remove_duplicates(df)
            n_dupes = len(df) - len(transformed)
            return transformed, f"Found {n_dupes} duplicate rows out of {len(df)} total.", []

        elif action == "scale_columns":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more numeric columns to scale.")
            transformed = cleaning.scale_columns(df, columns=columns,
                                                  method=input.clean_scale_method())
            summary = f"scale_columns method='{input.clean_scale_method()}' on {columns}."

        elif action == "encode_columns":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more categorical columns to encode.")
            transformed = cleaning.encode_columns(df, columns=columns,
                                                   method=input.clean_encode_method())
            summary = f"encode_columns method='{input.clean_encode_method()}' on {columns}."

        elif action == "handle_outliers":
            column = input.clean_single_column()
            if not column:
                raise ValueError("Select a column for outlier handling.")
            diagnostics = cleaning.detect_outliers(df, column=column,
                                                    iqr_multiplier=float(input.clean_iqr()))
            transformed = cleaning.handle_outliers(df, column=column,
                                                    action=input.clean_outlier_action(),
                                                    iqr_multiplier=float(input.clean_iqr()))
            return transformed, (
                f"Outlier handling on '{column}': {diagnostics['n_outliers']} outliers "
                f"(Q1={diagnostics['q1']:.2f}, Q3={diagnostics['q3']:.2f}, "
                f"IQR={diagnostics['iqr']:.2f}, "
                f"bounds=[{diagnostics['lower_bound']:.2f}, {diagnostics['upper_bound']:.2f}], "
                f"multiplier={input.clean_iqr()})."
            ), []

        elif action == "standardize_text":
            columns = list(input.clean_columns() or [])
            numeric_selected = [c for c in columns
                                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_selected:
                extra_msgs.append(("warning",
                    f"Columns {numeric_selected} are numeric and will be converted to strings."))
            transformed = cleaning.standardize_text(
                df, columns=columns or None, case=input.clean_text_case(),
            )
            summary = f"standardize_text case='{input.clean_text_case()}' → shape {transformed.shape}."

        elif action == "coerce_types":
            columns = list(input.clean_columns() or [])
            if not columns:
                raise ValueError("Select one or more columns to coerce.")
            transformed = cleaning.coerce_column_types(df, columns=columns,
                                                        target=input.clean_coerce_target())
            summary = f"coerce_types target='{input.clean_coerce_target()}' on {columns}."

        else:
            raise ValueError(f"Unsupported cleaning action: {action}")

        return transformed, summary, extra_msgs

    def apply_transformed_result(
        transformed: pd.DataFrame,
        *,
        mode: str,
        prefix: str,
        label: str,
        transform: str,
        source_key: str | None = None,
    ) -> str:
        datasets = datasets_state.get()
        src_key = source_key or active_key_state.get()
        if src_key is None:
            raise ValueError("No active dataset to update.")
        if mode == "current":
            datasets_state.set(overwrite_dataset_version(datasets, src_key, transformed,
                                                          transform=transform))
            active_key_state.set(src_key)
            return src_key
        datasets, key = register_dataset_version(
            datasets, transformed,
            prefix=prefix, label=label,
            source_key=src_key, transform=transform,
        )
        datasets_state.set(datasets)
        active_key_state.set(key)
        return key

    @reactive.effect
    @reactive.event(input.preview_clean_btn)
    def _preview_cleaning() -> None:
        try:
            transformed, summary, extra_msgs = compute_cleaning_result()
            for level, msg in extra_msgs:
                push_message(level, msg)
            df = clean_active_df()
            action = input.clean_action()
            if action == "remove_duplicates" and df is not None:
                dupes = cleaning.get_duplicates(df)
                cleaning_preview_df.set(
                    dupes.head(20) if not dupes.empty else transformed.head(20)
                )
            else:
                cleaning_preview_df.set(transformed.head(20))
            cleaning_preview_meta.set(summary)

            is_row_removal = (
                action == "remove_duplicates"
                or (action == "handle_missing"
                    and input.clean_strategy() in ("drop_rows", "drop_cols", "knn"))
                or (action == "handle_outliers" and input.clean_outlier_action() == "remove")
            )
            if is_row_removal and df is not None:
                clean_comparison_fig.set(
                    build_rowcount_figure(len(df), len(transformed),
                                         action.replace("_", " ").title())
                )
            else:
                if action == "handle_outliers":
                    col = input.clean_single_column()
                else:
                    cols = list(input.clean_columns() or [])
                    col = cols[0] if cols else None
                if (col and df is not None and col in df.columns
                        and col in transformed.columns):
                    clean_comparison_fig.set(
                        build_comparison_figure(df[col], transformed[col], col)
                    )
                else:
                    clean_comparison_fig.set(None)
            log_ab_event(
                "preview_clean",
                success=True,
                details=f"summary={summary}",
            )
            push_message("info", "Cleaning preview updated.")
        except Exception as exc:
            log_ab_event("preview_clean", success=False, details=str(exc))
            push_message("error", f"Cleaning preview failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_clean_btn)
    def _apply_cleaning() -> None:
        try:
            transformed, summary, extra_msgs = compute_cleaning_result()
            for level, msg in extra_msgs:
                push_message(level, msg)
            # Build descriptive key
            action = input.clean_action()
            columns = list(input.clean_columns() or [])
            method = None
            if action == "handle_missing":
                method = input.clean_strategy()
            elif action == "scale_columns":
                method = input.clean_scale_method()
            elif action == "encode_columns":
                method = input.clean_encode_method()
            elif action == "handle_outliers":
                method = input.clean_outlier_action()
                columns = [input.clean_single_column()]
            elif action == "standardize_text":
                method = input.clean_text_case()
            elif action == "coerce_types":
                method = input.clean_coerce_target()

            desc_key = generate_descriptive_key(action, columns or None, method)
            src_key = _tab_picker_key("clean_df_picker")
            desc_label = f"{desc_key.replace('_', ' ')} (from {src_key})"

            target_key = apply_transformed_result(
                transformed,
                mode=input.clean_save_mode(),
                prefix=desc_key,
                label=desc_label,
                transform=summary,
                source_key=src_key,
            )
            cleaning_preview_df.set(transformed.head(20))
            cleaning_preview_meta.set(summary)
            log_ab_event(
                "apply_clean",
                success=True,
                details=f"target_key={target_key}; summary={summary}",
            )
            push_message("success", f"Cleaning applied → [{target_key}].")
        except Exception as exc:
            log_ab_event("apply_clean", success=False, details=str(exc))
            push_message("error", f"Cleaning apply failed: {exc}")

    # ── Feature Engineering operations ──────────────────────────────────

    @output
    @render.ui
    def feature_custom_expr_columns():
        df = feature_active_df()
        if df is None:
            return ui.tags.small()
        cols = ", ".join(df.columns.tolist())
        return ui.div(
            {"style": "margin-top:6px;"},
            ui.tags.small(
                {"style": "color:#555;"},
                ui.strong("Available columns: "),
                cols,
            ),
        )

    def compute_feature_result() -> tuple[pd.DataFrame, str, dict]:
        df = feature_active_df()
        if df is None:
            raise ValueError("Load a dataset first.")

        method = input.feature_method()
        col2 = input.feature_col2() or None

        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            df, method,
            input.feature_col1() or None,
            col2=col2,
            bins=int(input.feature_bins()),
            new_column=input.feature_new_column() or None,
            labels=bool(input.feature_labels()),
            prefix=input.feature_prefix() or None,
            drop_first=bool(input.feature_drop_first()),
            strategy=input.feature_fill_strategy(),
            fill_value=coerce_text_value(input.feature_fill_value()),
            expr=input.feature_custom_expr() if method == "custom_expr" else None,
        )
        parts = [f"{meta['feature_type']} → columns: {meta['output_columns']}"]
        if meta.get("formula"):
            parts.append(f"Formula: {meta['formula']}")
        if meta.get("mean") is not None:
            parts.append(f"mean={meta['mean']:.4f}, std={meta['std']:.4f}")
        if meta.get("min") is not None and meta.get("max") is not None:
            parts.append(f"min={meta['min']:.4f}, max={meta['max']:.4f}")
        if meta.get("fill_value_used") is not None:
            parts.append(f"fill_value={meta['fill_value_used']}")
        if meta.get("n_zero_denominator"):
            parts.append(f"zero-denominator rows: {meta['n_zero_denominator']}")
        if meta.get("rows_removed") is not None:
            parts.append(f"rows removed: {meta['rows_removed']}")
        return transformed, " | ".join(parts), meta

    @reactive.effect
    @reactive.event(input.preview_feature_btn)
    def _preview_feature() -> None:
        try:
            transformed, summary, meta = compute_feature_result()
            feature_preview_df.set(transformed.head(20))
            feature_preview_meta.set(summary)
            df = feature_active_df()
            input_col = (meta.get("input_columns") or [None])[0]
            output_cols = meta.get("output_columns", [])
            out_col = output_cols[0] if output_cols else None
            if (input_col and out_col and df is not None
                    and input_col in df.columns and out_col in transformed.columns):
                feature_comparison_fig.set(
                    build_comparison_figure(df[input_col], transformed[out_col],
                                            f"{input_col} → {out_col}")
                )
            else:
                feature_comparison_fig.set(None)
            if (input_col and out_col and df is not None
                    and input_col in df.columns and out_col in transformed.columns
                    and pd.api.types.is_numeric_dtype(df[input_col])
                    and pd.api.types.is_numeric_dtype(transformed[out_col])):
                b = df[input_col].dropna()
                a = transformed[out_col].dropna()
                push_message("info",
                    f"Before: mean={b.mean():.3f}, std={b.std():.3f} | "
                    f"After: mean={a.mean():.3f}, std={a.std():.3f}")
            push_message("info",
                "Feature preview updated. To undo, switch to a previous version in Load tab.")
        except Exception as exc:
            push_message("error", f"Feature preview failed: {exc}")

    @reactive.effect
    @reactive.event(input.apply_feature_btn)
    def _apply_feature() -> None:
        try:
            transformed, summary, meta = compute_feature_result()
            method = input.feature_method()
            col1 = input.feature_col1() or None
            col2 = input.feature_col2() or None
            new_col = input.feature_new_column() or None

            # Build descriptive prefix
            if method == "custom_expr":
                cols_for_key = [new_col] if new_col else None
            elif method in ("interaction", "ratio"):
                cols_for_key = [c for c in [col1, col2] if c]
            else:
                cols_for_key = [col1] if col1 else (
                    [new_col] if new_col else meta.get("output_columns")
                )
            desc_key = generate_descriptive_key(method, cols_for_key, None)
            src_key = _tab_picker_key("feature_df_picker")
            desc_label = f"{desc_key.replace('_', ' ')} (from {src_key})"

            target_key = apply_transformed_result(
                transformed,
                mode=input.feature_save_mode(),
                prefix=desc_key,
                label=desc_label,
                transform=summary,
                source_key=src_key,
            )
            feature_preview_df.set(transformed.head(20))
            feature_preview_meta.set(summary)
            push_message("success", f"Feature engineering applied → [{target_key}].")
        except Exception as exc:
            push_message("error", f"Feature apply failed: {exc}")

    # ── Filter ──────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.apply_filter_btn)
    def _apply_filter() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        expr = input.filter_expr().strip()
        if not expr:
            push_message("warning", "Enter a pandas query expression first.")
            return
        try:
            filtered = EDA.apply_filter(df, expr)
            src_key = _tab_picker_key("eda_df_picker")
            desc_key = generate_descriptive_key("filter", expr=expr)
            desc_label = f"{src_key} | {desc_key}"
            target_key = apply_transformed_result(
                filtered,
                mode=input.filter_save_mode(),
                prefix=desc_key,
                label=desc_label,
                transform=f"filter: {expr}",
                source_key=src_key,
            )
            push_message("success",
                         f"Filter applied → [{target_key}]. "
                         f"Rows: {len(df):,} → {len(filtered):,}.")
        except Exception as exc:
            push_message(
                "error",
                f"Filter failed: {exc}. "
                "Check syntax — wrap each condition in parentheses before combining with "
                "& or |, e.g. (\"col_cat\" == \"sex\") & (\"col_num\" >= 5). "
                "Use == for equality, backticks for column names with spaces."
            )

    # ── EDA plot handlers ───────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.render_1d_btn)
    def _render_1d() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        column = input.plot1d_column()
        if not column:
            push_message("warning", "Choose a column for the 1D plot.")
            return
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                payload = EDA.plot_numeric_1d(df, column=column,
                                               bins=int(input.plot1d_bins()),
                                               normalize=bool(input.plot1d_normalize()))
            else:
                payload = EDA.plot_categorical_1d(df, column=column,
                                                   normalize=bool(input.plot1d_normalize()))
            plot1d_payload.set(payload)
            if payload.get("status") == "warning":
                push_message("warning", payload.get("message", "1D plot rendered with warnings."))
            elif payload.get("status") == "error":
                push_message("error", payload.get("message", "1D plot failed."))
            else:
                push_message("success", "1D plot rendered.")
                col_data = df[column].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    plot1d_stats_text.set(
                        f"n={len(col_data):,}  |  mean={col_data.mean():.3f}  |  "
                        f"median={col_data.median():.3f}  |  std={col_data.std():.3f}  |  "
                        f"skew={col_data.skew():.3f}  |  kurtosis={col_data.kurtosis():.3f}"
                    )
                else:
                    vc = col_data.value_counts()
                    plot1d_stats_text.set(
                        f"n={len(col_data):,}  |  unique={vc.shape[0]}  |  "
                        f"top='{vc.index[0]}' ({vc.iloc[0]:,})"
                    )
        except Exception as exc:
            push_message("error", f"1D plot failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_2d_btn)
    def _render_2d() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            kind = input.plot2d_kind()
            payload = EDA.plot_two_columns(
                df, x=input.plot2d_x(), y=input.plot2d_y(),
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
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.regression_analysis(
                df, x=input.regression_x(), y=input.regression_y(),
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
                from scipy.stats import pearsonr
                common = df[[input.regression_x(), input.regression_y()]].dropna()
                if len(common) > 2:
                    r, p = pearsonr(common.iloc[:, 0], common.iloc[:, 1])
                    push_message("info",
                        f"Pearson r = {r:.4f}, R\u00b2 = {r**2:.4f}, "
                        f"p-value = {p:.2e} (n={len(common)})")
        except Exception as exc:
            push_message("error", f"Regression failed: {exc}")

    @reactive.effect
    @reactive.event(input.render_multiline_btn)
    def _render_multiline() -> None:
        df = eda_active_df()
        if df is None:
            push_message("warning", "Load a dataset first.")
            return
        try:
            payload = EDA.plot_multiline(
                df, column=input.multiline_value(),
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
        df = eda_active_df()
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

    # ── Download handlers ───────────────────────────────────────────────

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

    # ── Admin: A/B event log retrieval (password-gated) ─────────────────
    @output
    @render.ui
    def admin_download_area():
        if input.admin_pwd() != AB_ADMIN_PASSWORD:
            return ui.tags.small(
                {"class": "small-note"},
                "Enter the team password above to reveal the export link.",
            )
        if not AB_LOG_PATH.exists():
            return ui.tags.small(
                {"class": "small-note"},
                "Password accepted, but the event log does not exist yet "
                "(no sessions have logged any clean-tab events).",
            )
        return ui.download_button(
            "download_ab_events",
            f"Download ab_test_events.csv",
            class_="btn-dark",
        )

    @render.download(filename="ab_test_events.csv")
    def download_ab_events():
        if input.admin_pwd() != AB_ADMIN_PASSWORD:
            return
        if not AB_LOG_PATH.exists():
            return
        yield AB_LOG_PATH.read_bytes()

    # ── Render outputs ──────────────────────────────────────────────────

    @output
    @render.ui
    def message_stack():
        messages = messages_state.get()
        if not messages:
            return ui.div()
        _level_map = {"info": "alert-info", "success": "alert-success",
                      "warning": "alert-warning", "error": "alert-danger"}
        return ui.div(
            {"class": "alert-stack", "style": "padding: 0 12px;"},
            *[ui.div(
                {"class": f"alert {_level_map.get(item['level'], 'alert-secondary')} py-2 mb-1",
                 "role": "alert"},
                item["text"],
            ) for item in messages],
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
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Rows"),
                       ui.div({"class": "value"}, str(overview["n_rows"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Columns"),
                       ui.div({"class": "value"}, str(overview["n_cols"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Missing"),
                       ui.div({"class": "value"}, str(overview["n_missing"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Duplicates"),
                       ui.div({"class": "value"}, str(overview["n_duplicates"]))),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Source"),
                       ui.div({"class": "value"}, record["source_key"] or "-")),
                ui.div({"class": "metric"}, ui.div({"class": "label"}, "Transform"),
                       ui.div({"class": "value"}, record["transform"] or "-")),
            ),
        )

    @output
    @render.data_frame
    def history_table():
        return render.DataGrid(format_history_table(datasets_state.get()))

    # ── Overview outputs ────────────────────────────────────────────────

    @output
    @render.ui
    def overview_missing_content():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to see missing value summary.")
        info = cleaning.get_column_info(df)
        has_missing = (info[info["Missing"] > 0]
                       [["Column", "Missing", "Missing %"]]
                       .sort_values("Missing", ascending=False)
                       .reset_index(drop=True))
        if has_missing.empty:
            return ui.div(
                {"class": "alert alert-success py-2"},
                ui.tags.strong("No missing values detected in the current dataset.")
            )
        rows_html = [
            ui.tags.tr(
                ui.tags.td(str(r["Column"])),
                ui.tags.td(str(int(r["Missing"]))),
                ui.tags.td(f"{r['Missing %']:.1f}%"),
            )
            for _, r in has_missing.iterrows()
        ]
        return ui.div(
            ui.tags.p({"class": "small-note"},
                      f"{len(has_missing)} of {len(df.columns)} columns have missing values."),
            ui.tags.table(
                {"class": "table table-sm table-bordered table-hover"},
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("Column"),
                    ui.tags.th("Missing Count"),
                    ui.tags.th("Missing %"),
                )),
                ui.tags.tbody(*rows_html),
            ),
        )

    @output
    @render.ui
    def overview_duplicate_content():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to see duplicate summary.")
        dup_df = cleaning.get_duplicates(df)
        n_dup = len(dup_df)
        n_total = len(df)
        if n_dup == 0:
            return ui.div(
                {"class": "alert alert-success py-2"},
                ui.tags.strong("No duplicate detected in current dataset.")
            )
        example = dup_df.head(3)
        example_rows = [
            ui.tags.tr(*[ui.tags.td(str(v)) for v in row])
            for row in example.values
        ]
        return ui.div(
            ui.tags.p(
                {"class": "small-note"},
                f"{n_dup:,} duplicate rows found out of {n_total:,} total "
                f"({100 * n_dup / n_total:.1f}%). Showing up to 3 example rows:",
            ),
            ui.tags.table(
                {"class": "table table-sm table-bordered table-hover"},
                ui.tags.thead(ui.tags.tr(
                    *[ui.tags.th(str(c)) for c in example.columns]
                )),
                ui.tags.tbody(*example_rows),
            ),
            ui.tags.p(
                {"class": "small-note mt-2"},
                "Use Cleaning → Remove duplicates to eliminate these rows.",
            ),
        )

    @output
    @render.ui
    def overview_scale_content():
        df = current_df()
        if df is None:
            return ui.div({"class": "small-note"}, "Load a dataset to see scale summary.")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return ui.div({"class": "small-note"}, "No numeric columns in this dataset.")
        rows_html = []
        for col in num_cols:
            s = df[col].dropna()
            if s.empty:
                rows_html.append(ui.tags.tr(
                    ui.tags.td(col), ui.tags.td("—"), ui.tags.td("—"), ui.tags.td("—"),
                ))
            else:
                rows_html.append(ui.tags.tr(
                    ui.tags.td(col),
                    ui.tags.td(f"{s.min():.4g}"),
                    ui.tags.td(f"{s.max():.4g}"),
                    ui.tags.td(f"{s.mean():.4g}"),
                ))
        return ui.div(
            ui.tags.p({"class": "small-note"}, f"{len(num_cols)} numeric columns."),
            ui.tags.table(
                {"class": "table table-sm table-bordered table-hover"},
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("Column"),
                    ui.tags.th("Min"),
                    ui.tags.th("Max"),
                    ui.tags.th("Mean"),
                )),
                ui.tags.tbody(*rows_html),
            ),
            ui.tags.p(
                {"class": "small-note mt-2"},
                "Large scale differences across columns may affect k-NN imputation and "
                "distance-based algorithms. Consider scaling before applying such methods.",
            ),
        )

    # ── EDA outputs ─────────────────────────────────────────────────────

    @output
    @render.data_frame
    def head_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        payload = EDA.show_head(df, n=int(input.head_rows()))
        return render.DataGrid(dataframe_from_payload(payload))

    @output
    @render.data_frame
    def describe_num_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Note": ["No numeric columns in this dataset."]})
            )
        desc = num_df.describe().T.reset_index().rename(columns={"index": "column"})
        # Round for display
        for c in desc.columns:
            if c != "column":
                desc[c] = desc[c].apply(lambda v: round(v, 4) if pd.notnull(v) else v)
        return render.DataGrid(desc)

    @output
    @render.data_frame
    def describe_cat_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        cat_df = df.select_dtypes(exclude="number")
        if cat_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Note": ["No categorical columns in this dataset."]})
            )
        desc = cat_df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
        keep = [c for c in ["column", "count", "unique", "top", "freq"] if c in desc.columns]
        return render.DataGrid(desc[keep])

    @output
    @render.data_frame
    def column_types_table():
        df = eda_active_df()
        if df is None:
            return render.DataGrid(
                pd.DataFrame(columns=["column", "dtype", "is_numeric", "is_categorical"])
            )
        return render.DataGrid(current_column_types(df))

    # ── Cleaning outputs ─────────────────────────────────────────────────

    @output
    @render.ui
    def cleaning_result_summary():
        text = cleaning_preview_meta.get()
        return ui.div({"class": "small-note"}, text or "Preview a cleaning action to see a summary.")

    @output
    @render.data_frame
    def cleaning_preview_table():
        return render.DataGrid(cleaning_preview_df.get())

    # ── Feature Engineering outputs ──────────────────────────────────────

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
            "square": "Squares values (x²). Amplifies differences between large and small values.",
            "cube": "Cubes values (x³). Captures cubic relationships, preserves sign.",
            "interaction": "Multiplies two columns (x * y). Captures combined effects.",
            "ratio": "Divides col1 by col2. Useful for per-unit metrics (e.g., price per sqft).",
            "binning": "Groups continuous values into discrete bins using equal-width intervals.",
            "one_hot": "Creates binary 0/1 columns for each category. Required by most ML models.",
            "standardize": "Z-score normalization: (x - mean) / std. Centers data at 0.",
            "normalize": "Min-Max scaling to [0, 1]. Preserves shape, bounds values.",
            "fillna": "Replaces missing values with a computed or constant value.",
            "dropna": "Removes rows containing missing values in the selected column.",
            "custom_expr": (
                "Evaluates a custom algebraic expression to create a new column. "
                "Enter any pandas-eval expression referencing column names."
            ),
        }
        text = explanations.get(method, "")
        if not text:
            return ui.div()
        return ui.div({"class": "tip-box", "style": "margin-top: 8px;"},
                      ui.tags.small(text))

    @output
    @render.data_frame
    def feature_preview_table():
        return render.DataGrid(feature_preview_df.get())

    # ── Plot outputs ─────────────────────────────────────────────────────

    @output
    @render_plotly
    def plot_1d():
        payload = plot1d_payload.get()
        if payload is None:
            return empty_figure("Render a 1D plot to see output here.")
        fig = figure_from_payload(payload)
        if input.plot1d_logx():
            fig.update_xaxes(type="log")
        if input.plot1d_logy():
            fig.update_yaxes(type="log")
        return fig

    @output
    @render.ui
    def plot1d_stats():
        text = plot1d_stats_text.get()
        if not text:
            return ui.div()
        return ui.div({"class": "small-note", "style": "margin-top:6px;"}, text)

    @output
    @render_plotly
    def plot_2d():
        payload = plot2d_payload.get()
        if payload is None:
            return empty_figure("Render a 2D plot to see output here.")
        fig = figure_from_payload(payload)
        if input.plot2d_logx():
            fig.update_xaxes(type="log")
        if input.plot2d_logy():
            fig.update_yaxes(type="log")
        return fig

    @output
    @render_plotly
    def plot_regression():
        payload = regression_payload.get()
        if payload is None:
            return empty_figure("Render a regression plot to see output here.")
        fig = figure_from_payload(payload)
        if input.regression_logx():
            fig.update_xaxes(type="log")
        return fig

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
