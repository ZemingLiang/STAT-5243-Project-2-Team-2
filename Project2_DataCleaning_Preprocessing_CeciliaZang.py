"""
Project 2 - Cecilia Zang - Part 2: Data Cleaning and Preprocessing
Interactive Dash Web Application (can run directly in Spyder)
Dataset: Iris (seaborn built-in)
Targets Advanced level (8pt): interactive interface, dynamic preprocessing, real-time feedback
"""


# Imports

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import seaborn as sns
import base64
import io
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


# Initialize App

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "Data Cleaning & Preprocessing"


# Helper: build a DataTable from a DataFrame

def make_table(df, table_id, page_size=12):
    """Return a Dash DataTable component from a pandas DataFrame."""
    if df is None or df.empty:
        return html.P("No data available.", className="text-muted")
    # Round floats for display
    display_df = df.copy()
    for c in display_df.select_dtypes(include="float").columns:
        display_df[c] = display_df[c].round(4)
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": str(c), "id": str(c)} for c in display_df.columns],
        data=display_df.to_dict("records"),
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
        ],
        filter_action="native",
        sort_action="native",
    )


# Layout: Tabs


#  Tab 0: User Guide 
tab_guide = dbc.Card(dbc.CardBody([
    html.H3("Welcome to the Data Cleaning & Preprocessing App", className="mb-3"),
    dcc.Markdown("""
This application helps you interactively clean, preprocess, and transform datasets.

### How to Use
1. Load Data: Upload your own CSV / Excel / JSON file, or use the built-in *Iris* dataset.
2. Data Overview: Inspect shape, types, missing values, and duplicates at a glance.
3. Handle Missing Values: Choose from removal or multiple imputation strategies (mean, median, mode, constant).
4. Handle Duplicates: Detect and remove duplicate rows with one click.
5. Scaling / Normalization: Apply Standard, Min-Max, or Robust scaling to numeric columns.
6. Encode Categoricals: Apply Label Encoding or One-Hot Encoding to categorical columns.
7. Outlier Handling: Detect outliers via IQR and choose to remove or cap them.
8. Download: Export the cleaned dataset as CSV at any stage.

> Every section shows before → after feedback so you can see the impact of each operation in real time.
"""),
]), className="mt-3")

#  Tab 1: Load Data 
tab_load = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Data Source"),
            dbc.RadioItems(
                id="data-source",
                options=[
                    {"label": "Built-in Iris Dataset", "value": "builtin"},
                    {"label": "Upload Your Own File", "value": "upload"},
                ],
                value="builtin",
                className="mb-3",
            ),
            dcc.Upload(
                id="file-upload",
                children=html.Div(["Drag & Drop or ", html.A("Browse Files", style={"color": "#2c3e50", "fontWeight": "bold"})]),
                style={
                    "width": "100%", "height": "80px", "lineHeight": "80px",
                    "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
                    "textAlign": "center", "borderColor": "#adb5bd",
                },
                multiple=False,
            ),
            html.Small("Supported: CSV, Excel (.xlsx/.xls), JSON", className="text-muted"),
            html.Br(),
            dbc.Button("Load Dataset", id="btn-load", color="primary", className="mt-3 w-100"),
        ], width=4),
        dbc.Col([
            html.H5("Dataset Preview"),
            html.Div(id="load-status", className="mb-2"),
            html.Div(id="preview-table"),
        ], width=8),
    ]),
]), className="mt-3")

#  Tab 2: Data Overview 
tab_overview = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([html.H5("Basic Info"), html.Div(id="info-summary")], width=4),
        dbc.Col([html.H5("Column Types & Missing Values"), html.Div(id="col-info-table")], width=8),
    ], className="mb-4"),
    html.H5("Descriptive Statistics"),
    html.Div(id="desc-stats"),
]), className="mt-3")

#  Tab 3: Missing Values 
tab_missing = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Settings"),
            html.Label("Apply to Column"),
            dcc.Dropdown(id="missing-col", className="mb-2"),
            html.Label("Imputation Strategy"),
            dcc.Dropdown(
                id="missing-strategy",
                options=[
                    {"label": "Drop rows with missing values", "value": "drop_rows"},
                    {"label": "Drop columns with missing values", "value": "drop_cols"},
                    {"label": "Fill with Mean (numeric)", "value": "mean"},
                    {"label": "Fill with Median (numeric)", "value": "median"},
                    {"label": "Fill with Mode", "value": "mode"},
                    {"label": "Fill with Constant", "value": "constant"},
                ],
                value="drop_rows",
                className="mb-2",
            ),
            dbc.Input(id="fill-constant", placeholder="Constant value", type="text", className="mb-3"),
            dbc.Button("Apply", id="btn-missing", color="warning", className="w-100"),
        ], width=3),
        dbc.Col([html.H5("Before"), html.Div(id="missing-before")], width=4),
        dbc.Col([html.H5("After"), html.Div(id="missing-feedback", className="mb-2"), html.Div(id="missing-after")], width=5),
    ]),
]), className="mt-3")

#  Tab 4: Duplicates 
tab_dup = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.Div(id="dup-info", className="mb-3"),
            dbc.Button("Remove Duplicates", id="btn-dup", color="warning", className="w-100"),
            html.Div(id="dup-feedback", className="mt-3"),
        ], width=3),
        dbc.Col([html.H5("Duplicate Rows"), html.Div(id="dup-table")], width=9),
    ]),
]), className="mt-3")

#  Tab 5: Scaling 
tab_scale = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Settings"),
            html.Label("Select Numeric Columns"),
            dcc.Dropdown(id="scale-cols", multi=True, className="mb-2"),
            html.Label("Scaling Method"),
            dcc.Dropdown(
                id="scale-method",
                options=[
                    {"label": "StandardScaler (z-score)", "value": "standard"},
                    {"label": "MinMaxScaler (0–1)", "value": "minmax"},
                    {"label": "RobustScaler (IQR-based)", "value": "robust"},
                ],
                value="standard",
                className="mb-3",
            ),
            dbc.Button("Apply Scaling", id="btn-scale", color="success", className="w-100"),
        ], width=3),
        dbc.Col([html.H5("Before Scaling"), html.Div(id="scale-before")], width=4),
        dbc.Col([html.H5("After Scaling"), html.Div(id="scale-after")], width=5),
    ]),
]), className="mt-3")

#  Tab 6: Encoding 
tab_encode = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Settings"),
            html.Label("Select Categorical Columns"),
            dcc.Dropdown(id="encode-cols", multi=True, className="mb-2"),
            html.Label("Encoding Method"),
            dcc.Dropdown(
                id="encode-method",
                options=[
                    {"label": "Label Encoding", "value": "label"},
                    {"label": "One-Hot Encoding", "value": "onehot"},
                ],
                value="label",
                className="mb-3",
            ),
            dbc.Button("Apply Encoding", id="btn-encode", color="info", className="w-100"),
        ], width=3),
        dbc.Col([html.H5("Before Encoding"), html.Div(id="encode-before")], width=4),
        dbc.Col([html.H5("After Encoding"), html.Div(id="encode-after")], width=5),
    ]),
]), className="mt-3")

#  Tab 7: Outliers 
tab_outlier = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Settings"),
            html.Label("Select Column"),
            dcc.Dropdown(id="outlier-col", className="mb-2"),
            html.Label("IQR Multiplier"),
            dbc.Input(id="iqr-mult", type="number", value=1.5, min=0.5, max=5, step=0.1, className="mb-2"),
            html.Label("Action"),
            dcc.Dropdown(
                id="outlier-action",
                options=[
                    {"label": "Remove Outlier Rows", "value": "remove"},
                    {"label": "Cap / Winsorize", "value": "cap"},
                ],
                value="remove",
                className="mb-3",
            ),
            dbc.Button("Apply", id="btn-outlier", color="danger", className="w-100"),
        ], width=3),
        dbc.Col([html.H5("Outlier Detection"), html.Div(id="outlier-info"), html.Div(id="outlier-table")], width=4),
        dbc.Col([html.H5("After Handling"), html.Div(id="outlier-feedback"), html.Div(id="outlier-after")], width=5),
    ]),
]), className="mt-3")

#  Tab 8: Download 
tab_download = dbc.Card(dbc.CardBody([
    html.H5("Download Cleaned Dataset"),
    html.Div(id="download-summary", className="mb-3"),
    dbc.Button("Download as CSV", id="btn-download", color="primary"),
    dcc.Download(id="download-csv"),
]), className="mt-3")

#  Main Layout 
app.layout = dbc.Container([
    html.H2("🧹 Data Cleaning & Preprocessing", className="mt-3 mb-3",
            style={"fontWeight": "bold", "color": "#2c3e50"}),
    # Hidden store for the working dataframe (JSON)
    dcc.Store(id="store-data", storage_type="memory"),
    dbc.Tabs([
        dbc.Tab(tab_guide,   label="📖 User Guide"),
        dbc.Tab(tab_load,    label="1️⃣ Load Data"),
        dbc.Tab(tab_overview, label="2️⃣ Overview"),
        dbc.Tab(tab_missing, label="3️⃣ Missing Values"),
        dbc.Tab(tab_dup,     label="4️⃣ Duplicates"),
        dbc.Tab(tab_scale,   label="5️⃣ Scaling"),
        dbc.Tab(tab_encode,  label="6️⃣ Encoding"),
        dbc.Tab(tab_outlier, label="7️⃣ Outliers"),
        dbc.Tab(tab_download, label="⬇️ Download"),
    ]),
], fluid=True)



# Callbacks


#  1. Load Data 
@callback(
    Output("store-data", "data"),
    Output("load-status", "children"),
    Output("preview-table", "children"),
    Input("btn-load", "n_clicks"),
    State("data-source", "value"),
    State("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True,
)
def load_data(n, source, contents, filename):
    if source == "builtin":
        df = sns.load_dataset("iris")
    else:
        if contents is None:
            return dash.no_update, dbc.Alert("Please upload a file first.", color="warning"), dash.no_update
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        fname = filename.lower()
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif fname.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(decoded))
            elif fname.endswith(".json"):
                df = pd.read_json(io.StringIO(decoded.decode("utf-8")))
            else:
                return dash.no_update, dbc.Alert("Unsupported file format.", color="danger"), dash.no_update
        except Exception as e:
            return dash.no_update, dbc.Alert(f"Error reading file: {e}", color="danger"), dash.no_update

    status = dbc.Alert(f"✅ Loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns", color="success")
    table = make_table(df.head(20), "tbl-preview")
    return df.to_json(date_format="iso", orient="split"), status, table


#  2. Data Overview 
@callback(
    Output("info-summary", "children"),
    Output("col-info-table", "children"),
    Output("desc-stats", "children"),
    # Also populate dropdown options for other tabs
    Output("missing-col", "options"),
    Output("missing-col", "value"),
    Output("scale-cols", "options"),
    Output("scale-cols", "value"),
    Output("encode-cols", "options"),
    Output("encode-cols", "value"),
    Output("outlier-col", "options"),
    Input("store-data", "data"),
)
def update_overview(json_data):
    if json_data is None:
        empty = html.P("Load a dataset first.", className="text-muted")
        return empty, empty, empty, [], None, [], [], [], [], []
    df = pd.read_json(io.StringIO(json_data), orient="split")

    # Basic info
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_dup = int(df.duplicated().sum())
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    info = html.Ul([
        html.Li(f"Rows: {n_rows}"),
        html.Li(f"Columns: {n_cols}"),
        html.Li(f"Numeric columns: {len(num_cols)}"),
        html.Li(f"Categorical columns: {len(cat_cols)}"),
        html.Li(f"Total missing values: {n_missing}"),
        html.Li(f"Duplicate rows: {n_dup}"),
    ])

    # Column info table
    col_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / max(len(df), 1) * 100).round(2),
        "Unique": df.nunique().values,
    })
    col_tbl = make_table(col_df, "tbl-col-info")

    # Descriptive stats
    desc = df.describe(include="all").T.reset_index().rename(columns={"index": "Column"})
    desc_tbl = make_table(desc, "tbl-desc")

    # Dropdown options
    all_cols = [{"label": "All Columns", "value": "__all__"}] + [{"label": c, "value": c} for c in df.columns]
    num_opts = [{"label": c, "value": c} for c in num_cols]
    cat_opts = [{"label": c, "value": c} for c in cat_cols]
    outlier_opts = [{"label": c, "value": c} for c in num_cols]

    return (info, col_tbl, desc_tbl,
            all_cols, "__all__",
            num_opts, num_cols,
            cat_opts, cat_cols,
            outlier_opts)


#  3. Missing Values 
@callback(
    Output("store-data", "data", allow_duplicate=True),
    Output("missing-before", "children"),
    Output("missing-after", "children"),
    Output("missing-feedback", "children"),
    Input("btn-missing", "n_clicks"),
    State("store-data", "data"),
    State("missing-col", "value"),
    State("missing-strategy", "value"),
    State("fill-constant", "value"),
    prevent_initial_call=True,
)
def handle_missing(n, json_data, col_sel, strategy, constant):
    if json_data is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")

    # Before summary
    before_df = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / max(len(df), 1) * 100).round(2),
    })
    before_tbl = make_table(before_df, "tbl-miss-before", page_size=8)

    # Apply strategy
    cols = df.columns.tolist() if col_sel == "__all__" else [col_sel]
    if strategy == "drop_rows":
        df.dropna(subset=cols, inplace=True)
    elif strategy == "drop_cols":
        to_drop = [c for c in cols if df[c].isnull().any()]
        df.drop(columns=to_drop, inplace=True, errors="ignore")
    elif strategy == "mean":
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].mean())
    elif strategy == "median":
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
    elif strategy == "mode":
        for c in cols:
            if c in df.columns and not df[c].mode().empty:
                df[c] = df[c].fillna(df[c].mode()[0])
    elif strategy == "constant":
        val = constant if constant else "0"
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna(val)
    df.reset_index(drop=True, inplace=True)

    # After summary
    after_df = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / max(len(df), 1) * 100).round(2),
    })
    after_tbl = make_table(after_df, "tbl-miss-after", page_size=8)
    total = int(df.isnull().sum().sum())
    color = "success" if total == 0 else "warning"
    feedback = dbc.Alert(f"Remaining missing values: {total}", color=color)

    return df.to_json(date_format="iso", orient="split"), before_tbl, after_tbl, feedback


#  4. Duplicates 
@callback(
    Output("dup-info", "children"),
    Output("dup-table", "children"),
    Input("store-data", "data"),
)
def show_duplicates(json_data):
    if json_data is None:
        return html.P("Load data first."), html.P("")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    n = int(df.duplicated().sum())
    color = "success" if n == 0 else "danger"
    info = dbc.Alert(f"Duplicate rows found: {n}", color=color)
    dups = df[df.duplicated(keep=False)]
    if dups.empty:
        tbl = html.P("No duplicates found.", className="text-muted")
    else:
        tbl = make_table(dups.head(50), "tbl-dup")
    return info, tbl

@callback(
    Output("store-data", "data", allow_duplicate=True),
    Output("dup-feedback", "children"),
    Input("btn-dup", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def remove_duplicates(n, json_data):
    if json_data is None:
        return dash.no_update, dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    feedback = dbc.Alert(f"Removed {removed} duplicate rows. Current rows: {len(df)}", color="success")
    return df.to_json(date_format="iso", orient="split"), feedback


#  5. Scaling 
@callback(
    Output("scale-before", "children"),
    Input("store-data", "data"),
)
def scale_before(json_data):
    if json_data is None:
        return html.P("Load data first.")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return html.P("No numeric columns.")
    desc = df[num_cols].describe().T.reset_index().rename(columns={"index": "Column"})
    return make_table(desc, "tbl-scale-b")
@callback(
    Output("store-data", "data", allow_duplicate=True),
    Output("scale-after", "children"),
    Input("btn-scale", "n_clicks"),
    State("store-data", "data"),
    State("scale-cols", "value"),
    State("scale-method", "value"),
    prevent_initial_call=True,
)
def apply_scaling(n, json_data, cols, method):
    if json_data is None or not cols:
        return dash.no_update, html.P("Select columns first.")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    scaler_map = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
    scaler = scaler_map[method]
    df[cols] = scaler.fit_transform(df[cols])
    desc = df[cols].describe().T.reset_index().rename(columns={"index": "Column"})
    return df.to_json(date_format="iso", orient="split"), make_table(desc, "tbl-scale-a")


#  6. Encoding 
@callback(
    Output("encode-before", "children"),
    Input("store-data", "data"),
)
def encode_before(json_data):
    if json_data is None:
        return html.P("Load data first.")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if not cat_cols:
        return html.P("No categorical columns.")
    summary = pd.DataFrame({
        "Column": cat_cols,
        "Unique": [df[c].nunique() for c in cat_cols],
        "Sample Values": [", ".join(df[c].dropna().unique().astype(str)[:5]) for c in cat_cols],
    })
    return make_table(summary, "tbl-enc-b")

@callback(
    Output("store-data", "data", allow_duplicate=True),
    Output("encode-after", "children"),
    Input("btn-encode", "n_clicks"),
    State("store-data", "data"),
    State("encode-cols", "value"),
    State("encode-method", "value"),
    prevent_initial_call=True,
)
def apply_encoding(n, json_data, cols, method):
    if json_data is None or not cols:
        return dash.no_update, html.P("Select columns first.")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    if method == "label":
        le = LabelEncoder()
        for c in cols:
            if c in df.columns:
                df[c] = le.fit_transform(df[c].astype(str))
    elif method == "onehot":
        df = pd.get_dummies(df, columns=cols, drop_first=False, dtype=int)
    return df.to_json(date_format="iso", orient="split"), make_table(df.head(20), "tbl-enc-a")


#  7. Outliers 
@callback(
    Output("outlier-info", "children"),
    Output("outlier-table", "children"),
    Input("store-data", "data"),
    Input("outlier-col", "value"),
    Input("iqr-mult", "value"),
)
def detect_outliers(json_data, col, mult):
    if json_data is None or not col:
        return html.P("Load data and select a column."), html.P("")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return html.P("Select a numeric column."), html.P("")
    mult = mult if mult else 1.5
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - mult * iqr
    upper = q3 + mult * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    n_out = int(mask.sum())
    info = html.Div([
        html.P(f"Q1 = {q1:.4f},  Q3 = {q3:.4f},  IQR = {iqr:.4f}"),
        html.P(f"Lower bound = {lower:.4f},  Upper bound = {upper:.4f}"),
        html.P([html.Strong(f"Outliers detected: {n_out}")],
               style={"color": "red" if n_out else "green"}),
    ])
    outliers = df[mask]
    if outliers.empty:
        tbl = html.P("No outliers detected.", className="text-muted")
    else:
        tbl = make_table(outliers.head(50), "tbl-out")
    return info, tbl

@callback(
    Output("store-data", "data", allow_duplicate=True),
    Output("outlier-feedback", "children"),
    Output("outlier-after", "children"),
    Input("btn-outlier", "n_clicks"),
    State("store-data", "data"),
    State("outlier-col", "value"),
    State("iqr-mult", "value"),
    State("outlier-action", "value"),
    prevent_initial_call=True,
)
def handle_outliers(n, json_data, col, mult, action):
    if json_data is None or not col:
        return dash.no_update, dash.no_update, dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")
    mult = mult if mult else 1.5
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - mult * iqr
    upper = q3 + mult * iqr

    if action == "remove":
        df = df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)
    else:
        df[col] = df[col].clip(lower, upper)

    # Recount
    new_mask = (df[col] < lower) | (df[col] > upper)
    n_out = int(new_mask.sum())
    feedback = dbc.Alert(f"Remaining outliers in '{col}': {n_out}  |  Rows: {len(df)}", color="success" if n_out == 0 else "warning")
    return df.to_json(date_format="iso", orient="split"), feedback, make_table(df.head(20), "tbl-out-after")


#  8. Download 
@callback(
    Output("download-summary", "children"),
    Input("store-data", "data"),
)
def download_summary(json_data):
    if json_data is None:
        return html.P("No data to download.", className="text-muted")
    df = pd.read_json(io.StringIO(json_data), orient="split")
    return html.Ul([
        html.Li(f"Current dataset: {df.shape[0]} rows × {df.shape[1]} columns"),
        html.Li(f"Missing values: {int(df.isnull().sum().sum())}"),
        html.Li(f"Duplicates: {int(df.duplicated().sum())}"),
    ])

@callback(
    Output("download-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True,
)
def download_csv(n, json_data):
    if json_data is None:
        return dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient="split")
    return dcc.send_data_frame(df.to_csv, "cleaned_data.csv", index=False)




# Run — works directly in Spyder!

if __name__ == "__main__":
    print("Starting app at http://127.0.0.1:8050")
    print("Open the link above in your browser.")
    app.run(debug=False, port=8050)
