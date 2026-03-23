"""
Project 2 - Part 2: Data Cleaning and Preprocessing
Pure-function backend module (DataFrame → DataFrame)

Refactored per architecture review:
 
All cleaning logic extracted as stateless pure functions

No UI, no Dash, no state management

Each function: pd.DataFrame in → pd.DataFrame (or summary dict) out

Ready for orchestration via API layer / pipeline
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Optional, Union



#  1. Data Loading


def load_builtin_iris() -> pd.DataFrame:
    """Load the built-in Iris dataset."""
    return sns.load_dataset("iris")


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(filepath, **kwargs)


def load_excel(filepath: str, **kwargs) -> pd.DataFrame:
    """Load an Excel file (.xlsx / .xls) into a DataFrame."""
    return pd.read_excel(filepath, **kwargs)


def load_json(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a JSON file into a DataFrame."""
    return pd.read_json(filepath, **kwargs)



#  2. Data Overview / Inspection


def get_overview(df: pd.DataFrame) -> dict:
    """
    Return a summary dict with basic dataset information.

    Returns
    
    dict with keys:
        n_rows, n_cols, n_missing, n_duplicates,
        numeric_columns, categorical_columns
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_missing": int(df.isnull().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising each column's dtype,
    non-null count, missing count / percentage, and unique count.
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / max(len(df), 1) * 100).round(2),
        "Unique": df.nunique().values,
    })


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns."""
    return (
        df.describe(include="all")
        .T
        .reset_index()
        .rename(columns={"index": "Column"})
    )



#  3. Missing-Value Handling


def handle_missing(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    strategy: str = "drop_rows",
    constant_value: Optional[str] = None,
) -> pd.DataFrame:
    """
    Handle missing values and return a cleaned DataFrame.

    Parameters
    
    df : DataFrame
    columns : list of column names to operate on (None = all columns)
    strategy : one of
        'drop_rows'  – drop rows that have NaN in *columns*
        'drop_cols'  – drop columns (from *columns*) that contain any NaN
        'mean'       – fill NaN with column mean  (numeric only)
        'median'     – fill NaN with column median (numeric only)
        'mode'       – fill NaN with column mode
        'constant'   – fill NaN with *constant_value*
    constant_value : value used when strategy='constant'

    Returns
    
    pd.DataFrame – a new DataFrame with missing values handled
    """
    df = df.copy()
    cols = columns if columns else df.columns.tolist()

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
        val = constant_value if constant_value is not None else "0"
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna(val)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df.reset_index(drop=True)



#  4. Duplicate Handling


def get_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return all duplicate rows (including all copies)."""
    return df[df.duplicated(keep=False)]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and return a new DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)



#  5. Scaling / Normalization


def scale_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> pd.DataFrame:
    """
    Scale numeric columns in-place and return a new DataFrame.

    Parameters
    
    columns : list of numeric column names
    method  : 'standard' | 'minmax' | 'robust'
    """
    df = df.copy()
    scaler_map = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scaler_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(scaler_map)}")

    scaler = scaler_map[method]()
    df[columns] = scaler.fit_transform(df[columns])
    return df



#  6. Categorical Encoding


def encode_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "label",
) -> pd.DataFrame:
    """
    Encode categorical columns and return a new DataFrame.

    Parameters
    
    columns : list of categorical column names
    method  : 'label' | 'onehot'
    """
    df = df.copy()
    if method == "label":
        le = LabelEncoder()
        for c in columns:
            if c in df.columns:
                df[c] = le.fit_transform(df[c].astype(str))
    elif method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'label' or 'onehot'.")
    return df



#  7. Outlier Handling


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> dict:
    """
    Detect outliers via the IQR method and return diagnostic info.

    Returns
    
    dict with keys:
        q1, q3, iqr, lower_bound, upper_bound,
        n_outliers, outlier_mask (boolean Series)
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": int(mask.sum()),
        "outlier_mask": mask,
    }


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    action: str = "remove",
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Handle outliers in a single column and return a new DataFrame.

    Parameters
    
    action : 'remove' – drop outlier rows
             'cap'    – winsorize (clip to bounds)
    """
    df = df.copy()
    info = detect_outliers(df, column, iqr_multiplier)
    lower, upper = info["lower_bound"], info["upper_bound"]

    if action == "remove":
        df = df[(df[column] >= lower) & (df[column] <= upper)].reset_index(drop=True)
    elif action == "cap":
        df[column] = df[column].clip(lower, upper)
    else:
        raise ValueError(f"Unknown action: {action}. Choose 'remove' or 'cap'.")
    return df



#  8. Export


def export_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save the DataFrame to a CSV file."""
    df.to_csv(filepath, index=False, **kwargs)



#  Pipeline helper (optional convenience)


def run_pipeline(
    df: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """
    Execute a sequence of cleaning steps declaratively.

    Parameters
    
    steps : list of dicts, each with 'action' and relevant params.
        Example:
        [
            {"action": "handle_missing", "strategy": "mean"},
            {"action": "remove_duplicates"},
            {"action": "scale_columns", "columns": ["col1"], "method": "standard"},
            {"action": "encode_columns", "columns": ["species"], "method": "onehot"},
            {"action": "handle_outliers", "column": "col1", "action_type": "cap"},
        ]
    """
    dispatch = {
        "handle_missing": lambda d, p: handle_missing(d, **{k: v for k, v in p.items() if k != "action"}),
        "remove_duplicates": lambda d, _: remove_duplicates(d),
        "scale_columns": lambda d, p: scale_columns(d, **{k: v for k, v in p.items() if k != "action"}),
        "encode_columns": lambda d, p: encode_columns(d, **{k: v for k, v in p.items() if k != "action"}),
        "handle_outliers": lambda d, p: handle_outliers(
            d,
            column=p["column"],
            action=p.get("action_type", "remove"),
            iqr_multiplier=p.get("iqr_multiplier", 1.5),
        ),
    }

    for step in steps:
        action = step["action"]
        if action not in dispatch:
            raise ValueError(f"Unknown pipeline action: {action}")
        df = dispatch[action](df, step)
    return df



#  demo

if __name__ == "__main__":

    raw = load_builtin_iris()
    print("Raw shape:", raw.shape)
    print("Overview:", get_overview(raw))

    cleaned = run_pipeline(raw, [
        {"action": "remove_duplicates"},
        {"action": "handle_missing", "strategy": "mean"},
        {"action": "scale_columns", "columns": ["sepal_length", "sepal_width",
                                                  "petal_length", "petal_width"],
         "method": "standard"},
        {"action": "encode_columns", "columns": ["species"], "method": "label"},
    ])
    print("Cleaned shape:", cleaned.shape)
    print(cleaned.head())