"""
Project 2 - Part 2: Data Cleaning and Preprocessing
====================================================

Pure-function backend module providing a complete data-cleaning toolkit.

Every public function follows a **stateless, pure-function** contract:

    pd.DataFrame in  -->  pd.DataFrame (or summary dict) out

This means there is no UI code, no Dash/Shiny state, and no side-effects
beyond the returned object.  The module is designed to be orchestrated by
an external API layer, interactive notebook, or the built-in
``run_pipeline`` convenience function.

Sections
--------
1. Data Loading        – CSV, Excel, JSON, RDS, built-in Iris
2. Data Inspection     – shape, dtypes, missing counts, descriptive stats
3. Missing-Value Handling – drop or impute (mean / median / mode / constant)
4. Duplicate Handling  – detect and remove duplicate rows
5. Scaling             – StandardScaler, MinMaxScaler, RobustScaler
6. Categorical Encoding – LabelEncoder, one-hot (get_dummies)
7. Outlier Handling    – IQR-based detection, removal, and winsorization
8. Export              – write cleaned data to CSV
9. Pipeline Runner     – declarative step-by-step cleaning pipeline
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Optional, Union


# ---------------------------------------------------------------------------
#  1. Data Loading
# ---------------------------------------------------------------------------


def load_builtin_iris() -> pd.DataFrame:
    """Load the built-in Iris dataset from seaborn.

    Returns
    -------
    pd.DataFrame
        The classic 150-row Iris dataset with columns
        ``sepal_length``, ``sepal_width``, ``petal_length``,
        ``petal_width``, and ``species``.
    """
    return sns.load_dataset("iris")


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ``.csv`` file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    return pd.read_csv(filepath, **kwargs)


def load_excel(filepath: str, **kwargs) -> pd.DataFrame:
    """Load an Excel file (.xlsx / .xls) into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    return pd.read_excel(filepath, **kwargs)


def load_json(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a JSON file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ``.json`` file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_json``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    try:
        return pd.read_json(filepath, **kwargs)
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse JSON as a flat table: {exc}. "
            "Ensure the JSON file contains a flat array of records or a column-oriented object."
        ) from exc


# ---------------------------------------------------------------------------
#  2. Data Overview / Inspection
# ---------------------------------------------------------------------------


def get_overview(df: pd.DataFrame) -> dict:
    """Return a summary dict with basic dataset information.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_cols``, ``n_missing``, ``n_duplicates``,
        ``numeric_columns``, ``categorical_columns``.
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
    """Return a per-column summary of dtype, nulls, and unique counts.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        One row per column with fields ``Column``, ``Dtype``,
        ``Non-Null``, ``Missing``, ``Missing %``, and ``Unique``.
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        # Guard against zero-length DataFrames with max(..., 1)
        "Missing %": (df.isnull().sum().values / max(len(df), 1) * 100).round(2),
        "Unique": df.nunique().values,
    })


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Transposed output of ``df.describe(include='all')`` with an
        added ``Column`` field.
    """
    return (
        df.describe(include="all")
        .T
        .reset_index()
        .rename(columns={"index": "Column"})
    )


# ---------------------------------------------------------------------------
#  3. Missing-Value Handling
# ---------------------------------------------------------------------------


def handle_missing(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    strategy: str = "drop_rows",
    constant_value: Optional[str] = None,
) -> pd.DataFrame:
    """Handle missing values and return a cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str] or None
        Column names to operate on.  ``None`` means all columns.
    strategy : str
        One of:

        * ``'drop_rows'``  -- drop rows that have NaN in *columns*
        * ``'drop_cols'``  -- drop columns (from *columns*) that contain any NaN
        * ``'mean'``       -- fill NaN with column mean  (numeric only)
        * ``'median'``     -- fill NaN with column median (numeric only)
        * ``'mode'``       -- fill NaN with column mode
        * ``'constant'``   -- fill NaN with *constant_value*
    constant_value : str or None
        Value used when ``strategy='constant'``.  Defaults to ``"0"``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing values handled.
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
                fill_val = df[c].mean()
                if pd.notna(fill_val):
                    df[c] = df[c].fillna(fill_val)
                # Skip all-null columns where mean is NaN
    elif strategy == "median":
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                fill_val = df[c].median()
                if pd.notna(fill_val):
                    df[c] = df[c].fillna(fill_val)
    elif strategy == "mode":
        for c in cols:
            if c in df.columns and not df[c].mode().empty:
                # mode() returns a Series; take the first (most frequent) value
                df[c] = df[c].fillna(df[c].mode()[0])
    elif strategy == "constant":
        val = constant_value if constant_value is not None else "0"
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna(val)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
#  4. Duplicate Handling
# ---------------------------------------------------------------------------


def get_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return all duplicate rows, including every copy.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Subset of *df* where ``duplicated(keep=False)`` is True.
    """
    return df[df.duplicated(keep=False)]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame with a reset integer index.
    """
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
#  5. Scaling / Normalization
# ---------------------------------------------------------------------------


def scale_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> pd.DataFrame:
    """Scale numeric columns and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str]
        Numeric column names to scale.
    method : str
        Scaling algorithm to apply:

        * ``'standard'`` -- **StandardScaler**: centres each column to
          mean = 0, std = 1 using ``(x - mean) / std``.
        * ``'minmax'``   -- **MinMaxScaler**: rescales each column to
          the [0, 1] range via ``(x - min) / (max - min)``.
        * ``'robust'``   -- **RobustScaler**: centres using the median
          and scales by the IQR ``(x - median) / IQR``, making it
          resistant to outliers.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the specified columns scaled.
    """
    df = df.copy()
    scaler_map = {
        "standard": StandardScaler,   # z-score: (x - mean) / std
        "minmax": MinMaxScaler,       # rescale to [0, 1]: (x - min) / (max - min)
        "robust": RobustScaler,       # outlier-resistant: (x - median) / IQR
    }
    if method not in scaler_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(scaler_map)}")

    # Instantiate the chosen scaler, fit on the selected columns, and
    # replace those columns with the transformed values.
    # Validate that all selected columns are numeric before scaling
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Cannot scale non-numeric columns: {non_numeric}. Select only numeric columns."
        )

    scaler = scaler_map[method]()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# ---------------------------------------------------------------------------
#  6. Categorical Encoding
# ---------------------------------------------------------------------------


def encode_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "label",
) -> pd.DataFrame:
    """Encode categorical columns and return a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : list[str]
        Categorical column names to encode.
    method : str
        Encoding strategy:

        * ``'label'``  -- **LabelEncoder**: maps each unique category to
          a consecutive integer (0, 1, 2, ...).  Each column is encoded
          independently.
        * ``'onehot'`` -- **One-hot (pd.get_dummies)**: creates a new
          binary (0/1) column for *every* category in the original column.
          The original column is dropped.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the specified columns encoded.
    """
    df = df.copy()
    if method == "label":
        # LabelEncoder works on a single column at a time, so we loop.
        # Values are cast to str first to handle mixed types gracefully.
        le = LabelEncoder()
        for c in columns:
            if c in df.columns:
                df[c] = le.fit_transform(df[c].astype(str))
    elif method == "onehot":
        # get_dummies expands each categorical column into k binary columns
        # (one per unique value).  dtype=int ensures 0/1 integers rather
        # than True/False booleans.
        df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'label' or 'onehot'.")
    return df


# ---------------------------------------------------------------------------
#  7. Outlier Handling
# ---------------------------------------------------------------------------


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> dict:
    """Detect outliers via the IQR method and return diagnostic info.

    The **Interquartile Range (IQR)** method works as follows:

    1. Compute Q1 (25th percentile) and Q3 (75th percentile).
    2. IQR = Q3 - Q1  (the spread of the middle 50 % of data).
    3. Lower bound = Q1 - ``iqr_multiplier`` * IQR.
    4. Upper bound = Q3 + ``iqr_multiplier`` * IQR.
    5. Any value below the lower bound or above the upper bound is
       flagged as an outlier.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    column : str
        Numeric column to inspect.
    iqr_multiplier : float, default 1.5
        Multiplier applied to the IQR to determine the fence width.
        A value of 1.5 flags mild outliers; 3.0 flags extreme outliers.

    Returns
    -------
    dict
        Keys: ``q1``, ``q3``, ``iqr``, ``lower_bound``, ``upper_bound``,
        ``n_outliers``, ``outlier_mask`` (boolean Series).
    """
    # Step 1: Compute the first and third quartiles
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    # Step 2: IQR is the spread of the middle 50% of the distribution
    iqr = q3 - q1

    # Step 3-4: Fences define the acceptable range; anything outside is an outlier
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    # Step 5: Boolean mask -- True for rows that fall outside the fences
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
    """Handle outliers in a single numeric column.

    Uses the IQR method (see ``detect_outliers``) to identify outlier
    rows, then either removes them or caps (winsorizes) the values.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    column : str
        Numeric column to clean.
    action : str
        * ``'remove'`` -- drop rows whose value falls outside the IQR
          fences.
        * ``'cap'``    -- clip (winsorize) outlier values to the nearest
          fence boundary so no data rows are lost.
    iqr_multiplier : float, default 1.5
        Multiplier forwarded to ``detect_outliers``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with outliers handled.
    """
    df = df.copy()

    # Reuse detect_outliers to get the IQR-based fence boundaries
    info = detect_outliers(df, column, iqr_multiplier)
    lower, upper = info["lower_bound"], info["upper_bound"]

    if action == "remove":
        # Keep only rows within [lower, upper]
        df = df[(df[column] >= lower) & (df[column] <= upper)].reset_index(drop=True)
    elif action == "cap":
        # Winsorize: clip values so they sit at the fence boundaries
        df[column] = df[column].clip(lower, upper)
    else:
        raise ValueError(f"Unknown action: {action}. Choose 'remove' or 'cap'.")
    return df


# ---------------------------------------------------------------------------
#  8. Export
# ---------------------------------------------------------------------------


def export_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save the DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to export.
    filepath : str
        Destination path for the ``.csv`` file.
    **kwargs
        Additional keyword arguments forwarded to ``df.to_csv``.
    """
    df.to_csv(filepath, index=False, **kwargs)


# ---------------------------------------------------------------------------
#  9. Pipeline Runner (convenience helper)
# ---------------------------------------------------------------------------


def run_pipeline(
    df: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """Execute a sequence of cleaning steps declaratively.

    Each step is a dict whose ``"action"`` key selects a cleaning
    function, and whose remaining keys are forwarded as keyword
    arguments to that function.

    The pipeline processes steps **sequentially**: the output DataFrame
    of step *n* becomes the input of step *n + 1*.

    Parameters
    ----------
    df : pd.DataFrame
        The initial (raw) dataset.
    steps : list[dict]
        Ordered cleaning steps.  Example::

            [
                {"action": "handle_missing", "strategy": "mean"},
                {"action": "remove_duplicates"},
                {"action": "scale_columns",
                 "columns": ["col1"], "method": "standard"},
                {"action": "encode_columns",
                 "columns": ["species"], "method": "onehot"},
                {"action": "handle_outliers",
                 "column": "col1", "action_type": "cap"},
            ]

    Returns
    -------
    pd.DataFrame
        The fully cleaned dataset after all steps have been applied.
    """
    # Dispatch table: maps action names to callables.
    # Each lambda receives (DataFrame, step_dict) and returns a new DataFrame.
    # The "action" key is stripped from kwargs before forwarding so it does not
    # collide with actual function parameters.
    dispatch = {
        "handle_missing": lambda d, p: handle_missing(d, **{k: v for k, v in p.items() if k != "action"}),
        "remove_duplicates": lambda d, _: remove_duplicates(d),
        "scale_columns": lambda d, p: scale_columns(d, **{k: v for k, v in p.items() if k != "action"}),
        "encode_columns": lambda d, p: encode_columns(d, **{k: v for k, v in p.items() if k != "action"}),
        # handle_outliers needs special treatment because the step dict uses
        # "action_type" (to avoid clashing with the top-level "action" key)
        # while the function parameter is named "action".
        "handle_outliers": lambda d, p: handle_outliers(
            d,
            column=p["column"],
            action=p.get("action_type", "remove"),
            iqr_multiplier=p.get("iqr_multiplier", 1.5),
        ),
    }

    # Walk through each step in order, threading the DataFrame through
    for step in steps:
        action = step["action"]
        if action not in dispatch:
            raise ValueError(f"Unknown pipeline action: {action}")
        df = dispatch[action](df, step)
    return df


# ---------------------------------------------------------------------------
#  10. RDS Loading
# ---------------------------------------------------------------------------


def load_rds(filepath: str, **kwargs) -> pd.DataFrame:
    """Load an RDS file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .rds file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    import pyreadr
    result = pyreadr.read_r(filepath)
    # RDS files contain a single R object; extract the first (and usually only) one
    return list(result.values())[0]


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------

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
