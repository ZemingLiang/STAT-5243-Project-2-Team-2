from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dataset_store import (
    DatasetNotFoundError,
    DatasetStoreError,
    create_derived_dataset,
    get_dataset_by_id,
    get_dataset_metadata,
    update_dataset_by_id,
)


# Frontend integration note
"""
Frontend integration note
-------------------------
This module follows the same backend response style as EDA.py.

Typical request flow:
1. Frontend already has a dataset_id from the upload / preprocessing stage.
2. Frontend sends:
       dataset_id + feature method + method parameters + save_mode
3. Backend:
       - loads DataFrame from dataset_store
       - applies feature engineering
       - either:
           a) preview only
           b) overwrite current dataset_id
           c) create a derived dataset_id
4. Backend returns JSON-friendly response.

Supported save modes
--------------------
- "preview_only":
    Do not modify dataset_store. Only return transformed preview + metadata.
- "overwrite":
    Overwrite the existing dataset_id in dataset_store.
- "derived":
    Create a new derived dataset_id in dataset_store.

Standard response envelope
--------------------------
Success:
{
  "status": "success",
  "data": { ... },
  "message": "..."
}

Warning:
{
  "status": "warning",
  "message": "...",
  "data": { ... }
}

Error:
{
  "status": "error",
  "message": "...",
  "details": { ... }
}
"""


# Generic helpers
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
    """Return True if a Series should be treated as categorical."""
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _require_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any listed columns are not numeric."""
    bad = [col for col in columns if not _is_numeric(df[col])]
    if bad:
        raise ValueError(f"These columns must be numeric: {bad}")


def _require_categorical(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any listed columns are not categorical-like."""
    bad = [col for col in columns if not _is_categorical(df[col])]
    if bad:
        raise ValueError(f"These columns must be categorical-like: {bad}")


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


def _preview_df(df: pd.DataFrame, n: int = 20) -> dict[str, Any]:
    """Return a small JSON-friendly preview of a DataFrame."""
    out = df.head(n)
    return {
        "n_requested": int(n),
        "n_returned": int(len(out)),
        "columns": [str(c) for c in out.columns],
        "rows": _records_from_df(out),
    }


def _dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return compact summary metadata about a DataFrame."""
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": [str(c) for c in df.select_dtypes(include="number").columns.tolist()],
        "categorical_columns": [str(c) for c in df.select_dtypes(exclude="number").columns.tolist()],
    }


def _new_columns(before_df: pd.DataFrame, after_df: pd.DataFrame) -> list[str]:
    """Return columns newly added by a transformation."""
    before_cols = set(before_df.columns)
    return [str(c) for c in after_df.columns if c not in before_cols]


# Core dataframe-level feature transforms
def _apply_log_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create a log-transformed feature using log1p(x).

    Requirement:
    - column must be numeric
    - all non-null values must be >= -1
    """
    _require_numeric(df, [col1])

    series = pd.to_numeric(df[col1], errors="coerce")
    valid = series.dropna()
    if (valid < -1).any():
        raise ValueError(
            f"log1p requires all non-null values in '{col1}' to be >= -1."
        )

    out = df.copy()
    output_name = new_column if new_column else f"log_{col1}"
    out[output_name] = np.log1p(series)

    meta = {
        "feature_type": "log",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = log1p({col1})",
    }
    return out, meta


def _apply_square_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a squared feature."""
    _require_numeric(df, [col1])

    out = df.copy()
    output_name = new_column if new_column else f"{col1}_squared"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") ** 2

    meta = {
        "feature_type": "square",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1}^2",
    }
    return out, meta


def _apply_cube_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a cubed feature."""
    _require_numeric(df, [col1])

    out = df.copy()
    output_name = new_column if new_column else f"{col1}_cubed"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") ** 3

    meta = {
        "feature_type": "cube",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1}^3",
    }
    return out, meta


def _apply_interaction_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create an interaction feature col1 * col2."""
    _require_numeric(df, [col1, col2])

    out = df.copy()
    output_name = new_column if new_column else f"{col1}_x_{col2}"
    out[output_name] = (
        pd.to_numeric(out[col1], errors="coerce")
        * pd.to_numeric(out[col2], errors="coerce")
    )

    meta = {
        "feature_type": "interaction",
        "input_columns": [col1, col2],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1} * {col2}",
    }
    return out, meta


def _apply_ratio_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create a ratio feature col1 / col2.

    Zero denominators are converted to NaN instead of raising an exception.
    """
    _require_numeric(df, [col1, col2])

    out = df.copy()
    numerator = pd.to_numeric(out[col1], errors="coerce")
    denominator = pd.to_numeric(out[col2], errors="coerce")

    output_name = new_column if new_column else f"{col1}_div_{col2}"
    zero_mask = denominator == 0
    out[output_name] = numerator / denominator.mask(zero_mask, np.nan)

    meta = {
        "feature_type": "ratio",
        "input_columns": [col1, col2],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1} / {col2}",
        "n_zero_denominator": int(zero_mask.sum()),
        "note": "Zero denominators are converted to NaN.",
    }
    return out, meta


def _apply_binning_feature(
    df: pd.DataFrame,
    col1: str,
    bins: int,
    *,
    new_column: str | None = None,
    labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create a binned version of a numeric column.

    If labels=False, output is integer-coded bins.
    If labels=True, output is interval labels (categorical).
    """
    _require_numeric(df, [col1])

    if bins < 2:
        raise ValueError("bins must be at least 2.")

    out = df.copy()
    series = pd.to_numeric(out[col1], errors="coerce")
    output_name = new_column if new_column else f"{col1}_binned"

    if labels:
        out[output_name] = pd.cut(series, bins=bins, include_lowest=True)
    else:
        out[output_name] = pd.cut(
            series,
            bins=bins,
            labels=False,
            include_lowest=True,
        )

    meta = {
        "feature_type": "binning",
        "input_columns": [col1],
        "output_columns": [output_name],
        "bins": int(bins),
        "labels": bool(labels),
        "formula": f"{output_name} = pd.cut({col1}, bins={bins})",
    }
    return out, meta


def _apply_one_hot_encoding(
    df: pd.DataFrame,
    col1: str,
    *,
    prefix: str | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One-hot encode a categorical column and append the dummy columns."""
    _require_categorical(df, [col1])

    out = df.copy()
    dummy_prefix = prefix if prefix else col1

    dummies = pd.get_dummies(
        out[col1],
        prefix=dummy_prefix,
        drop_first=drop_first,
        dummy_na=False,
    )

    out = pd.concat([out, dummies], axis=1)

    meta = {
        "feature_type": "one_hot",
        "input_columns": [col1],
        "output_columns": [str(c) for c in dummies.columns],
        "drop_first": bool(drop_first),
        "prefix": dummy_prefix,
        "formula": f"get_dummies({col1}, prefix='{dummy_prefix}', drop_first={drop_first})",
    }
    return out, meta


def _apply_standardize_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Standardize a numeric column:
        z = (x - mean) / std
    """
    _require_numeric(df, [col1])

    series = pd.to_numeric(df[col1], errors="coerce")
    mean_val = series.mean()
    std_val = series.std()

    if pd.isna(std_val) or std_val == 0:
        raise ValueError(f"Column '{col1}' cannot be standardized because std is 0 or NaN.")

    out = df.copy()
    output_name = new_column if new_column else f"{col1}_zscore"
    out[output_name] = (series - mean_val) / std_val

    meta = {
        "feature_type": "standardize",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = ({col1} - mean({col1})) / std({col1})",
        "mean": float(mean_val),
        "std": float(std_val),
    }
    return out, meta


def _apply_normalize_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Min-max normalize a numeric column:
        (x - min) / (max - min)
    """
    _require_numeric(df, [col1])

    series = pd.to_numeric(df[col1], errors="coerce")
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        raise ValueError(f"Column '{col1}' cannot be normalized because min=max or values are invalid.")

    out = df.copy()
    output_name = new_column if new_column else f"{col1}_normalized"
    out[output_name] = (series - min_val) / (max_val - min_val)

    meta = {
        "feature_type": "normalize",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = ({col1} - min({col1})) / (max({col1}) - min({col1}))",
        "min": float(min_val),
        "max": float(max_val),
    }
    return out, meta


def _apply_fillna_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    strategy: str = "mean",   # mean | median | mode | constant
    fill_value: Any | None = None,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fill missing values in a column.
    - mean / median for numeric columns
    - mode for any column
    - constant for any column
    """
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' does not exist.")

    out = df.copy()
    series = out[col1]
    output_name = new_column if new_column else col1

    strategy = strategy.lower()

    if strategy == "mean":
        _require_numeric(out, [col1])
        value = pd.to_numeric(series, errors="coerce").mean()

    elif strategy == "median":
        _require_numeric(out, [col1])
        value = pd.to_numeric(series, errors="coerce").median()

    elif strategy == "mode":
        mode_vals = series.mode(dropna=True)
        if len(mode_vals) == 0:
            raise ValueError(f"Column '{col1}' has no mode available for filling.")
        value = mode_vals.iloc[0]

    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value is required when strategy='constant'.")
        value = fill_value

    else:
        raise ValueError("strategy must be one of: mean, median, mode, constant.")

    if output_name == col1:
        out[col1] = series.fillna(value)
    else:
        out[output_name] = series.fillna(value)

    meta = {
        "feature_type": "fillna",
        "input_columns": [col1],
        "output_columns": [output_name],
        "strategy": strategy,
        "fill_value_used": _json_ready(value),
        "formula": f"{output_name} = fillna({col1}, strategy='{strategy}')",
    }
    return out, meta


def _apply_dropna_feature(
    df: pd.DataFrame,
    col1: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Drop rows with missing values.
    If col1 is provided, only consider missingness in that column.
    """
    out = df.copy()
    before_rows = len(out)

    if col1:
        if col1 not in out.columns:
            raise ValueError(f"Column '{col1}' does not exist.")
        out = out.dropna(subset=[col1])
        scope = [col1]
    else:
        out = out.dropna()
        scope = "all_columns"

    after_rows = len(out)

    meta = {
        "feature_type": "dropna",
        "input_columns": [col1] if col1 else [],
        "output_columns": [],
        "scope": scope,
        "rows_removed": int(before_rows - after_rows),
        "formula": f"dropna(subset={scope})",
    }
    return out, meta


# Dataframe-level dispatcher
def apply_feature_engineering_to_df(
    df: pd.DataFrame,
    method: str,
    col1: str | None = None,
    *,
    col2: str | None = None,
    bins: int = 4,
    new_column: str | None = None,
    labels: bool = False,
    prefix: str | None = None,
    drop_first: bool = False,
    strategy: str = "mean",
    fill_value: Any | None = None,
) -> dict[str, Any]:
    """
    Apply feature engineering directly to a pandas DataFrame.

    This is useful for local testing and for internal reuse by the dataset_id-level API.
    """
    if not isinstance(df, pd.DataFrame):
        return _error("Input must be a pandas DataFrame.")

    method = method.lower()

    cols_to_check = []
    if col1 and method not in {"dropna"}:
        cols_to_check.append(col1)
    elif col1 and method == "dropna":
        cols_to_check.append(col1)

    if col2:
        cols_to_check.append(col2)

    err = _validate_columns(df, cols_to_check)
    if err:
        return _error(err)

    try:
        if method == "log":
            if not col1:
                return _error("col1 is required for log.")
            out, meta = _apply_log_feature(df, col1, new_column=new_column)

        elif method == "square":
            if not col1:
                return _error("col1 is required for square.")
            out, meta = _apply_square_feature(df, col1, new_column=new_column)

        elif method == "cube":
            if not col1:
                return _error("col1 is required for cube.")
            out, meta = _apply_cube_feature(df, col1, new_column=new_column)

        elif method == "interaction":
            if not col1 or not col2:
                return _error("col1 and col2 are required for interaction.")
            out, meta = _apply_interaction_feature(
                df, col1, col2, new_column=new_column
            )

        elif method == "ratio":
            if not col1 or not col2:
                return _error("col1 and col2 are required for ratio.")
            out, meta = _apply_ratio_feature(
                df, col1, col2, new_column=new_column
            )

        elif method == "binning":
            if not col1:
                return _error("col1 is required for binning.")
            out, meta = _apply_binning_feature(
                df,
                col1,
                bins=bins,
                new_column=new_column,
                labels=labels,
            )

        elif method in {"one_hot", "onehot", "one-hot"}:
            if not col1:
                return _error("col1 is required for one_hot.")
            out, meta = _apply_one_hot_encoding(
                df,
                col1,
                prefix=prefix,
                drop_first=drop_first,
            )

        elif method == "standardize":
            if not col1:
                return _error("col1 is required for standardize.")
            out, meta = _apply_standardize_feature(df, col1, new_column=new_column)

        elif method == "normalize":
            if not col1:
                return _error("col1 is required for normalize.")
            out, meta = _apply_normalize_feature(df, col1, new_column=new_column)

        elif method == "fillna":
            if not col1:
                return _error("col1 is required for fillna.")
            out, meta = _apply_fillna_feature(
                df,
                col1,
                strategy=strategy,
                fill_value=fill_value,
                new_column=new_column,
            )

        elif method == "dropna":
            out, meta = _apply_dropna_feature(df, col1=col1)

        else:
            return _error(
                "Invalid feature engineering method.",
                details={
                    "allowed": [
                        "log",
                        "square",
                        "cube",
                        "interaction",
                        "ratio",
                        "binning",
                        "one_hot",
                        "standardize",
                        "normalize",
                        "fillna",
                        "dropna",
                    ]
                },
            )

        payload = {
            "feature_meta": meta,
            "new_columns": _new_columns(df, out),
            "dataset_summary": _dataset_summary(out),
            "preview": _preview_df(out, n=20),
        }
        return _success(
            _json_ready(payload),
            message="Feature engineering applied successfully.",
        )

    except Exception as exc:
        return _error(
            "Feature engineering failed.",
            details={"exception": str(exc)},
        )


# Dataset-id-level API-friendly entry point
def apply_feature_engineering(
    dataset_id: str,
    method: str,
    col1: str | None = None,
    *,
    col2: str | None = None,
    bins: int = 4,
    new_column: str | None = None,
    labels: bool = False,
    prefix: str | None = None,
    drop_first: bool = False,
    strategy: str = "mean",
    fill_value: Any | None = None,
    save_mode: str = "overwrite",   # overwrite | derived | preview_only
    derived_suffix: str | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """
    Apply feature engineering to a stored dataset.

    Parameters
    ----------
    dataset_id:
        Existing dataset id in dataset_store.
    method:
        Feature method name.
    col1, col2:
        Input columns.
    bins:
        For binning only.
    new_column:
        Optional custom output column name for single-output transforms.
    labels:
        For binning only. If True, return interval labels instead of integer codes.
    prefix:
        For one-hot encoding only.
    drop_first:
        For one-hot encoding only.
    strategy:
        For fillna only. One of: mean, median, mode, constant.
    fill_value:
        For fillna only when strategy='constant'.
    save_mode:
        - "overwrite": overwrite current dataset_id
        - "derived": create a new derived dataset_id
        - "preview_only": do not save; only return preview
    derived_suffix:
        Optional custom filename suffix when save_mode="derived".
    preview_rows:
        Number of preview rows to return.

    Returns
    -------
    dict
        JSON-friendly response matching the EDA.py style.
    """
    try:
        df = get_dataset_by_id(dataset_id)
        source_meta = get_dataset_metadata(dataset_id)
    except DatasetNotFoundError as exc:
        return _error(
            "Dataset not found.",
            details={"dataset_id": dataset_id, "exception": str(exc)},
        )
    except DatasetStoreError as exc:
        return _error(
            "Failed to load dataset.",
            details={"dataset_id": dataset_id, "exception": str(exc)},
        )

    method_norm = method.lower()

    cols_to_check = []
    if col1 and method_norm not in {"dropna"}:
        cols_to_check.append(col1)
    elif col1 and method_norm == "dropna":
        cols_to_check.append(col1)

    if col2:
        cols_to_check.append(col2)

    err = _validate_columns(df, cols_to_check)
    if err:
        return _error(err)

    try:
        if method_norm == "log":
            if not col1:
                return _error("col1 is required for log.")
            df_new, meta = _apply_log_feature(df, col1, new_column=new_column)

        elif method_norm == "square":
            if not col1:
                return _error("col1 is required for square.")
            df_new, meta = _apply_square_feature(df, col1, new_column=new_column)

        elif method_norm == "cube":
            if not col1:
                return _error("col1 is required for cube.")
            df_new, meta = _apply_cube_feature(df, col1, new_column=new_column)

        elif method_norm == "interaction":
            if not col1 or not col2:
                return _error("col1 and col2 are required for interaction.")
            df_new, meta = _apply_interaction_feature(
                df, col1, col2, new_column=new_column
            )

        elif method_norm == "ratio":
            if not col1 or not col2:
                return _error("col1 and col2 are required for ratio.")
            df_new, meta = _apply_ratio_feature(
                df, col1, col2, new_column=new_column
            )

        elif method_norm == "binning":
            if not col1:
                return _error("col1 is required for binning.")
            df_new, meta = _apply_binning_feature(
                df,
                col1,
                bins=bins,
                new_column=new_column,
                labels=labels,
            )

        elif method_norm in {"one_hot", "onehot", "one-hot"}:
            if not col1:
                return _error("col1 is required for one_hot.")
            df_new, meta = _apply_one_hot_encoding(
                df,
                col1,
                prefix=prefix,
                drop_first=drop_first,
            )

        elif method_norm == "standardize":
            if not col1:
                return _error("col1 is required for standardize.")
            df_new, meta = _apply_standardize_feature(df, col1, new_column=new_column)

        elif method_norm == "normalize":
            if not col1:
                return _error("col1 is required for normalize.")
            df_new, meta = _apply_normalize_feature(df, col1, new_column=new_column)

        elif method_norm == "fillna":
            if not col1:
                return _error("col1 is required for fillna.")
            df_new, meta = _apply_fillna_feature(
                df,
                col1,
                strategy=strategy,
                fill_value=fill_value,
                new_column=new_column,
            )

        elif method_norm == "dropna":
            df_new, meta = _apply_dropna_feature(df, col1=col1)

        else:
            return _error(
                "Invalid feature engineering method.",
                details={
                    "allowed": [
                        "log",
                        "square",
                        "cube",
                        "interaction",
                        "ratio",
                        "binning",
                        "one_hot",
                        "standardize",
                        "normalize",
                        "fillna",
                        "dropna",
                    ]
                },
            )

    except Exception as exc:
        return _error(
            "Feature engineering failed.",
            details={"exception": str(exc)},
        )

    new_cols = _new_columns(df, df_new)
    preview = _preview_df(df_new, n=preview_rows)
    summary = _dataset_summary(df_new)

    save_mode = save_mode.lower()

    if save_mode == "preview_only":
        return _success(
            {
                "save_mode": "preview_only",
                "source_dataset_id": dataset_id,
                "source_dataset_filename": source_meta["filename"],
                "feature_meta": meta,
                "new_columns": new_cols,
                "dataset_summary": summary,
                "preview": preview,
            },
            message="Preview generated successfully. Dataset store was not modified.",
        )

    if save_mode == "overwrite":
        try:
            updated_meta = update_dataset_by_id(dataset_id, df_new)
        except Exception as exc:
            return _error(
                "Failed to overwrite dataset.",
                details={"dataset_id": dataset_id, "exception": str(exc)},
            )

        return _success(
            {
                "save_mode": "overwrite",
                "dataset_id": dataset_id,
                "feature_meta": meta,
                "new_columns": new_cols,
                "dataset_metadata": updated_meta,
                "dataset_summary": summary,
                "preview": preview,
            },
            message="Feature engineering applied and original dataset overwritten successfully.",
        )

    if save_mode == "derived":
        suffix = derived_suffix if derived_suffix else f"feature_{method_norm}"
        try:
            derived_meta = create_derived_dataset(
                df=df_new,
                source_dataset_id=dataset_id,
                filename_suffix=suffix,
            )
        except Exception as exc:
            return _error(
                "Failed to create derived dataset.",
                details={
                    "source_dataset_id": dataset_id,
                    "exception": str(exc),
                },
            )

        return _success(
            {
                "save_mode": "derived",
                "source_dataset_id": dataset_id,
                "derived_dataset_id": derived_meta["dataset_id"],
                "feature_meta": meta,
                "new_columns": new_cols,
                "dataset_metadata": derived_meta,
                "dataset_summary": summary,
                "preview": preview,
            },
            message="Feature engineering applied and derived dataset created successfully.",
        )

    return _error(
        "Invalid save_mode.",
        details={"allowed": ["overwrite", "derived", "preview_only"]},
    )


# convenience function for frontend dropdowns / UI configuration
def feature_engineering_capabilities() -> dict[str, Any]:
    """
    Return a JSON-friendly description of available feature-engineering methods.

    Useful for frontend dropdown population or user guide rendering.
    """
    payload = {
        "methods": [
            {
                "method": "log",
                "label": "Log Transform",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "square",
                "label": "Square Feature",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "cube",
                "label": "Cube Feature",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "interaction",
                "label": "Interaction (col1 × col2)",
                "requires": {"col1": True, "col2": True},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "ratio",
                "label": "Ratio (col1 ÷ col2)",
                "requires": {"col1": True, "col2": True},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "binning",
                "label": "Binning",
                "requires": {"col1": True, "col2": False},
                "options": {"bins": True, "new_column": True, "labels": True},
                "output_type": "categorical_or_integer",
            },
            {
                "method": "one_hot",
                "label": "One-Hot Encoding",
                "requires": {"col1": True, "col2": False},
                "options": {"prefix": True, "drop_first": True},
                "output_type": "multiple_numeric_columns",
            },
            {
                "method": "standardize",
                "label": "Standardize (Z-score)",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "normalize",
                "label": "Normalize (Min-Max)",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
                "output_type": "numeric",
            },
            {
                "method": "fillna",
                "label": "Fill Missing Values",
                "requires": {"col1": True, "col2": False},
                "options": {
                    "strategy": ["mean", "median", "mode", "constant"],
                    "fill_value": True,
                    "new_column": True,
                },
                "output_type": "same_as_input",
            },
            {
                "method": "dropna",
                "label": "Drop Missing Rows",
                "requires": {"col1": False, "col2": False},
                "options": {"col1_optional": True},
                "output_type": "row_filtered_dataframe",
            },
        ],
        "save_modes": ["preview_only", "overwrite", "derived"],
    }
    return _success(payload)