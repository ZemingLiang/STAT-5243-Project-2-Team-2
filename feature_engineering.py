from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _require_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    bad = [col for col in columns if not _is_numeric(df[col])]
    if bad:
        raise ValueError(f"These columns must be numeric: {bad}")


def _require_categorical(df: pd.DataFrame, columns: list[str]) -> None:
    bad = [col for col in columns if not _is_categorical(df[col])]
    if bad:
        raise ValueError(f"These columns must be categorical-like: {bad}")


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Invalid column(s): {missing}")


def _apply_log_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1])
    series = pd.to_numeric(df[col1], errors="coerce")
    valid = series.dropna()
    if (valid < -1).any():
        raise ValueError(f"log1p requires all non-null values in '{col1}' to be >= -1.")

    out = df.copy()
    output_name = new_column or f"log_{col1}"
    out[output_name] = np.log1p(series)
    return out, {
        "feature_type": "log",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = log1p({col1})",
    }


def _apply_square_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1])
    out = df.copy()
    output_name = new_column or f"{col1}_squared"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") ** 2
    return out, {
        "feature_type": "square",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1}^2",
    }


def _apply_cube_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1])
    out = df.copy()
    output_name = new_column or f"{col1}_cubed"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") ** 3
    return out, {
        "feature_type": "cube",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1}^3",
    }


def _apply_interaction_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1, col2])
    out = df.copy()
    output_name = new_column or f"{col1}_x_{col2}"
    out[output_name] = pd.to_numeric(out[col1], errors="coerce") * pd.to_numeric(
        out[col2], errors="coerce"
    )
    return out, {
        "feature_type": "interaction",
        "input_columns": [col1, col2],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1} * {col2}",
    }


def _apply_ratio_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1, col2])
    out = df.copy()
    numerator = pd.to_numeric(out[col1], errors="coerce")
    denominator = pd.to_numeric(out[col2], errors="coerce")
    zero_mask = denominator == 0
    output_name = new_column or f"{col1}_div_{col2}"
    out[output_name] = numerator / denominator.mask(zero_mask, np.nan)
    return out, {
        "feature_type": "ratio",
        "input_columns": [col1, col2],
        "output_columns": [output_name],
        "formula": f"{output_name} = {col1} / {col2}",
        "n_zero_denominator": int(zero_mask.sum()),
        "note": "Zero denominators are converted to NaN.",
    }


def _apply_binning_feature(
    df: pd.DataFrame,
    col1: str,
    bins: int,
    *,
    new_column: str | None = None,
    labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1])
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    out = df.copy()
    series = pd.to_numeric(out[col1], errors="coerce")
    output_name = new_column or f"{col1}_binned"
    if labels:
        out[output_name] = pd.cut(series, bins=bins, include_lowest=True)
    else:
        out[output_name] = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    return out, {
        "feature_type": "binning",
        "input_columns": [col1],
        "output_columns": [output_name],
        "bins": int(bins),
        "labels": bool(labels),
        "formula": f"{output_name} = pd.cut({col1}, bins={bins})",
    }


def _apply_one_hot_encoding(
    df: pd.DataFrame,
    col1: str,
    *,
    prefix: str | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_categorical(df, [col1])
    out = df.copy()
    dummy_prefix = prefix or col1
    dummies = pd.get_dummies(
        out[col1],
        prefix=dummy_prefix,
        drop_first=drop_first,
        dummy_na=False,
    )
    out = pd.concat([out, dummies], axis=1)
    return out, {
        "feature_type": "one_hot",
        "input_columns": [col1],
        "output_columns": [str(c) for c in dummies.columns],
        "drop_first": bool(drop_first),
        "prefix": dummy_prefix,
        "formula": f"get_dummies({col1}, prefix='{dummy_prefix}', drop_first={drop_first})",
    }


def _apply_standardize_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1])
    series = pd.to_numeric(df[col1], errors="coerce")
    mean_val = series.mean()
    std_val = series.std()
    if pd.isna(std_val) or std_val == 0:
        raise ValueError(f"Column '{col1}' cannot be standardized because std is 0 or NaN.")

    out = df.copy()
    output_name = new_column or f"{col1}_zscore"
    out[output_name] = (series - mean_val) / std_val
    return out, {
        "feature_type": "standardize",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = ({col1} - mean({col1})) / std({col1})",
        "mean": float(mean_val),
        "std": float(std_val),
    }


def _apply_normalize_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_numeric(df, [col1])
    series = pd.to_numeric(df[col1], errors="coerce")
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        raise ValueError(
            f"Column '{col1}' cannot be normalized because min=max or values are invalid."
        )

    out = df.copy()
    output_name = new_column or f"{col1}_normalized"
    out[output_name] = (series - min_val) / (max_val - min_val)
    return out, {
        "feature_type": "normalize",
        "input_columns": [col1],
        "output_columns": [output_name],
        "formula": f"{output_name} = ({col1} - min({col1})) / (max({col1}) - min({col1}))",
        "min": float(min_val),
        "max": float(max_val),
    }


def _apply_fillna_feature(
    df: pd.DataFrame,
    col1: str,
    *,
    strategy: str = "mean",
    fill_value: Any | None = None,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' does not exist.")

    out = df.copy()
    series = out[col1]
    output_name = new_column or col1
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
    return out, {
        "feature_type": "fillna",
        "input_columns": [col1],
        "output_columns": [output_name],
        "strategy": strategy,
        "fill_value_used": value,
        "formula": f"{output_name} = fillna({col1}, strategy='{strategy}')",
    }


def _apply_dropna_feature(
    df: pd.DataFrame,
    col1: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    return out, {
        "feature_type": "dropna",
        "input_columns": [col1] if col1 else [],
        "output_columns": [],
        "scope": scope,
        "rows_removed": int(before_rows - len(out)),
        "formula": f"dropna(subset={scope})",
    }


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    method_norm = method.lower()
    cols_to_check = []
    if col1:
        cols_to_check.append(col1)
    if col2:
        cols_to_check.append(col2)
    _validate_columns(df, cols_to_check)

    if method_norm == "log":
        if not col1:
            raise ValueError("col1 is required for log.")
        return _apply_log_feature(df, col1, new_column=new_column)
    if method_norm == "square":
        if not col1:
            raise ValueError("col1 is required for square.")
        return _apply_square_feature(df, col1, new_column=new_column)
    if method_norm == "cube":
        if not col1:
            raise ValueError("col1 is required for cube.")
        return _apply_cube_feature(df, col1, new_column=new_column)
    if method_norm == "interaction":
        if not col1 or not col2:
            raise ValueError("col1 and col2 are required for interaction.")
        return _apply_interaction_feature(df, col1, col2, new_column=new_column)
    if method_norm == "ratio":
        if not col1 or not col2:
            raise ValueError("col1 and col2 are required for ratio.")
        return _apply_ratio_feature(df, col1, col2, new_column=new_column)
    if method_norm == "binning":
        if not col1:
            raise ValueError("col1 is required for binning.")
        return _apply_binning_feature(
            df,
            col1,
            bins=bins,
            new_column=new_column,
            labels=labels,
        )
    if method_norm in {"one_hot", "onehot", "one-hot"}:
        if not col1:
            raise ValueError("col1 is required for one_hot.")
        return _apply_one_hot_encoding(df, col1, prefix=prefix, drop_first=drop_first)
    if method_norm == "standardize":
        if not col1:
            raise ValueError("col1 is required for standardize.")
        return _apply_standardize_feature(df, col1, new_column=new_column)
    if method_norm == "normalize":
        if not col1:
            raise ValueError("col1 is required for normalize.")
        return _apply_normalize_feature(df, col1, new_column=new_column)
    if method_norm == "fillna":
        if not col1:
            raise ValueError("col1 is required for fillna.")
        return _apply_fillna_feature(
            df,
            col1,
            strategy=strategy,
            fill_value=fill_value,
            new_column=new_column,
        )
    if method_norm == "dropna":
        return _apply_dropna_feature(df, col1=col1)

    raise ValueError(
        "Invalid feature engineering method. Allowed: "
        "log, square, cube, interaction, ratio, binning, one_hot, "
        "standardize, normalize, fillna, dropna."
    )


def feature_engineering_capabilities() -> dict[str, Any]:
    return {
        "methods": [
            {
                "method": "log",
                "label": "Log Transform",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "square",
                "label": "Square Feature",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "cube",
                "label": "Cube Feature",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "interaction",
                "label": "Interaction (col1 x col2)",
                "requires": {"col1": True, "col2": True},
                "options": {"new_column": True},
            },
            {
                "method": "ratio",
                "label": "Ratio (col1 / col2)",
                "requires": {"col1": True, "col2": True},
                "options": {"new_column": True},
            },
            {
                "method": "binning",
                "label": "Binning",
                "requires": {"col1": True, "col2": False},
                "options": {"bins": True, "new_column": True, "labels": True},
            },
            {
                "method": "one_hot",
                "label": "One-Hot Encoding",
                "requires": {"col1": True, "col2": False},
                "options": {"prefix": True, "drop_first": True},
            },
            {
                "method": "standardize",
                "label": "Standardize (Z-score)",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
            },
            {
                "method": "normalize",
                "label": "Normalize (Min-Max)",
                "requires": {"col1": True, "col2": False},
                "options": {"new_column": True},
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
            },
            {
                "method": "dropna",
                "label": "Drop Missing Rows",
                "requires": {"col1": False, "col2": False},
                "options": {"col1_optional": True},
            },
        ]
    }
