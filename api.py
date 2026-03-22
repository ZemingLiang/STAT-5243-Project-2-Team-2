# api.py
"""
Service / API layer.

This module sits between the HTTP route handlers (or any caller) and the
pure logic modules.  It:

- Has no knowledge of HTTP frameworks.
- Composes ``EDA`` (pure analytics) and ``dataset_store`` (persistence).
- Always returns a JSON-friendly ``dict`` — never raises.

Separation of concerns
-----------------------
- ``EDA.apply_filter``        — pure filtering, raises ValueError on bad input.
- ``dataset_store.*``         — load / persist / retrieve DataFrames by id.
- ``api.filter_and_save_dataset`` — orchestration only; error → JSON error.
"""

from __future__ import annotations

from typing import Any

import EDA
import dataset_store


def filter_and_save_dataset(
    dataset_id: str,
    filter_expr: str,
) -> dict[str, Any]:
    """
    Apply a filter to an existing dataset and save the result as a new one.

    Orchestration steps:
    1. Retrieve the source DataFrame from the store.
    2. Apply *filter_expr* via ``EDA.apply_filter`` (raises on bad input).
    3. Register the filtered DataFrame via ``dataset_store.register_dataframe``.
    4. Return a JSON-friendly summary.

    Parameters
    ----------
    dataset_id:
        ID of the source dataset already loaded into the store.
    filter_expr:
        pandas ``DataFrame.query()``-compatible expression string, e.g.
        ``"age > 30"`` or ``"gender == 'Female' and stress_level < 5"``.

    Returns
    -------
    dict
        On success::

            {
                "status": "success",
                "data": {
                    "source_dataset_id": "...",
                    "new_dataset_id": "...",
                    "n_rows_before": 15000,
                    "n_rows_after": 8143
                }
            }

        On error::

            { "status": "error", "message": "..." }
    """
    # ── 1. Retrieve source DataFrame ─────────────────────────────────────────
    try:
        df = dataset_store.get_dataset_by_id(dataset_id)
    except dataset_store.DatasetNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    except dataset_store.DatasetStoreError as exc:
        return {"status": "error", "message": str(exc)}

    n_rows_before = int(len(df))

    # ── 2. Apply filter (pure; raises ValueError on bad input) ───────────────
    try:
        filtered_df = EDA.apply_filter(df, filter_expr)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    # ── 3. Register filtered DataFrame ───────────────────────────────────────
    try:
        meta = dataset_store.register_dataframe(
            filtered_df,
            parent_dataset_id=dataset_id,
            kind="filtered",
            transform={"filter_expr": filter_expr},
        )
    except dataset_store.DatasetValidationError as exc:
        return {"status": "error", "message": str(exc)}
    except dataset_store.DatasetStoreError as exc:
        return {"status": "error", "message": str(exc)}

    # ── 4. Return summary ─────────────────────────────────────────────────────
    return {
        "status": "success",
        "data": {
            "source_dataset_id": dataset_id,
            "new_dataset_id": meta["dataset_id"],
            "n_rows_before": n_rows_before,
            "n_rows_after": int(len(filtered_df)),
        },
    }
