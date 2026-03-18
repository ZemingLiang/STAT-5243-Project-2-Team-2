# dataset_store.py

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO

import pandas as pd


# =========================
# Configuration
# =========================

UPLOAD_DIR = Path("./uploaded_datasets")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".csv"}
DEFAULT_ENCODING = "utf-8"


# =========================
# Exceptions
# =========================

class DatasetStoreError(Exception):
    """Base exception for dataset store errors."""


class DatasetNotFoundError(DatasetStoreError):
    """Raised when a dataset_id does not exist."""


class DatasetValidationError(DatasetStoreError):
    """Raised when an uploaded file is invalid."""


# =========================
# Metadata model
# =========================

@dataclass
class DatasetMetadata:
    dataset_id: str
    filename: str
    stored_path: str
    file_type: str
    created_at: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================
# In-memory store
# =========================

_DATASET_CACHE: dict[str, pd.DataFrame] = {}
_DATASET_INDEX: dict[str, DatasetMetadata] = {}
_LOCK = RLock()


# =========================
# Public API
# =========================

def load_uploaded_csv(
    file_obj: BinaryIO,
    filename: str,
    *,
    save_raw_file: bool = True,
    cache_in_memory: bool = True,
    encoding: str = DEFAULT_ENCODING,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load an uploaded CSV into the dataset store.

    Steps:
    1. Validate filename / extension
    2. Assign dataset_id
    3. Optionally save the raw CSV to disk
    4. Read CSV into pandas DataFrame
    5. Store metadata
    6. Optionally cache DataFrame in memory

    Parameters
    ----------
    file_obj:
        File-like object from upload handling layer.
    filename:
        Original filename from the client.
    save_raw_file:
        Whether to persist the uploaded CSV to disk.
    cache_in_memory:
        Whether to keep the DataFrame in RAM for fast later access.
    encoding:
        Text encoding for CSV reading.
    read_csv_kwargs:
        Extra kwargs forwarded to pandas.read_csv().

    Returns
    -------
    dict
        JSON-friendly dataset summary for API responses.
    """
    read_csv_kwargs = read_csv_kwargs or {}
    _validate_csv_filename(filename)

    dataset_id = _generate_dataset_id()
    safe_name = _make_safe_filename(filename)
    stored_filename = f"{dataset_id}__{safe_name}"
    stored_path = UPLOAD_DIR / stored_filename

    raw_bytes = _read_all_bytes(file_obj)
    if len(raw_bytes) == 0:
        raise DatasetValidationError("Uploaded CSV file is empty.")

    if save_raw_file:
        _save_raw_file(raw_bytes, stored_path)

    df = _read_csv_from_bytes(raw_bytes, encoding=encoding, read_csv_kwargs=read_csv_kwargs)

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        filename=filename,
        stored_path=str(stored_path) if save_raw_file else "",
        file_type="csv",
        created_at=_utc_now_iso(),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns.tolist()],
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        size_bytes=len(raw_bytes),
    )

    with _LOCK:
        _DATASET_INDEX[dataset_id] = metadata
        if cache_in_memory:
            _DATASET_CACHE[dataset_id] = df

    return metadata.to_dict()


def get_dataset_by_id(dataset_id: str) -> pd.DataFrame:
    """
    Retrieve a dataset DataFrame by dataset_id.

    First checks the in-memory cache.
    If not cached, reloads from disk if the raw file was saved.
    """
    with _LOCK:
        if dataset_id in _DATASET_CACHE:
            return _DATASET_CACHE[dataset_id]

        metadata = _DATASET_INDEX.get(dataset_id)

    if metadata is None:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")

    if not metadata.stored_path:
        raise DatasetStoreError(
            f"Dataset '{dataset_id}' is not in memory and has no stored file to reload from."
        )

    path = Path(metadata.stored_path)
    if not path.exists():
        raise DatasetStoreError(
            f"Stored file for dataset '{dataset_id}' does not exist: {metadata.stored_path}"
        )

    df = pd.read_csv(path)

    with _LOCK:
        _DATASET_CACHE[dataset_id] = df

    return df


def get_dataset_metadata(dataset_id: str) -> dict[str, Any]:
    """Return dataset metadata as a JSON-friendly dict."""
    with _LOCK:
        metadata = _DATASET_INDEX.get(dataset_id)

    if metadata is None:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")

    return metadata.to_dict()


def list_datasets() -> list[dict[str, Any]]:
    """Return metadata for all currently known datasets."""
    with _LOCK:
        items = list(_DATASET_INDEX.values())

    return [item.to_dict() for item in items]


def delete_dataset(
    dataset_id: str,
    *,
    delete_raw_file: bool = True,
    clear_cache: bool = True,
) -> None:
    """
    Remove dataset metadata and optionally delete the raw file and cache entry.
    """
    with _LOCK:
        metadata = _DATASET_INDEX.pop(dataset_id, None)
        if clear_cache:
            _DATASET_CACHE.pop(dataset_id, None)

    if metadata is None:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")

    if delete_raw_file and metadata.stored_path:
        path = Path(metadata.stored_path)
        if path.exists():
            path.unlink()


def clear_dataset_cache(dataset_id: str | None = None) -> None:
    """
    Clear one cached DataFrame or the entire in-memory cache.

    This does not remove metadata or raw files.
    """
    with _LOCK:
        if dataset_id is None:
            _DATASET_CACHE.clear()
            return

        if dataset_id not in _DATASET_INDEX:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")

        _DATASET_CACHE.pop(dataset_id, None)


def is_dataset_cached(dataset_id: str) -> bool:
    """Return whether the dataset is currently in the RAM cache."""
    with _LOCK:
        if dataset_id not in _DATASET_INDEX:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")
        return dataset_id in _DATASET_CACHE


def get_dataset_summary(dataset_id: str) -> dict[str, Any]:
    """
    Convenience summary combining metadata with current cache state.
    """
    metadata = get_dataset_metadata(dataset_id)
    metadata["is_cached"] = is_dataset_cached(dataset_id)
    return metadata


# =========================
# Private helpers
# =========================

def _generate_dataset_id() -> str:
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_csv_filename(filename: str) -> None:
    if not filename or not filename.strip():
        raise DatasetValidationError("Filename is missing.")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise DatasetValidationError(
            f"Unsupported file type '{suffix}'. Allowed types: {sorted(ALLOWED_SUFFIXES)}"
        )


def _make_safe_filename(filename: str) -> str:
    """
    Very simple filename sanitization.
    Keeps only basename and replaces spaces.
    """
    base = Path(filename).name.strip().replace(" ", "_")
    return base or "uploaded.csv"


def _read_all_bytes(file_obj: BinaryIO) -> bytes:
    """
    Read all bytes from a file-like object safely.
    Resets to beginning first if possible.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    raw = file_obj.read()

    if isinstance(raw, str):
        raw = raw.encode(DEFAULT_ENCODING)

    if not isinstance(raw, (bytes, bytearray)):
        raise DatasetValidationError("Uploaded file object did not produce bytes.")

    return bytes(raw)


def _save_raw_file(raw_bytes: bytes, stored_path: Path) -> None:
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(raw_bytes)


def _read_csv_from_bytes(
    raw_bytes: bytes,
    *,
    encoding: str,
    read_csv_kwargs: dict[str, Any],
) -> pd.DataFrame:
    try:
        text_buffer = io.BytesIO(raw_bytes)
        df = pd.read_csv(text_buffer, encoding=encoding, **read_csv_kwargs)
    except Exception as exc:
        raise DatasetValidationError(f"Failed to parse CSV: {exc}") from exc

    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("Parsed object is not a pandas DataFrame.")

    return df