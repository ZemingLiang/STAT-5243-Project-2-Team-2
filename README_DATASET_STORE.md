# Dataset Store — Reference

`dataset_store.py` manages the lifecycle of datasets: receiving an uploaded file (or a programmatically-derived DataFrame), persisting it to disk, caching in RAM, and making it retrievable by a stable `dataset_id`.

EDA functions in `EDA.py` receive a plain `pd.DataFrame` and have no knowledge of this layer. The API layer translates a `dataset_id` from an HTTP request into a DataFrame via `get_dataset_by_id(dataset_id)`, and persists derived DataFrames via `register_dataframe(...)`.

---

## Storage model

```
User uploads CSV
    ↓
load_uploaded_csv(file_obj, filename)
    ↓  validates, assigns dataset_id (UUID4)
    ↓  optionally writes raw bytes to ./uploaded_datasets/<id>__<filename>
    ↓  parses into pd.DataFrame
    ↓  stores DatasetMetadata in _DATASET_INDEX  (kind="uploaded")
    ↓  optionally caches DataFrame in _DATASET_CACHE
    → returns metadata dict (incl. dataset_id)

API layer creates a derived dataset (e.g. filtered view)
    ↓
register_dataframe(filtered_df, parent_dataset_id=..., kind="filtered", transform={...})
    ↓  assigns new dataset_id (UUID4)
    ↓  writes CSV to ./uploaded_datasets/<new_id>__derived.csv
    ↓  stores DatasetMetadata in _DATASET_INDEX  (kind="filtered", parent_dataset_id=...)
    ↓  caches DataFrame in _DATASET_CACHE
    → returns metadata dict (incl. new dataset_id + provenance)

Later requests: get_dataset_by_id(dataset_id)
    ↓  checks _DATASET_CACHE (fast path)
    ↓  if miss: reloads from disk and re-caches
    → returns pd.DataFrame
```

Both the in-memory index and cache are plain Python dicts protected by a `threading.RLock`. The store is **in-process and non-persistent** — restarting the server clears all state.

---

## Public API

### `load_uploaded_csv(file_obj, filename, *, save_raw_file=True, cache_in_memory=True, encoding="utf-8", read_csv_kwargs=None) → dict`

Load and register an uploaded CSV.

| Parameter | Default | Description |
|---|---|---|
| `file_obj` | — | Binary file-like object from the HTTP upload handler |
| `filename` | — | Original filename from the client (used for validation and display) |
| `save_raw_file` | `True` | Write raw bytes to `./uploaded_datasets/` |
| `cache_in_memory` | `True` | Keep the parsed DataFrame in RAM |
| `encoding` | `"utf-8"` | Text encoding passed to `pandas.read_csv` |
| `read_csv_kwargs` | `None` | Extra kwargs forwarded to `pandas.read_csv` |

Returns a `DatasetMetadata` dict. Raises `DatasetValidationError` for empty files, unsupported extensions (only `.csv` is allowed), or CSV parse failures.

---

### `get_dataset_by_id(dataset_id) → pd.DataFrame`

Retrieve the DataFrame for a known dataset.

- Checks the in-memory cache first (O(1)).
- On a cache miss, reloads from the saved file path and re-caches.
- Raises `DatasetNotFoundError` if the ID is unknown.
- Raises `DatasetStoreError` if not cached and no saved file exists.

---

### `register_dataframe(df, parent_dataset_id=None, kind="derived", transform=None) → dict`

Register an in-memory DataFrame as a new dataset entry — the complement to `load_uploaded_csv` for programmatically-created DataFrames.

| Parameter | Default | Description |
|---|---|---|
| `df` | — | DataFrame to register. Must not be empty. |
| `parent_dataset_id` | `None` | `dataset_id` of the source this was derived from |
| `kind` | `"derived"` | Label for the derivation type, e.g. `"filtered"`, `"joined"` |
| `transform` | `None` | JSON-serialisable dict describing the transformation, e.g. `{"filter_expr": "age > 30"}` |

Assigns a new `dataset_id`, writes the DataFrame to disk as a CSV, stores provenance metadata, caches in RAM, and returns the metadata dict. Raises `DatasetValidationError` if `df` is not a DataFrame or is empty.

Provenance fields in returned metadata: `parent_dataset_id`, `kind`, `transform`.

---

### `get_dataset_metadata(dataset_id) → dict`

Return the stored `DatasetMetadata` as a JSON-friendly dict.

Fields: `dataset_id`, `filename`, `stored_path`, `file_type`, `created_at` (ISO 8601 UTC), `n_rows`, `n_cols`, `columns`, `dtypes`, `size_bytes`, `parent_dataset_id`, `kind`, `transform`.

---

### `list_datasets() → list[dict]`

Return metadata for all currently registered datasets.

---

### `delete_dataset(dataset_id, *, delete_raw_file=True, clear_cache=True) → None`

Remove a dataset from the index and optionally delete the raw file and cache entry. Raises `DatasetNotFoundError` if the ID is unknown.

---

### `clear_dataset_cache(dataset_id=None) → None`

Evict one or all DataFrames from the RAM cache without removing metadata or raw files. Pass `dataset_id=None` to clear everything.

---

### `is_dataset_cached(dataset_id) → bool`

Return whether a DataFrame is currently in the RAM cache.

---

### `get_dataset_summary(dataset_id) → dict`

Convenience wrapper: returns `get_dataset_metadata(...)` plus an `"is_cached"` boolean field.

---

## Exceptions

| Exception | When raised |
|---|---|
| `DatasetStoreError` | Base class for all store errors |
| `DatasetNotFoundError` | `dataset_id` not in index |
| `DatasetValidationError` | Empty file, unsupported extension, or CSV parse failure |

---

## Configuration constants

| Constant | Default | Meaning |
|---|---|---|
| `UPLOAD_DIR` | `./uploaded_datasets` | Directory for saved raw CSV files |
| `ALLOWED_SUFFIXES` | `{".csv"}` | Accepted file extensions |
| `DEFAULT_ENCODING` | `"utf-8"` | Default text encoding for CSV reading |

---

## Typical API-layer usage

```python
# Upload endpoint
from dataset_store import load_uploaded_csv, DatasetValidationError

def upload(file_bytes: bytes, filename: str) -> dict:
    from io import BytesIO
    try:
        meta = load_uploaded_csv(BytesIO(file_bytes), filename)
    except DatasetValidationError as exc:
        return {"status": "error", "message": str(exc)}
    return {"status": "success", "dataset_id": meta["dataset_id"]}


# EDA endpoint
from dataset_store import get_dataset_by_id, DatasetNotFoundError
import EDA

def get_head(dataset_id: str, n: int = 5) -> dict:
    try:
        df = get_dataset_by_id(dataset_id)
    except DatasetNotFoundError:
        return {"status": "error", "message": f"Dataset '{dataset_id}' not found."}
    return EDA.show_head(df, n=n)


# Filter-and-save endpoint — use api.py, which orchestrates EDA + dataset_store
import api

def filter_dataset(dataset_id: str, filter_expr: str) -> dict:
    # Returns {"status": "success", "data": {"new_dataset_id": ..., ...}}
    # or      {"status": "error",   "message": "..."}
    return api.filter_and_save_dataset(dataset_id, filter_expr)


# Manually register a derived DataFrame (lower-level, e.g. after a join)
from dataset_store import register_dataframe

def save_joined(df_joined, left_id: str, right_id: str) -> dict:
    meta = register_dataframe(
        df_joined,
        parent_dataset_id=left_id,
        kind="joined",
        transform={"left_dataset_id": left_id, "right_dataset_id": right_id},
    )
    return {"status": "success", "new_dataset_id": meta["dataset_id"]}
```

---

## Limitations

- **Single-process only.** The in-memory cache and index are not shared across worker processes. For multi-worker deployments, replace the cache with a distributed store (e.g. Redis) or rely solely on disk reload.
- **No persistence across restarts.** The index is lost when the process exits. Clients should re-upload datasets after a server restart.
- **No size limits enforced.** Very large CSVs will consume proportional RAM. Consider adding a file-size check in the upload handler before calling `load_uploaded_csv`.
- **CSV only.** Other tabular formats (Excel, Parquet, JSON) are not currently supported. Add entries to `ALLOWED_SUFFIXES` and extend `_read_csv_from_bytes` to add support.
