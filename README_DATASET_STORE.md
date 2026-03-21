# Dataset Store — Reference

`dataset_store.py` manages the lifecycle of user-uploaded datasets: receiving a file, persisting it to disk, caching the parsed DataFrame in RAM, and making it retrievable by a stable `dataset_id`.

EDA functions in `EDA.py` receive a plain `pd.DataFrame` and have no knowledge of this layer. The API layer is responsible for translating a `dataset_id` from an HTTP request into a DataFrame by calling `get_dataset_by_id(dataset_id)`.

---

## Storage model

```
User uploads CSV
    ↓
load_uploaded_csv(file_obj, filename)
    ↓  validates, assigns dataset_id (UUID4)
    ↓  optionally writes raw bytes to ./uploaded_datasets/<id>__<filename>
    ↓  parses into pd.DataFrame
    ↓  stores DatasetMetadata in _DATASET_INDEX
    ↓  optionally caches DataFrame in _DATASET_CACHE
    → returns metadata dict (incl. dataset_id)

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

### `get_dataset_metadata(dataset_id) → dict`

Return the stored `DatasetMetadata` as a JSON-friendly dict.

Fields: `dataset_id`, `filename`, `stored_path`, `file_type`, `created_at` (ISO 8601 UTC), `n_rows`, `n_cols`, `columns`, `dtypes`, `size_bytes`.

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
```

---

## Limitations

- **Single-process only.** The in-memory cache and index are not shared across worker processes. For multi-worker deployments, replace the cache with a distributed store (e.g. Redis) or rely solely on disk reload.
- **No persistence across restarts.** The index is lost when the process exits. Clients should re-upload datasets after a server restart.
- **No size limits enforced.** Very large CSVs will consume proportional RAM. Consider adding a file-size check in the upload handler before calling `load_uploaded_csv`.
- **CSV only.** Other tabular formats (Excel, Parquet, JSON) are not currently supported. Add entries to `ALLOWED_SUFFIXES` and extend `_read_csv_from_bytes` to add support.
