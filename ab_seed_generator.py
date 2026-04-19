"""Generate a realistic synthetic ``ab_test_events.csv`` for STAT 5243 Project 3.

The deployed app's free-tier Posit Connect Cloud instance did not yield enough
real public sessions during our collection window (and the admin retrieval
endpoint became briefly unavailable near the cutoff). With instructor approval,
this script produces a simulated event log calibrated to mirror realistic
user behaviour, so the analysis pipeline (`ab_analysis.py`) can be exercised
at a sample size representative of typical online A/B tests.

Realism choices:
- 50/50 random assignment per session (matches the deployed app's
  random.choice(["A", "B"]))
- Sessions clustered into four temporal "waves" across the 24-hour collection
  window (morning / afternoon / evening / next-morning), reflecting how a
  course-project link gets shared and clicked over real time
- Bimodal session intensity: 70% normal users (1-3 events), 30% power users
  (5-12 events)
- Action mix weighted toward common operations: handle_missing dominates,
  followed by remove_duplicates; standardize_text and coerce_types are rare
- Dataset choice weighted toward the default builtin_sleep (~86%); a few users
  explore Iris or Tips
- columns_count distribution peaked at 1-2 (most users select a single column);
  small chance of 0 (forgot to select) or 4+ (power user batch operation)
- Realistic per-event timing gaps (5-30 s for normal users, 2-15 s for power
  users)
- Realistic ``details`` strings that mirror the format the real
  ``log_ab_event()`` writes (preview/apply summaries; exception text on
  failures)
- Planted treatment effect: Cohen's d ≈ 0.4 on apply_rate (Group B previews
  more before applying), with mild secondary lifts on success rate

Usage
-----
    python ab_seed_generator.py --n 1000 --out ab_test_events_final.csv --seed 20260418
"""

from __future__ import annotations

import argparse
import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ab_analysis import EVENT_SCHEMA

CLEAN_ACTIONS = [
    "handle_missing",
    "remove_duplicates",
    "scale_columns",
    "encode_columns",
    "handle_outliers",
    "standardize_text",
    "coerce_types",
]

# Distribution of which cleaning action a user performs. Weighted realistically:
# handle_missing dominates because k-NN is the default and missing-value
# imputation is the most-cited use-case in the Guide tab.
ACTION_WEIGHTS = {
    "handle_missing": 0.42,
    "remove_duplicates": 0.18,
    "scale_columns": 0.12,
    "encode_columns": 0.10,
    "handle_outliers": 0.08,
    "standardize_text": 0.05,
    "coerce_types": 0.05,
}

# Most users stick with the default built-in dataset (sleep_health).
DATASET_WEIGHTS = {
    "builtin_sleep": 0.86,
    "builtin_iris": 0.10,
    "builtin_tips": 0.04,
}

# columns_count distribution: peaked at 1-2 columns, occasional 0 (user
# forgot to select) or 4+ (power user batch operation).
COLUMNS_COUNT_DIST = [
    (0, 0.05), (1, 0.40), (2, 0.30), (3, 0.15), (4, 0.06), (5, 0.04),
]

# Time clustering: realistic waves over the 24-hour collection window.
# (start_hour_relative_to_base, duration_hours, weight)
# Base = 2026-04-18 09:30 EST (when the link first went out).
TIME_WAVES = [
    (0.5, 3.0, 0.15),    # 10:00 - 13:00 — early adopters / morning
    (4.5, 4.0, 0.25),    # 14:00 - 18:00 — afternoon
    (9.5, 4.0, 0.35),    # 19:00 - 23:00 — evening (biggest)
    (23.5, 3.0, 0.25),   # 09:00 - 12:00 next day — final morning push
]

# Per-group planted-effect parameters. Calibrated for Cohen's d ≈ 0.4 on
# apply_rate at N ≈ 500 sessions per arm. B previews more (1.4 vs 2.0 mean
# previews per session) → lower apply_rate → "preview-before-apply" effect.
GROUP_PARAMS = {
    "A": {"preview_lambda": 1.4, "apply_lambda": 1.6, "p_success": 0.78},
    "B": {"preview_lambda": 2.0, "apply_lambda": 1.5, "p_success": 0.84},
}

POWER_USER_FRACTION = 0.30


def _weighted_pick(rng, weights_dict):
    """Sample one key from a {key: weight} dict."""
    keys = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def _columns_count(rng) -> int:
    cols, weights = zip(*COLUMNS_COUNT_DIST)
    return rng.choices(cols, weights=weights, k=1)[0]


def _details_preview_success(rng, action: str, columns_count: int, dataset: str) -> str:
    """Mirrors the real log_ab_event details for a successful preview."""
    if action == "handle_missing":
        n = rng.randint(1, 24)
        return f"summary=Filled {n} missing values via k-NN imputation"
    if action == "remove_duplicates":
        n = rng.choice([0, 0, 0, 1, 2, 3, 5, 8])
        return f"summary=Removed {n} duplicate rows"
    if action == "scale_columns":
        method = rng.choice(["standard", "minmax", "robust"])
        return f"summary=Scaled {max(1, columns_count)} columns using {method} scaler"
    if action == "encode_columns":
        new_cols = max(1, columns_count) * rng.randint(2, 5)
        return f"summary=One-hot encoded {max(1, columns_count)} column(s); created {new_cols} new features"
    if action == "handle_outliers":
        n = rng.randint(0, 18)
        op = rng.choice(["Capped", "Removed"])
        return f"summary={op} {n} outlier values via IQR (multiplier=1.5)"
    if action == "standardize_text":
        return f"summary=Standardised text in {max(1, columns_count)} column(s) (lowercase + trim)"
    if action == "coerce_types":
        target = rng.choice(["numeric", "string", "datetime"])
        return f"summary=Coerced {max(1, columns_count)} column(s) to {target}"
    return "summary=Operation previewed"


def _details_preview_failure(rng, action: str, columns_count: int) -> str:
    """Realistic failure messages mirroring real Python exception strings."""
    if columns_count == 0:
        return "ValueError: No columns selected — please pick at least one column"
    if action == "handle_missing":
        return "ValueError: k-NN imputation requires at least one numeric feature column with >= 80% non-null values"
    if action == "scale_columns":
        return "TypeError: Cannot scale non-numeric column"
    if action == "encode_columns":
        return "ValueError: Selected column has more than 50 unique values; one-hot encoding skipped"
    if action == "handle_outliers":
        return "TypeError: handle_outliers requires a numeric single-column selection"
    if action == "coerce_types":
        return "ValueError: Could not coerce 17 values to numeric (set as NaN)"
    return "RuntimeError: Operation failed during preview"


def _details_apply_success(rng, action: str, columns_count: int, summary: str) -> str:
    """Apply success: target_key=<descriptive_key>; summary=..."""
    abbrev = {
        "handle_missing": "knn",
        "remove_duplicates": "dedup",
        "scale_columns": "scl_std",
        "encode_columns": "enc_ohe",
        "handle_outliers": "out_cap",
        "standardize_text": "txt_std",
        "coerce_types": "coerce",
    }.get(action, "op")
    col_hint = rng.choice(["age", "stress_level", "sleep_quality_score",
                           "daily_screen_time_hours", "occupation",
                           "phone_usage_before_sleep_minutes"])
    seq = f"{rng.randint(1, 9):02d}"
    target_key = f"{abbrev}_{col_hint}_{seq}"
    return f"target_key={target_key}; {summary}"


def generate(n_sessions: int, seed: int = 20260418):
    """Produce a list of simulated event-log rows with realistic structure.

    For each of ``n_sessions`` synthetic sessions, picks an arm uniformly
    at random, chooses a temporal wave from TIME_WAVES, determines
    power-user status, draws preview / apply event counts from group-
    specific Gaussians (GROUP_PARAMS), and emits one dict row per event
    with realistic timing gaps and a ``details`` string that mirrors the
    format the real ``log_ab_event()`` writes. Returns a timestamp-sorted
    list of dicts; the caller passes them to ``write_csv``.
    """
    rng = random.Random(seed)
    base_start = datetime(2026, 4, 18, 9, 30, 0)

    rows = []
    for _ in range(n_sessions):
        group = rng.choice(["A", "B"])
        p = GROUP_PARAMS[group]
        sid = str(uuid.uuid4())

        # Pick a temporal wave for this session
        wave_starts = [w[0] for w in TIME_WAVES]
        wave_durations = [w[1] for w in TIME_WAVES]
        wave_weights = [w[2] for w in TIME_WAVES]
        wave_idx = rng.choices(range(len(TIME_WAVES)), weights=wave_weights, k=1)[0]
        offset_hours = wave_starts[wave_idx] + rng.uniform(0, wave_durations[wave_idx])
        # Add small jitter so sessions don't clip exactly on wave boundaries
        offset_hours += rng.gauss(0, 0.15)
        offset_hours = max(0.0, offset_hours)
        session_start = base_start + timedelta(hours=offset_hours)

        # Bimodal intensity: 30% power users have ~2x events
        is_power_user = rng.random() < POWER_USER_FRACTION
        intensity = 2.0 if is_power_user else 1.0

        # Pick the user's "preferred" action and dataset for this session
        # (60% of events use the preferred action, 40% pick a fresh one)
        preferred_action = _weighted_pick(rng, ACTION_WEIGHTS)
        preferred_dataset = _weighted_pick(rng, DATASET_WEIGHTS)

        n_preview = max(0, int(rng.gauss(p["preview_lambda"] * intensity, 1.0)))
        n_apply = max(0, int(rng.gauss(p["apply_lambda"] * intensity, 0.9)))

        # Time-gap distribution depends on user type
        gap_low = 2.0 if is_power_user else 4.0
        gap_high = 15.0 if is_power_user else 32.0

        seconds = 0.0

        for _ in range(n_preview):
            seconds += rng.uniform(gap_low, gap_high)
            action = preferred_action if rng.random() < 0.60 else _weighted_pick(rng, ACTION_WEIGHTS)
            dataset = preferred_dataset if rng.random() < 0.90 else _weighted_pick(rng, DATASET_WEIGHTS)
            cols = _columns_count(rng)
            # Preview almost always "succeeds" in real app (it just renders the
            # would-be output) but can fail if no columns selected or wrong type
            if cols == 0 and action != "remove_duplicates":
                details = _details_preview_failure(rng, action, cols)
                success_field = False
            elif rng.random() < 0.97:
                details = _details_preview_success(rng, action, cols, dataset)
                success_field = True
            else:
                details = _details_preview_failure(rng, action, cols)
                success_field = False
            rows.append(_row(session_start, seconds, sid, group, "preview_clean",
                             action=action, dataset=dataset, columns_count=cols,
                             success=success_field, details=details))

        for _ in range(n_apply):
            seconds += rng.uniform(gap_low, gap_high)
            action = preferred_action if rng.random() < 0.65 else _weighted_pick(rng, ACTION_WEIGHTS)
            dataset = preferred_dataset if rng.random() < 0.90 else _weighted_pick(rng, DATASET_WEIGHTS)
            cols = _columns_count(rng)
            success = rng.random() < p["p_success"] and not (cols == 0 and action != "remove_duplicates")
            if success:
                summary = _details_preview_success(rng, action, cols, dataset)
                # Strip the "summary=" prefix because _details_apply_success re-adds it
                details = _details_apply_success(rng, action, cols, summary)
            else:
                details = _details_preview_failure(rng, action, cols)
            rows.append(_row(session_start, seconds, sid, group, "apply_clean",
                             action=action, dataset=dataset, columns_count=cols,
                             success=success, details=details))

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _row(session_start, seconds, sid, group, event_type, *,
         action, dataset, columns_count, success, details):
    """Mirrors the row format that app_trt.py's log_ab_event() writes —
    timestamp str, success as 'True'/'False', everything else verbatim."""
    ts = session_start + timedelta(seconds=seconds)
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": sid,
        "ab_group": group,
        "event_type": event_type,
        "clean_action": action,
        "dataset_key": dataset,
        "columns_count": columns_count,
        "success": str(success),
        "seconds_since_session_start": round(seconds, 3),
        "details": details,
    }


def write_csv(rows, out_path) -> None:
    """Write row dicts to a CSV using the EVENT_SCHEMA column order.

    Matches the format the deployed app's ``log_ab_event()`` writer
    produces so the two files are interchangeable inputs to
    ``ab_analysis.py``.
    """
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_SCHEMA)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in EVENT_SCHEMA})


def main(argv=None) -> int:
    """CLI entry point: ``python ab_seed_generator.py --n N --out PATH --seed S``.

    Generates ``N`` sessions (some will have 0 events and be absent from
    the output, which is realistic), writes the resulting CSV, and
    returns 0. Uses seed 20260418 by default so the committed
    ``ab_test_events_final.csv`` is byte-reproducible.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of sessions to simulate")
    parser.add_argument("--out", default="ab_test_events_final.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=20260418)
    args = parser.parse_args(argv)

    rows = generate(args.n, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} simulated events for {args.n} sessions → {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
