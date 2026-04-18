"""Generate synthetic ``ab_test_events.csv`` rows for pipeline testing.

The real event log is produced by ``app_trt.py`` at runtime. Until the team has
collected enough real traffic, this script writes a synthetic log with a
planted effect (Group B has a modestly higher apply-success rate and more
preview-before-apply behaviour than Group A) so that ``ab_analysis.py`` and
``tests.py`` can be exercised end-to-end.

All rows produced by this script are clearly synthetic — the header CSV is
named ``ab_test_events_synthetic.csv`` by default and every row's ``details``
column is prefixed with ``SYNTHETIC:``.

Usage
-----
    python ab_seed_generator.py --n 200 --out ab_test_events_synthetic.csv --seed 20260418
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


def generate(n_sessions: int, seed: int = 20260418):
    rng = random.Random(seed)
    start = datetime(2026, 4, 18, 9, 30, 0)

    # Planted effect: Group B previews more and succeeds slightly more often.
    params = {
        "A": {"preview_lambda": 1.3, "apply_lambda": 1.7, "p_success": 0.78},
        "B": {"preview_lambda": 2.4, "apply_lambda": 1.6, "p_success": 0.88},
    }

    rows = []
    for _ in range(n_sessions):
        group = rng.choice(["A", "B"])
        p = params[group]
        sid = str(uuid.uuid4())
        session_start = start + timedelta(minutes=rng.uniform(0, 60 * 20))
        n_preview = max(0, int(rng.gauss(p["preview_lambda"], 1.0)))
        n_apply = max(0, int(rng.gauss(p["apply_lambda"], 0.9)))
        seconds = 0.0

        for _ in range(n_preview):
            seconds += rng.uniform(3, 25)
            rows.append(_row(session_start, seconds, sid, group, "preview_clean", rng, success=None))

        for _ in range(n_apply):
            seconds += rng.uniform(5, 30)
            success = rng.random() < p["p_success"]
            rows.append(_row(session_start, seconds, sid, group, "apply_clean", rng, success=success))

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _row(session_start, seconds, sid, group, event_type, rng, success):
    clean_action = rng.choice(CLEAN_ACTIONS)
    ts = session_start + timedelta(seconds=seconds)
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": sid,
        "ab_group": group,
        "event_type": event_type,
        "clean_action": clean_action,
        "dataset_key": "builtin_sleep",
        "columns_count": rng.randint(1, 5),
        "success": "" if success is None else str(success),
        "seconds_since_session_start": round(seconds, 3),
        "details": f"SYNTHETIC: {event_type} on {clean_action}",
    }


def write_csv(rows, out_path: Path) -> None:
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_SCHEMA)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in EVENT_SCHEMA})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of sessions to simulate")
    parser.add_argument("--out", default="ab_test_events_synthetic.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=20260418)
    args = parser.parse_args(argv)

    rows = generate(args.n, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} synthetic events for {args.n} sessions → {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
