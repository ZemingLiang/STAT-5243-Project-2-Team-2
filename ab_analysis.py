"""A/B-test analysis pipeline for STAT 5243 Project 3.

Reads the append-only event log produced by ``app_trt.py`` (``ab_test_events.csv``
by default), aggregates events to one row per session, compares Group A vs.
Group B on four pre-registered metrics, and emits a Markdown summary plus
matplotlib figures.

Usage
-----
    python ab_analysis.py ab_test_events.csv --out figures/ --seed 20260418

The module is also importable so ``tests.py`` can call ``session_metrics`` and
``compare_groups`` directly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVENT_SCHEMA = [
    "timestamp",
    "session_id",
    "ab_group",
    "event_type",
    "clean_action",
    "dataset_key",
    "columns_count",
    "success",
    "seconds_since_session_start",
    "details",
]

METRIC_FAMILY = [
    "apply_rate",
    "preview_to_apply_conversion",
    "apply_success_rate",
    "successful_actions_per_session",
]


@dataclass
class ComparisonResult:
    metric: str
    group_a_mean: float
    group_b_mean: float
    group_a_n: int
    group_b_n: int
    statistic: float
    p_value: float
    p_value_bonferroni: float
    effect: float
    effect_label: str
    ci_low: float
    ci_high: float
    test_name: str


def load_events(path: str | Path) -> pd.DataFrame:
    """Load an ``ab_test_events.csv`` log and normalise dtypes."""
    df = pd.read_csv(path)
    missing = [c for c in EVENT_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Log file is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["columns_count"] = pd.to_numeric(df["columns_count"], errors="coerce").fillna(0).astype(int)
    df["seconds_since_session_start"] = pd.to_numeric(df["seconds_since_session_start"], errors="coerce")
    df["success_bool"] = df["success"].map(_to_bool)
    df["ab_group"] = df["ab_group"].astype(str).str.upper().str.strip()
    df = df[df["ab_group"].isin(["A", "B"])].copy()
    return df


def _to_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def session_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse the event log to one row per ``session_id`` × ``ab_group``.

    Columns:
        session_id, ab_group, n_preview, n_apply, n_apply_success,
        session_duration_s, distinct_clean_actions, apply_rate,
        preview_to_apply_conversion, apply_success_rate,
        successful_actions_per_session.
    """
    if events.empty:
        return pd.DataFrame(columns=[
            "session_id", "ab_group", "n_preview", "n_apply", "n_apply_success",
            "session_duration_s", "distinct_clean_actions", *METRIC_FAMILY,
        ])

    grouped = events.groupby(["session_id", "ab_group"], dropna=False)
    rows = []
    for (sid, group), sub in grouped:
        n_preview = int((sub["event_type"] == "preview_clean").sum())
        n_apply = int((sub["event_type"] == "apply_clean").sum())
        n_apply_success = int(((sub["event_type"] == "apply_clean") & (sub["success_bool"] == True)).sum())
        duration = float(sub["seconds_since_session_start"].max() or 0.0)
        distinct_actions = int(sub["clean_action"].replace("", np.nan).dropna().nunique())

        total_events = n_preview + n_apply
        apply_rate = (n_apply / total_events) if total_events > 0 else 0.0
        preview_to_apply_conversion = 1.0 if (n_preview > 0 and n_apply > 0) else 0.0
        apply_success_rate = (n_apply_success / n_apply) if n_apply > 0 else 0.0
        successful_actions = n_apply_success

        rows.append({
            "session_id": sid,
            "ab_group": group,
            "n_preview": n_preview,
            "n_apply": n_apply,
            "n_apply_success": n_apply_success,
            "session_duration_s": duration,
            "distinct_clean_actions": distinct_actions,
            "apply_rate": apply_rate,
            "preview_to_apply_conversion": preview_to_apply_conversion,
            "apply_success_rate": apply_success_rate,
            "successful_actions_per_session": float(successful_actions),
        })
    return pd.DataFrame(rows)


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sb.mean() - sa.mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return (float(lo), float(hi))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if pooled == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def cohens_d_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap 95% CI on Cohen's *d* itself (the standardised effect size).

    Percentile bootstrap — resamples each group with replacement ``n_boot``
    times and reports the 2.5th and 97.5th percentiles of the resampled *d*
    distribution. Complements the existing mean-difference CI in
    ``_bootstrap_diff_ci``.
    """
    if rng is None:
        rng = np.random.default_rng(20260418)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    ds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        ds[i] = _cohens_d(sa, sb)
    lo, hi = np.quantile(ds, [0.025, 0.975])
    return (float(lo), float(hi))


def check_assumptions(a: np.ndarray, b: np.ndarray) -> dict:
    """Assumption checks for parametric two-sample comparison.

    Returns Shapiro-Wilk normality tests per group (W statistic + p-value)
    and Levene's test for equal variance across the pooled sample. The
    caller can decide whether to trust Welch's *t*-test or fall back to
    Mann-Whitney U based on the p-values reported here.
    """
    result = {"normality_a": {"W": float("nan"), "p": float("nan")},
              "normality_b": {"W": float("nan"), "p": float("nan")},
              "equal_variance": {"W": float("nan"), "p": float("nan")}}
    # Shapiro-Wilk caps at N=5000; sample if needed
    rng = np.random.default_rng(20260418)
    def _sw(x):
        if len(x) < 3:
            return float("nan"), float("nan")
        if len(x) > 5000:
            x = rng.choice(x, size=5000, replace=False)
        try:
            W, p = stats.shapiro(x)
            return float(W), float(p)
        except Exception:
            return float("nan"), float("nan")
    W_a, p_a = _sw(np.asarray(a, dtype=float))
    W_b, p_b = _sw(np.asarray(b, dtype=float))
    result["normality_a"] = {"W": W_a, "p": p_a}
    result["normality_b"] = {"W": W_b, "p": p_b}
    if len(a) >= 2 and len(b) >= 2:
        try:
            W_lev, p_lev = stats.levene(a, b, center="median")
            result["equal_variance"] = {"W": float(W_lev), "p": float(p_lev)}
        except Exception:
            pass
    return result


def fdr_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction on a list of raw p-values.

    Returns the adjusted p-values in the same order as the input.
    Less conservative than Bonferroni at the same family-wise level and
    is the industry-standard correction for A/B-test multi-metric families.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = p * n / ranks
    # Enforce monotonicity (BH step-up)
    sorted_p = adjusted[order]
    for i in range(n - 2, -1, -1):
        sorted_p[i] = min(sorted_p[i], sorted_p[i + 1])
    adjusted[order] = sorted_p
    return [float(min(x, 1.0)) for x in adjusted]


def tost_equivalence(
    a: np.ndarray,
    b: np.ndarray,
    bound: float,
    alpha: float = 0.05,
) -> dict:
    """Two One-Sided Tests (TOST) for equivalence of means.

    Follows Lakens (2017). The equivalence bounds are ``[-bound, +bound]``
    on the mean-difference scale. Rejects the null of "different means"
    (i.e., concludes equivalence) if the max of the two one-sided p-values
    is < ``alpha``. Useful for formally claiming that a null secondary
    metric genuinely shows no meaningful effect rather than merely
    "failing to reject".
    """
    if len(a) < 2 or len(b) < 2:
        return {"p_lower": float("nan"), "p_upper": float("nan"),
                "max_p": float("nan"), "equivalent": False,
                "bound": bound, "mean_diff": float("nan")}
    diff = float(b.mean() - a.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    if se == 0:
        return {"p_lower": 0.0, "p_upper": 0.0, "max_p": 0.0,
                "equivalent": True, "bound": bound, "mean_diff": diff}
    df = len(a) + len(b) - 2  # conservative
    t_lower = (diff - (-bound)) / se
    t_upper = (diff - bound) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)   # H0: diff <= -bound
    p_upper = stats.t.cdf(t_upper, df)        # H0: diff >= +bound
    max_p = float(max(p_lower, p_upper))
    return {"p_lower": float(p_lower), "p_upper": float(p_upper),
            "max_p": max_p, "equivalent": bool(max_p < alpha),
            "bound": float(bound), "mean_diff": diff}


def srm_test(session_df: pd.DataFrame, expected_a: float = 0.5) -> dict:
    """Sample Ratio Mismatch (SRM) χ² test.

    Industry-standard validity check (Microsoft / LinkedIn / Uber) that
    asks whether observed arm proportions are consistent with the
    planned 50/50 random assignment. A significant SRM (p < 0.01 by
    convention) indicates a bug in assignment, logging, or an unbalanced
    dropout pattern that invalidates downstream causal inference.
    """
    sessions = session_df.drop_duplicates("session_id")[["session_id", "ab_group"]]
    n_a = int((sessions["ab_group"] == "A").sum())
    n_b = int((sessions["ab_group"] == "B").sum())
    n = n_a + n_b
    if n == 0:
        return {"n_a": 0, "n_b": 0, "chi2": float("nan"),
                "p_value": float("nan"), "ratio_observed": float("nan"),
                "srm_flag": False}
    expected_a_n = n * expected_a
    expected_b_n = n * (1 - expected_a)
    chi2 = ((n_a - expected_a_n) ** 2 / expected_a_n
            + (n_b - expected_b_n) ** 2 / expected_b_n)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"n_a": n_a, "n_b": n_b, "chi2": float(chi2),
            "p_value": float(p_value),
            "ratio_observed": float(n_a / n),
            "srm_flag": bool(p_value < 0.01)}


def subgroup_analysis(
    events: pd.DataFrame,
    session_df: pd.DataFrame,
    subgroup_col: str = "clean_action",
    metric: str = "apply_rate",
    min_sessions_per_cell: int = 30,
) -> pd.DataFrame:
    """Stratified analysis of the primary metric by a subgroup column.

    For each level of ``subgroup_col`` in ``events``, assigns each
    session to the level it used most and runs the same Welch's t-test /
    Cohen's d comparison as ``compare_groups``. Levels with fewer than
    ``min_sessions_per_cell`` sessions per arm are skipped and reported
    as insufficient N.
    """
    # Assign each session to its modal subgroup value
    mode_by_session = (events.groupby("session_id")[subgroup_col]
                       .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None))
    session_with_mode = session_df.copy()
    session_with_mode[subgroup_col] = session_with_mode["session_id"].map(mode_by_session)

    rows = []
    for level in sorted(session_with_mode[subgroup_col].dropna().unique()):
        sub = session_with_mode[session_with_mode[subgroup_col] == level]
        a = sub.loc[sub["ab_group"] == "A", metric].to_numpy(dtype=float)
        b = sub.loc[sub["ab_group"] == "B", metric].to_numpy(dtype=float)
        n_a, n_b = len(a), len(b)
        if n_a < min_sessions_per_cell or n_b < min_sessions_per_cell:
            rows.append({"subgroup": level, "n_a": n_a, "n_b": n_b,
                         "mean_a": float(a.mean()) if n_a else float("nan"),
                         "mean_b": float(b.mean()) if n_b else float("nan"),
                         "cohens_d": float("nan"), "p_value": float("nan"),
                         "note": "insufficient N"})
            continue
        d = _cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"subgroup": level, "n_a": n_a, "n_b": n_b,
                     "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                     "cohens_d": float(d), "p_value": float(p),
                     "note": ""})
    return pd.DataFrame(rows)


def compare_groups(
    session_df: pd.DataFrame,
    metric: str,
    *,
    n_tests: int = 1,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> ComparisonResult:
    """Compare metric between Group A and Group B."""
    if metric not in session_df.columns:
        raise KeyError(f"metric {metric!r} not in session_df")

    rng = np.random.default_rng(seed)
    a = session_df.loc[session_df["ab_group"] == "A", metric].to_numpy(dtype=float)
    b = session_df.loc[session_df["ab_group"] == "B", metric].to_numpy(dtype=float)

    if metric == "preview_to_apply_conversion":
        # two-proportion z-test
        successes_a = int(a.sum())
        successes_b = int(b.sum())
        n_a = len(a)
        n_b = len(b)
        if n_a == 0 or n_b == 0:
            stat, p = float("nan"), float("nan")
        else:
            p_a = successes_a / n_a
            p_b = successes_b / n_b
            p_pool = (successes_a + successes_b) / (n_a + n_b)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
            stat = 0.0 if se == 0 else (p_b - p_a) / se
            p = 2 * (1 - stats.norm.cdf(abs(stat))) if np.isfinite(stat) else float("nan")
        effect = float(b.mean() - a.mean()) if n_a and n_b else float("nan")
        ci_low, ci_high = _bootstrap_diff_ci(a, b, n_boot, rng)
        return ComparisonResult(
            metric=metric,
            group_a_mean=float(a.mean()) if n_a else float("nan"),
            group_b_mean=float(b.mean()) if n_b else float("nan"),
            group_a_n=n_a,
            group_b_n=n_b,
            statistic=float(stat),
            p_value=float(p),
            p_value_bonferroni=min(1.0, float(p) * n_tests) if np.isfinite(p) else float("nan"),
            effect=effect,
            effect_label="proportion diff (B − A)",
            ci_low=ci_low,
            ci_high=ci_high,
            test_name="two-proportion z-test",
        )

    if metric == "successful_actions_per_session":
        # non-parametric, count-valued → Mann-Whitney U
        if len(a) == 0 or len(b) == 0:
            stat, p = float("nan"), float("nan")
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = _cohens_d(a, b)
        ci_low, ci_high = _bootstrap_diff_ci(a, b, n_boot, rng)
        return ComparisonResult(
            metric=metric,
            group_a_mean=float(a.mean()) if len(a) else float("nan"),
            group_b_mean=float(b.mean()) if len(b) else float("nan"),
            group_a_n=len(a),
            group_b_n=len(b),
            statistic=float(stat),
            p_value=float(p),
            p_value_bonferroni=min(1.0, float(p) * n_tests) if np.isfinite(p) else float("nan"),
            effect=d,
            effect_label="Cohen's d",
            ci_low=ci_low,
            ci_high=ci_high,
            test_name="Mann-Whitney U",
        )

    # default: Welch's t-test for continuous metrics (apply_rate, apply_success_rate)
    if len(a) < 2 or len(b) < 2:
        stat, p = float("nan"), float("nan")
    else:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
    d = _cohens_d(a, b)
    ci_low, ci_high = _bootstrap_diff_ci(a, b, n_boot, rng)
    return ComparisonResult(
        metric=metric,
        group_a_mean=float(a.mean()) if len(a) else float("nan"),
        group_b_mean=float(b.mean()) if len(b) else float("nan"),
        group_a_n=len(a),
        group_b_n=len(b),
        statistic=float(stat),
        p_value=float(p),
        p_value_bonferroni=min(1.0, float(p) * n_tests) if np.isfinite(p) else float("nan"),
        effect=d,
        effect_label="Cohen's d",
        ci_low=ci_low,
        ci_high=ci_high,
        test_name="Welch's t-test",
    )


def multi_metric_report(
    session_df: pd.DataFrame,
    metrics: Iterable[str] = METRIC_FAMILY,
    *,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run ``compare_groups`` over a family of metrics with Bonferroni correction."""
    metrics = list(metrics)
    results = [compare_groups(session_df, m, n_tests=len(metrics), n_boot=n_boot, seed=seed) for m in metrics]
    return pd.DataFrame([r.__dict__ for r in results])


def balance_check(events: pd.DataFrame) -> dict:
    """Sanity-check session assignment is roughly 50/50."""
    sessions = events.drop_duplicates("session_id")[["session_id", "ab_group"]]
    n_a = int((sessions["ab_group"] == "A").sum())
    n_b = int((sessions["ab_group"] == "B").sum())
    n = n_a + n_b
    if n == 0:
        return {"n_a": 0, "n_b": 0, "p_value": float("nan")}
    p = stats.binomtest(n_a, n, p=0.5).pvalue
    return {"n_a": n_a, "n_b": n_b, "p_value": float(p)}


def plot_apply_rate_by_group(session_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, color in [("A", "#94a3b8"), ("B", "#4361ee")]:
        vals = session_df.loc[session_df["ab_group"] == group, "apply_rate"].to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.55, label=f"Group {group} (n={len(vals)})", color=color, edgecolor="white")
    ax.set_xlabel("Per-session apply_rate = applies / (applies + previews)")
    ax.set_ylabel("Sessions")
    ax.set_title("Figure 1 — Per-session apply_rate by group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_preview_apply_funnel(session_df: pd.DataFrame, out_path: Path) -> None:
    stages = ["Sessions", "≥1 preview", "≥1 apply", "≥1 successful apply"]
    data = {}
    for group in ["A", "B"]:
        sub = session_df[session_df["ab_group"] == group]
        data[group] = [
            len(sub),
            int((sub["n_preview"] > 0).sum()),
            int((sub["n_apply"] > 0).sum()),
            int((sub["n_apply_success"] > 0).sum()),
        ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    x = np.arange(len(stages))
    ax.bar(x - width / 2, data["A"], width, label="Group A", color="#94a3b8", edgecolor="white")
    ax.bar(x + width / 2, data["B"], width, label="Group B", color="#4361ee", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Sessions")
    ax.set_title("Figure 2 — Preview→Apply funnel by group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_events_per_session(session_df: pd.DataFrame, out_path: Path) -> None:
    """Overlaid histogram of total events per session, by arm.

    A data-quality visualization showing the per-session engagement
    distribution matches realistic user behaviour (right-skewed, a small
    head of power users, a long tail). Used as Figure 5 to document data
    quality for the simulated dataset (§3.3.1).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sessions = session_df.copy()
    sessions["total_events"] = sessions["n_preview"] + sessions["n_apply"]
    for group, colour in [("A", "#94a3b8"), ("B", "#4361ee")]:
        vals = sessions.loc[sessions["ab_group"] == group, "total_events"].to_numpy()
        if len(vals) == 0:
            continue
        bin_edges = np.arange(0, int(vals.max()) + 2) - 0.5
        ax.hist(vals, bins=bin_edges, alpha=0.55, label=f"Group {group} (n={len(vals)})",
                color=colour, edgecolor="white")
    ax.set_xlabel("Total events per session (preview + apply)")
    ax.set_ylabel("Sessions")
    ax.set_title("Figure 5 — Per-session event count distribution by group")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_successful_actions_distribution(session_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [
        session_df.loc[session_df["ab_group"] == "A", "successful_actions_per_session"].to_numpy(),
        session_df.loc[session_df["ab_group"] == "B", "successful_actions_per_session"].to_numpy(),
    ]
    labels = [f"A (n={len(data[0])})", f"B (n={len(data[1])})"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#94a3b8", "#4361ee"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Successful cleaning actions per session")
    ax.set_title("Figure 3 — Successful-actions distribution by group")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _format_float(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def required_n_per_arm(
    effect_size_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Required sample size per arm for a two-sample, two-sided *t*-test.

    Uses the standard asymptotic z-approximation:
        n_per_arm = 2 * (z_{α/2} + z_{1-β})^2 / d^2
    Rounds up to the nearest integer. Good to 2–3 digits vs exact
    non-central *t* calculations for |d| > 0.2; sufficient for
    planning-stage sample size decisions.
    """
    if abs(effect_size_d) < 1e-6 or not np.isfinite(effect_size_d):
        return 10**9  # effectively infeasible
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_beta = float(stats.norm.ppf(power))
    n = 2.0 * (z_alpha + z_beta) ** 2 / (effect_size_d ** 2)
    return int(np.ceil(n))


def minimum_detectable_effect(
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Inverse of ``required_n_per_arm``: the smallest |Cohen's d|
    detectable at the given N, α, and power.
    """
    if n_per_arm < 2:
        return float("inf")
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_beta = float(stats.norm.ppf(power))
    return float((z_alpha + z_beta) * np.sqrt(2.0 / n_per_arm))


def plot_forest(results_df: pd.DataFrame, out_path: Path) -> None:
    """Forest plot of per-metric mean-difference effects with 95% bootstrap CIs.

    Standard industry visualization for A/B-test multi-metric results.
    Each row is a metric; the dot is the observed mean-difference
    ``B minus A`` point estimate, the horizontal bar is the 95 %
    percentile-bootstrap CI (already on the mean-difference scale in the
    ``ci_low`` / ``ci_high`` columns of the ComparisonResult), and a
    dashed vertical line at zero marks the no-effect reference. Colours
    rows green if the CI excludes zero (effect detected), grey if it
    straddles zero (null). Cohen's d is annotated next to each row for
    effect-magnitude context, but the x-axis scale is the raw mean
    difference so the error bars are consistent with the CI.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    metrics = list(results_df["metric"])
    lows = list(results_df["ci_low"])
    highs = list(results_df["ci_high"])
    a_means = list(results_df["group_a_mean"])
    b_means = list(results_df["group_b_mean"])
    d_values = list(results_df["effect"])  # Cohen's d for continuous; prop-diff for binary
    labels = [m.replace("_", " ") for m in metrics]

    y_positions = np.arange(len(metrics))[::-1]
    mean_diffs = [(b - a) if np.isfinite(a) and np.isfinite(b) else float("nan")
                  for a, b in zip(a_means, b_means)]

    for y, center, lo, hi, d_val in zip(y_positions, mean_diffs, lows, highs, d_values):
        if np.isfinite(lo) and np.isfinite(hi):
            colour = "#10b981" if (lo > 0 or hi < 0) else "#94a3b8"
        else:
            colour = "#94a3b8"
        # errorbar expects non-negative distances from center
        if np.isfinite(center) and np.isfinite(lo) and np.isfinite(hi):
            err_lo = max(0.0, center - lo)
            err_hi = max(0.0, hi - center)
            ax.errorbar(
                center, y, xerr=[[err_lo], [err_hi]],
                fmt="o", color=colour, capsize=5, markersize=10,
                linewidth=2.5, ecolor=colour,
            )
            # Annotate Cohen's d (or Δp for the binary) at the right edge
            x_text = hi + 0.01 * (max(highs) - min(lows) or 1.0)
            ax.annotate(
                f"d={d_val:+.2f}" if abs(d_val) < 5 else f"d={d_val:.1f}",
                xy=(x_text, y), ha="left", va="center", fontsize=9, color=colour,
            )
    ax.axvline(0, linestyle="--", color="#334155", alpha=0.6, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean difference (Group B minus Group A) with 95 % bootstrap CI")
    ax.set_title("Figure 4 — Forest plot of metric effects with 95 % CI")
    ax.grid(axis="x", alpha=0.3)
    # Expand x-limits slightly to make room for annotations
    x_min = min(lows) if all(np.isfinite(x) for x in lows) else -1
    x_max = max(highs) if all(np.isfinite(x) for x in highs) else 1
    span = x_max - x_min
    ax.set_xlim(x_min - 0.08 * span, x_max + 0.20 * span)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _power_welch_t(effect_size_d: float, n_a: int, n_b: int, alpha: float = 0.05) -> float:
    """Approximate two-sided Welch's t power for Cohen's d, balanced-equivalent formulation."""
    if n_a < 2 or n_b < 2 or not np.isfinite(effect_size_d):
        return float("nan")
    n_eff = 2.0 / (1.0 / n_a + 1.0 / n_b)
    ncp = effect_size_d * np.sqrt(n_eff / 2.0)
    df = n_a + n_b - 2
    crit = stats.t.ppf(1 - alpha / 2, df)
    power = (1 - stats.nct.cdf(crit, df, ncp)) + stats.nct.cdf(-crit, df, ncp)
    return float(max(0.0, min(1.0, power)))


METRIC_NUM = {
    "apply_rate": 1,
    "preview_to_apply_conversion": 2,
    "apply_success_rate": 3,
    "successful_actions_per_session": 4,
}


def compute_fill_values(
    events: pd.DataFrame,
    session_df: pd.DataFrame,
    results_df: pd.DataFrame,
    balance: dict,
    *,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> dict[str, str]:
    """Compute every ``{placeholder}`` → replacement string used in REPORT.md."""
    fill: dict[str, str] = {}
    rng = np.random.default_rng(seed)

    # --- §3.3 Sample table ----------------------------------------------------
    a = session_df[session_df["ab_group"] == "A"]
    b = session_df[session_df["ab_group"] == "B"]
    fill["N_A"] = str(len(a))
    fill["N_B"] = str(len(b))
    fill["P_A"] = str(int(a["n_preview"].sum()))
    fill["P_B"] = str(int(b["n_preview"].sum()))
    fill["Ap_A"] = str(int(a["n_apply"].sum()))
    fill["Ap_B"] = str(int(b["n_apply"].sum()))
    fill["S_A"] = str(int(((a["n_preview"] + a["n_apply"]) > 0).sum()))
    fill["S_B"] = str(int(((b["n_preview"] + b["n_apply"]) > 0).sum()))

    # --- §3.5 Randomisation balance ------------------------------------------
    fill["bal_p"] = _format_float(balance["p_value"])

    # --- §4.2 Descriptives + §4.3 Inferentials -------------------------------
    for metric, i in METRIC_NUM.items():
        a_vals = session_df.loc[session_df["ab_group"] == "A", metric].to_numpy(dtype=float)
        b_vals = session_df.loc[session_df["ab_group"] == "B", metric].to_numpy(dtype=float)
        mu_a = float(a_vals.mean()) if len(a_vals) else float("nan")
        mu_b = float(b_vals.mean()) if len(b_vals) else float("nan")
        sd_a = float(a_vals.std(ddof=1)) if len(a_vals) > 1 else float("nan")
        sd_b = float(b_vals.std(ddof=1)) if len(b_vals) > 1 else float("nan")
        diff = mu_b - mu_a
        lift_pct = (diff / mu_a * 100) if mu_a else float("nan")
        fill[f"mu_A_{i}"] = _format_float(mu_a)
        fill[f"mu_B_{i}"] = _format_float(mu_b)
        fill[f"sd_A_{i}"] = _format_float(sd_a)
        fill[f"sd_B_{i}"] = _format_float(sd_b)
        fill[f"lift_{i}"] = (
            f"{diff:+.4f}" + (f" ({lift_pct:+.1f}%)" if np.isfinite(lift_pct) else "")
        )

    # Inferential rows keyed by test type
    row_by_metric = {r["metric"]: r for _, r in results_df.iterrows()}
    # apply_rate: Welch's t (primary)
    r1 = row_by_metric["apply_rate"]
    fill["t_1"] = _format_float(r1["statistic"], 3)
    fill["p_1"] = _format_float(r1["p_value"])
    fill["p_1_adj"] = _format_float(r1["p_value_bonferroni"])
    fill["d_1"] = _format_float(r1["effect"])
    fill["lo_1"] = _format_float(r1["ci_low"])
    fill["hi_1"] = _format_float(r1["ci_high"])
    fill["dec_1"] = "reject H0" if r1["p_value_bonferroni"] < 0.05 else "fail to reject H0"

    # apply_rate robustness — Mann-Whitney U on the same column
    a_vals = session_df.loc[session_df["ab_group"] == "A", "apply_rate"].to_numpy(dtype=float)
    b_vals = session_df.loc[session_df["ab_group"] == "B", "apply_rate"].to_numpy(dtype=float)
    if len(a_vals) >= 2 and len(b_vals) >= 2:
        U_stat, pu = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    else:
        U_stat, pu = float("nan"), float("nan")
    fill["U_1"] = _format_float(U_stat, 1)
    fill["pu_1"] = _format_float(pu)
    fill["pu_1_adj"] = _format_float(min(1.0, pu * 4)) if np.isfinite(pu) else "—"
    fill["decu_1"] = (
        "reject H0" if np.isfinite(pu) and pu * 4 < 0.05 else "fail to reject H0"
    )

    # preview_to_apply_conversion — 2-prop z
    r2 = row_by_metric["preview_to_apply_conversion"]
    fill["z_2"] = _format_float(r2["statistic"], 3)
    fill["p_2"] = _format_float(r2["p_value"])
    fill["p_2_adj"] = _format_float(r2["p_value_bonferroni"])
    fill["dp_2"] = _format_float(r2["effect"])
    fill["lo_2"] = _format_float(r2["ci_low"])
    fill["hi_2"] = _format_float(r2["ci_high"])
    fill["dec_2"] = "reject H0" if r2["p_value_bonferroni"] < 0.05 else "fail to reject H0"

    # apply_success_rate — Welch's t
    r3 = row_by_metric["apply_success_rate"]
    fill["t_3"] = _format_float(r3["statistic"], 3)
    fill["p_3"] = _format_float(r3["p_value"])
    fill["p_3_adj"] = _format_float(r3["p_value_bonferroni"])
    fill["d_3"] = _format_float(r3["effect"])
    fill["lo_3"] = _format_float(r3["ci_low"])
    fill["hi_3"] = _format_float(r3["ci_high"])
    fill["dec_3"] = "reject H0" if r3["p_value_bonferroni"] < 0.05 else "fail to reject H0"

    # successful_actions_per_session — Mann-Whitney U
    r4 = row_by_metric["successful_actions_per_session"]
    fill["U_4"] = _format_float(r4["statistic"], 1)
    fill["p_4"] = _format_float(r4["p_value"])
    fill["p_4_adj"] = _format_float(r4["p_value_bonferroni"])
    fill["dec_4"] = "reject H0" if r4["p_value_bonferroni"] < 0.05 else "fail to reject H0"

    # --- §4.5 Power + sensitivity --------------------------------------------
    power = _power_welch_t(r1["effect"], r1["group_a_n"], r1["group_b_n"])
    fill["power_1"] = _format_float(power, 3)
    # Sensitivity: identical for this logger (we do not partition error-only sessions)
    fill["p_1_incl"] = _format_float(r1["p_value"]) + " (unchanged — log does not partition error-only sessions)"

    # --- §4.6 Statistical robustness -----------------------------------------
    # Cohen's d CI on the primary metric
    apply_a = session_df.loc[session_df["ab_group"] == "A", "apply_rate"].to_numpy(dtype=float)
    apply_b = session_df.loc[session_df["ab_group"] == "B", "apply_rate"].to_numpy(dtype=float)
    d_lo, d_hi = cohens_d_ci(apply_a, apply_b, n_boot=n_boot, rng=np.random.default_rng(seed))
    fill["d_1_lo"] = _format_float(d_lo)
    fill["d_1_hi"] = _format_float(d_hi)

    # Assumption checks for the primary metric
    assump = check_assumptions(apply_a, apply_b)
    fill["sw_W_a"] = _format_float(assump["normality_a"]["W"])
    fill["sw_p_a"] = _format_float(assump["normality_a"]["p"])
    fill["sw_W_b"] = _format_float(assump["normality_b"]["W"])
    fill["sw_p_b"] = _format_float(assump["normality_b"]["p"])
    fill["lev_W"] = _format_float(assump["equal_variance"]["W"])
    fill["lev_p"] = _format_float(assump["equal_variance"]["p"])
    # Human-readable decisions at alpha=0.05
    fill["sw_decision_a"] = (
        "normal" if (np.isfinite(assump["normality_a"]["p"])
                     and assump["normality_a"]["p"] >= 0.05) else "non-normal"
    )
    fill["sw_decision_b"] = (
        "normal" if (np.isfinite(assump["normality_b"]["p"])
                     and assump["normality_b"]["p"] >= 0.05) else "non-normal"
    )
    fill["lev_decision"] = (
        "equal variances supported"
        if (np.isfinite(assump["equal_variance"]["p"])
            and assump["equal_variance"]["p"] >= 0.05)
        else "unequal variances \u2014 Welch's t-test (used here) is robust to this"
    )

    # FDR-corrected p-values across the 4 primary-family tests
    raw_ps = [r1["p_value"], r2["p_value"], r3["p_value"], r4["p_value"]]
    fdr_ps = fdr_correct(raw_ps)
    fill["p_1_fdr"] = _format_float(fdr_ps[0])
    fill["p_2_fdr"] = _format_float(fdr_ps[1])
    fill["p_3_fdr"] = _format_float(fdr_ps[2])
    fill["p_4_fdr"] = _format_float(fdr_ps[3])

    # TOST equivalence test on the two null secondaries (apply_success_rate
    # and successful_actions_per_session). Use ±0.1 as the equivalence bound
    # on the mean-difference scale — a conservative threshold: differences
    # smaller than 10 % of the outcome scale are treated as practically
    # equivalent.
    succ_a = session_df.loc[session_df["ab_group"] == "A", "apply_success_rate"].to_numpy(dtype=float)
    succ_b = session_df.loc[session_df["ab_group"] == "B", "apply_success_rate"].to_numpy(dtype=float)
    tost3 = tost_equivalence(succ_a, succ_b, bound=0.1)
    fill["tost_3_lower_p"] = _format_float(tost3["p_lower"])
    fill["tost_3_upper_p"] = _format_float(tost3["p_upper"])
    fill["tost_3_decision"] = "equivalent" if tost3["equivalent"] else "not equivalent"

    acts_a = session_df.loc[session_df["ab_group"] == "A", "successful_actions_per_session"].to_numpy(dtype=float)
    acts_b = session_df.loc[session_df["ab_group"] == "B", "successful_actions_per_session"].to_numpy(dtype=float)
    tost4 = tost_equivalence(acts_a, acts_b, bound=0.3)  # 0.3 successful actions / session
    fill["tost_4_lower_p"] = _format_float(tost4["p_lower"])
    fill["tost_4_upper_p"] = _format_float(tost4["p_upper"])
    fill["tost_4_decision"] = "equivalent" if tost4["equivalent"] else "not equivalent"

    # SRM (Sample Ratio Mismatch) \u03c7\u00b2 test
    srm = srm_test(session_df)
    fill["srm_chi2"] = _format_float(srm["chi2"], 3)
    fill["srm_p"] = _format_float(srm["p_value"])
    fill["srm_ratio"] = _format_float(srm["ratio_observed"], 3)
    fill["srm_flag_text"] = (
        "no SRM detected" if not srm["srm_flag"]
        else "SRM DETECTED \u2014 assignment may be biased; interpret with caution"
    )

    # --- \u00a74.7 Subgroup analysis by cleaning action ---------------------------
    # Compute subgroup table if events are available (requires the caller to
    # supply them via the fill pipeline; we'll recompute it there since
    # compute_fill_values already has events). Produce a markdown table.
    if events is not None and len(events) > 0:
        sub_df = subgroup_analysis(events, session_df, "clean_action")
        lines = [
            "| Cleaning action | N (A / B) | Mean apply_rate (A / B) | Cohen's *d* | *p* (Welch) | Notes |",
            "|---|---|---|---|---|---|",
        ]
        em_dash = "\u2014"
        for _, row in sub_df.iterrows():
            note = row["note"] if row["note"] else em_dash
            lines.append(
                f"| `{row['subgroup']}` | {int(row['n_a'])} / {int(row['n_b'])} | "
                f"{_format_float(row['mean_a'])} / {_format_float(row['mean_b'])} | "
                f"{_format_float(row['cohens_d'])} | {_format_float(row['p_value'])} | "
                f"{note} |"
            )
        fill["subgroup_table"] = "\n".join(lines)
    else:
        fill["subgroup_table"] = "_(subgroup analysis unavailable \u2014 no events provided)_"

    # --- §2.6 Sample size / MDE placeholders --------------------------------
    # Target effect sizes and their required N at alpha=0.05, power=0.8
    n_for_small = required_n_per_arm(0.2)
    n_for_medium = required_n_per_arm(0.5)
    n_for_target = required_n_per_arm(0.4)
    # MDE at the N we actually have (per arm)
    n_per_arm_actual = min(r1["group_a_n"], r1["group_b_n"])
    mde_actual = minimum_detectable_effect(n_per_arm_actual)
    fill["n_for_small_effect"] = str(n_for_small)
    fill["n_for_medium_effect"] = str(n_for_medium)
    fill["n_for_target_effect"] = str(n_for_target)
    fill["n_per_arm_actual"] = str(n_per_arm_actual)
    fill["mde_actual"] = _format_float(mde_actual, 3)

    # --- §5 Narrative — auto-drafted from the numbers -------------------------
    fill["interp_paragraph_1"] = _draft_interpretation(r1, r2, r3, r4)
    fill["takeaway_1"] = _takeaway_primary(r1)
    fill["takeaway_2"] = _takeaway_secondary(r2, r3, r4)
    fill["takeaway_3"] = (
        f"The randomisation-balance p-value was {fill['bal_p']}, "
        f"indicating {'no' if balance['p_value'] >= 0.01 else 'possible'} deviation from a 50/50 split at α = 0.01."
    )
    fill["rollout_recommendation"] = _rollout(r1)

    return fill


def _draft_interpretation(r1, r2, r3, r4) -> str:
    primary_sig = r1["p_value_bonferroni"] < 0.05
    direction = "lower" if r1["effect"] < 0 else "higher"
    lines = []
    if primary_sig:
        lines.append(
            f"At the Bonferroni-corrected family-wise α = 0.05 level we reject H0 for the primary metric "
            f"`apply_rate` (Welch's t = {r1['statistic']:.3f}, Bonferroni-adjusted p = {r1['p_value_bonferroni']:.4f}). "
            f"Group B's mean apply_rate is {direction} than Group A's (Cohen's d = {r1['effect']:.3f}, "
            f"95 % bootstrap CI on the mean difference [{r1['ci_low']:.3f}, {r1['ci_high']:.3f}]), meaning "
            f"users in the guided treatment arm {'relied more on preview before applying' if r1['effect'] < 0 else 'applied more readily without previewing'} "
            f"than users in the control."
        )
    else:
        lines.append(
            f"At the Bonferroni-corrected family-wise α = 0.05 level we fail to reject H0 for the primary metric "
            f"`apply_rate` (Welch's t = {r1['statistic']:.3f}, Bonferroni-adjusted p = {r1['p_value_bonferroni']:.4f}, "
            f"Cohen's d = {r1['effect']:.3f}, 95 % CI [{r1['ci_low']:.3f}, {r1['ci_high']:.3f}]). The observed "
            f"difference between arms is consistent with noise at our sample size."
        )
    # Secondary metrics summary
    sig_secondaries = [
        (name, row) for name, row in [
            ("preview_to_apply_conversion", r2),
            ("apply_success_rate", r3),
            ("successful_actions_per_session", r4),
        ] if row["p_value_bonferroni"] < 0.05
    ]
    if sig_secondaries:
        names = ", ".join(f"`{n}`" for n, _ in sig_secondaries)
        lines.append(
            f"Among the secondary metrics, {names} also crossed the Bonferroni threshold, consistent with "
            f"the primary effect. The remaining secondary metrics did not reach significance after correction."
        )
    else:
        lines.append(
            "None of the secondary metrics crossed the Bonferroni threshold, so any treatment effect appears "
            "to be concentrated in the primary preview-vs-apply behaviour rather than in downstream success rate "
            "or throughput."
        )
    return " ".join(lines)


def _takeaway_primary(r1) -> str:
    if r1["p_value_bonferroni"] < 0.05:
        verb = "reduces" if r1["effect"] < 0 else "increases"
        return (
            f"The guided four-step Cleaning layout statistically significantly {verb} "
            f"the per-session `apply_rate` (Cohen's d = {r1['effect']:.3f})."
        )
    return (
        f"We could not detect a statistically significant effect of the guided layout on `apply_rate` "
        f"at this sample size (Cohen's d = {r1['effect']:.3f}, Bonferroni-adjusted p = {r1['p_value_bonferroni']:.4f})."
    )


def _takeaway_secondary(r2, r3, r4) -> str:
    parts = [
        f"`preview_to_apply_conversion` lift Δp = {r2['effect']:+.3f} (p_adj = {r2['p_value_bonferroni']:.4f})",
        f"`apply_success_rate` d = {r3['effect']:+.3f} (p_adj = {r3['p_value_bonferroni']:.4f})",
        f"`successful_actions_per_session` d = {r4['effect']:+.3f} (p_adj = {r4['p_value_bonferroni']:.4f})",
    ]
    return "Secondary-metric deltas: " + "; ".join(parts) + "."


def _rollout(r1) -> str:
    if r1["p_value_bonferroni"] < 0.05 and r1["effect"] < 0 and r1["ci_high"] < 0:
        return (
            "Recommend rolling Version B forward to all users. The primary effect is statistically "
            "significant after multiple-comparison correction, the direction favours increased "
            "preview-before-apply behaviour, and the 95 % bootstrap CI on the mean difference excludes zero."
        )
    if r1["p_value_bonferroni"] >= 0.05:
        return (
            "Do not roll Version B forward yet. Either collect more data to improve power, or iterate on "
            "the treatment (e.g., persistent banner, inline tutorial) before re-testing."
        )
    return (
        "Hold Version B in a follow-up test. The direction or confidence interval does not yet justify a "
        "full rollout; a second, longer run with user-scoped randomisation is recommended."
    )


def fill_report(template_path: str | Path, out_path: str | Path, fill_map: dict[str, str]) -> list[str]:
    """Read a Markdown template, substitute ``{placeholder}`` tokens, write the result.

    Returns a list of placeholder names that appeared in the template but had no mapping
    (so the user knows what still needs manual text).
    """
    template = Path(template_path).read_text(encoding="utf-8")

    import re
    pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
    unmatched: list[str] = []

    def _repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key in fill_map:
            return fill_map[key]
        unmatched.append(key)
        return match.group(0)

    filled = pattern.sub(_repl, template)
    Path(out_path).write_text(filled, encoding="utf-8")
    return sorted(set(unmatched))


def format_report(
    events: pd.DataFrame,
    session_df: pd.DataFrame,
    results: pd.DataFrame,
    balance: dict,
) -> str:
    lines = []
    lines.append("# A/B Analysis — STAT 5243 Project 3")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Total events: {len(events):,}")
    lines.append(f"- Sessions: Group A = {balance['n_a']}, Group B = {balance['n_b']}")
    lines.append(f"- Randomisation balance (two-sided binomial p vs 0.5): {_format_float(balance['p_value'])}")
    lines.append("")
    lines.append("## Metric comparisons (Bonferroni-corrected over the 4-metric family)")
    lines.append("")
    lines.append("| Metric | Test | Mean A | Mean B | Statistic | p (raw) | p (Bonferroni) | Effect | 95 % CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in results.iterrows():
        lines.append(
            f"| `{row['metric']}` | {row['test_name']} | "
            f"{_format_float(row['group_a_mean'])} | {_format_float(row['group_b_mean'])} | "
            f"{_format_float(row['statistic'])} | {_format_float(row['p_value'])} | "
            f"{_format_float(row['p_value_bonferroni'])} | "
            f"{_format_float(row['effect'])} ({row['effect_label']}) | "
            f"[{_format_float(row['ci_low'])}, {_format_float(row['ci_high'])}] |"
        )
    lines.append("")
    lines.append("Family-wise α = 0.05 (per-test threshold 0.0125 after Bonferroni).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B analysis for STAT 5243 Project 3")
    parser.add_argument("log_path", help="Path to ab_test_events.csv")
    parser.add_argument("--out", default="figures", help="Directory for output figures")
    parser.add_argument("--seed", type=int, default=20260418, help="Random seed for bootstrap")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--fill-template",
        default=None,
        help="Path to a Markdown template with {placeholder} tokens (e.g. REPORT.md)",
    )
    parser.add_argument(
        "--fill-out",
        default=None,
        help="Path to write the filled Markdown (defaults to <template>.filled.md)",
    )
    args = parser.parse_args(argv)

    events = load_events(args.log_path)
    session_df = session_metrics(events)
    balance = balance_check(events)
    results = multi_metric_report(session_df, seed=args.seed)

    report = format_report(events, session_df, results, balance)
    print(report)

    if not args.no_figures:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        plot_apply_rate_by_group(session_df, out / "apply_rate_by_group.png")
        plot_preview_apply_funnel(session_df, out / "preview_apply_funnel.png")
        plot_successful_actions_distribution(session_df, out / "successful_actions_distribution.png")
        plot_forest(results, out / "forest_plot.png")
        plot_events_per_session(session_df, out / "events_per_session.png")
        print(f"\nFigures written to {out.resolve()}/")

    if args.fill_template:
        template_path = Path(args.fill_template)
        out_path = Path(args.fill_out) if args.fill_out else template_path.with_suffix(".filled.md")
        fill_map = compute_fill_values(events, session_df, results, balance, seed=args.seed)
        unmatched = fill_report(template_path, out_path, fill_map)
        print(f"\nFilled template written to {out_path.resolve()}")
        if unmatched:
            print("Unmatched placeholders (left as-is):")
            for key in unmatched:
                print(f"  - {{{key}}}")
        else:
            print("All placeholders substituted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
