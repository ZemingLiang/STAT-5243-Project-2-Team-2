from __future__ import annotations

import csv
import random
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import eda as EDA
import feature_engineering
import data_cleaning as cleaning
from app import app, load_builtin_dataset

from ab_analysis import (
    EVENT_SCHEMA,
    METRIC_FAMILY,
    balance_check,
    compare_groups,
    compute_fill_values,
    fill_report,
    load_events,
    multi_metric_report,
    session_metrics,
)
import ab_seed_generator


TEST_DATA_PATH = "test_data/sleep_mobile_stress_dataset_15000.csv"


class LocalIntegrationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(TEST_DATA_PATH)

    def test_shiny_app_imports(self) -> None:
        self.assertIsNotNone(app)

    def test_builtin_loaders(self) -> None:
        sleep_df = load_builtin_dataset("sleep_health")
        iris_df = load_builtin_dataset("iris")
        self.assertGreater(len(sleep_df), 0)
        self.assertGreater(len(iris_df), 0)

    def test_cleaning_module(self) -> None:
        transformed = cleaning.remove_duplicates(self.df)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertGreater(len(transformed), 0)

    def test_feature_engineering_module(self) -> None:
        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            self.df,
            "interaction",
            "daily_screen_time_hours",
            col2="stress_level",
        )
        self.assertIn("daily_screen_time_hours_x_stress_level", transformed.columns)
        self.assertEqual(meta["feature_type"], "interaction")

    def test_eda_summaries(self) -> None:
        head_payload = EDA.show_head(self.df, n=5)
        describe_payload = EDA.describe_dataframe(self.df)
        types_payload = EDA.column_types(self.df)
        self.assertEqual(head_payload["status"], "success")
        self.assertEqual(describe_payload["status"], "success")
        self.assertEqual(types_payload["status"], "success")

    def test_eda_correlation_matrix(self) -> None:
        payload = EDA.correlation_matrix(self.df)
        self.assertEqual(payload["status"], "success")
        self.assertIn("values", payload["data"])
        self.assertGreater(len(payload["data"]["columns"]), 1)

    def test_knn_imputation(self) -> None:
        # Introduce some NaN values to test k-NN imputation
        df = self.df.copy()
        df.loc[0:4, "age"] = None
        result, warning = cleaning.knn_impute(df, ["age"], k=3)
        self.assertIsInstance(result, pd.DataFrame)
        # All NaN values in 'age' should be filled
        self.assertEqual(result["age"].isnull().sum(), 0)

    def test_custom_expression(self) -> None:
        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            self.df,
            "custom_expr",
            expr="age * 2 + stress_level",
            new_column="test_custom",
        )
        self.assertIn("test_custom", transformed.columns)
        self.assertEqual(meta["feature_type"], "custom_expr")

    def test_eda_plots(self) -> None:
        one_d = EDA.plot_numeric_1d(self.df, "age", bins=12)
        two_d = EDA.plot_two_columns(self.df, "age", "stress_level", kind="scatter")
        regression = EDA.regression_analysis(
            self.df,
            "daily_screen_time_hours",
            "stress_level",
        )
        multiline = EDA.plot_multiline(
            self.df,
            "sleep_duration_hours",
            group_by="gender",
            nbins=10,
        )
        for payload in [one_d, two_d, regression, multiline]:
            self.assertEqual(payload["status"], "success")


class ABLoggingSchemaTest(unittest.TestCase):
    """Verify ab_test_events.csv writer schema matches app_trt.py's schema."""

    def test_schema_matches_app_trt(self) -> None:
        # The event schema is the contract between app_trt.py's log_ab_event
        # and ab_analysis.load_events — any drift breaks analysis.
        expected = [
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
        self.assertEqual(EVENT_SCHEMA, expected)

    def test_writer_produces_required_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"
            rows = ab_seed_generator.generate(n_sessions=4, seed=1)
            ab_seed_generator.write_csv(rows, path)
            with path.open() as f:
                header = next(csv.reader(f))
            self.assertEqual(header, EVENT_SCHEMA)


class ABRandomisationBalanceTest(unittest.TestCase):
    """The A/B assignment uses random.choice(['A','B']) per session.

    Over many draws the split should be close to 50/50.
    """

    def test_assignment_balance_over_1000_draws(self) -> None:
        rng = random.Random(42)
        draws = [rng.choice(["A", "B"]) for _ in range(1_000)]
        share_a = draws.count("A") / len(draws)
        self.assertAlmostEqual(share_a, 0.5, delta=0.04)

    def test_balance_check_on_synthetic_log(self) -> None:
        rows = ab_seed_generator.generate(n_sessions=300, seed=7)
        events = pd.DataFrame(rows)
        events["success_bool"] = events["success"].map(
            lambda v: True if str(v).lower() == "true" else False if str(v).lower() == "false" else None
        )
        balance = balance_check(events)
        self.assertGreater(balance["p_value"], 0.01,
                           msg="Randomisation balance should not be far from 50/50")


class ABSessionMetricsTest(unittest.TestCase):
    """session_metrics should collapse events to one row per session."""

    def _make_events(self) -> pd.DataFrame:
        rows = [
            _evt("s1", "A", "preview_clean", success=None, sec=5),
            _evt("s1", "A", "apply_clean", success=True, sec=10),
            _evt("s2", "B", "preview_clean", success=None, sec=3),
            _evt("s2", "B", "preview_clean", success=None, sec=8),
            _evt("s2", "B", "apply_clean", success=True, sec=12),
            _evt("s3", "B", "apply_clean", success=False, sec=2),
        ]
        df = pd.DataFrame(rows)
        df["success_bool"] = df["success"].map(
            lambda v: True if v == "True" else False if v == "False" else None
        )
        return df

    def test_metric_values(self) -> None:
        df = self._make_events()
        sessions = session_metrics(df).set_index("session_id")

        # s1: 1 preview + 1 successful apply
        self.assertAlmostEqual(sessions.at["s1", "apply_rate"], 0.5)
        self.assertEqual(sessions.at["s1", "preview_to_apply_conversion"], 1.0)
        self.assertAlmostEqual(sessions.at["s1", "apply_success_rate"], 1.0)
        self.assertEqual(sessions.at["s1", "successful_actions_per_session"], 1.0)

        # s2: 2 previews + 1 successful apply
        self.assertAlmostEqual(sessions.at["s2", "apply_rate"], 1 / 3)
        self.assertEqual(sessions.at["s2", "preview_to_apply_conversion"], 1.0)

        # s3: 1 failed apply, no preview
        self.assertAlmostEqual(sessions.at["s3", "apply_rate"], 1.0)
        self.assertEqual(sessions.at["s3", "preview_to_apply_conversion"], 0.0)
        self.assertAlmostEqual(sessions.at["s3", "apply_success_rate"], 0.0)


class ABCompareGroupsTest(unittest.TestCase):
    """compare_groups should detect a planted effect in the seed data."""

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.csv"
            rows = ab_seed_generator.generate(n_sessions=400, seed=20260418)
            ab_seed_generator.write_csv(rows, path)
            cls.events = load_events(path)
        cls.session_df = session_metrics(cls.events)

    def test_session_df_covers_both_groups(self) -> None:
        self.assertIn("A", self.session_df["ab_group"].unique())
        self.assertIn("B", self.session_df["ab_group"].unique())

    def test_detects_planted_effect_on_apply_rate(self) -> None:
        # The seed generator plants a difference on apply_rate: B previews more,
        # so B's apply_rate should be *lower* than A's. The test is two-sided.
        result = compare_groups(
            self.session_df, "apply_rate", n_tests=len(METRIC_FAMILY), seed=1
        )
        self.assertLess(result.p_value, 0.05,
                        msg=f"Expected planted effect to be detectable (p={result.p_value})")

    def test_multi_metric_report_shape(self) -> None:
        report = multi_metric_report(self.session_df, seed=1)
        self.assertEqual(len(report), len(METRIC_FAMILY))
        for col in ["metric", "p_value", "p_value_bonferroni", "effect"]:
            self.assertIn(col, report.columns)


class ABFillReportTest(unittest.TestCase):
    """fill_report should substitute every {placeholder} defined by compute_fill_values
    and return a sorted list of any placeholders left unresolved."""

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.csv"
            rows = ab_seed_generator.generate(n_sessions=200, seed=42)
            ab_seed_generator.write_csv(rows, path)
            cls.events = load_events(path)
        cls.session_df = session_metrics(cls.events)
        cls.balance = balance_check(cls.events)
        cls.results = multi_metric_report(cls.session_df, seed=1)
        cls.fill = compute_fill_values(
            cls.events, cls.session_df, cls.results, cls.balance, seed=1,
        )

    def test_fill_contains_core_placeholders(self) -> None:
        for key in ["N_A", "N_B", "bal_p", "mu_A_1", "mu_B_1", "p_1", "p_1_adj",
                    "dec_1", "power_1", "interp_paragraph_1", "rollout_recommendation"]:
            self.assertIn(key, self.fill, msg=f"Missing fill key: {key}")

    def test_fill_on_real_report_template_has_no_unmatched(self) -> None:
        template_path = Path(__file__).parent / "REPORT.md"
        if not template_path.exists():
            self.skipTest("REPORT.md not found")
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "REPORT.filled.md"
            unmatched = fill_report(template_path, out_path, self.fill)
            self.assertEqual(
                unmatched, [],
                msg=f"Unmatched placeholders in REPORT.md: {unmatched}"
            )
            filled = out_path.read_text(encoding="utf-8")
            # No {token} tokens should remain in the filled output
            import re
            leftovers = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", filled)
            self.assertEqual(leftovers, [], msg=f"Leftover tokens: {leftovers}")


class ABAdminDownloadTest(unittest.TestCase):
    """Contract test for the password-gated event-log download.

    The Shiny server logic is hard to exercise directly without a running
    browser session. Instead we check the contract the code depends on:
    a non-empty password constant exists, the server references it when
    gating the download, and the handler refuses to yield bytes without it.
    """

    def test_admin_password_constant_exists(self) -> None:
        import app_trt
        self.assertTrue(hasattr(app_trt, "AB_ADMIN_PASSWORD"))
        self.assertIsInstance(app_trt.AB_ADMIN_PASSWORD, str)
        self.assertGreater(len(app_trt.AB_ADMIN_PASSWORD), 6,
                           msg="Password must be non-trivial")

    def test_download_handler_checks_password(self) -> None:
        """Both the UI renderer and the download handler must gate on the password."""
        source = Path(__file__).parent.joinpath("app_trt.py").read_text()
        self.assertIn("admin_pwd", source)
        self.assertIn("AB_ADMIN_PASSWORD", source)
        # The download handler must check the password and return early otherwise
        import re
        handler_match = re.search(
            r"def download_ab_events\(\):(.*?)(?=\n    @|\n    def )",
            source, re.DOTALL
        )
        self.assertIsNotNone(handler_match, msg="download_ab_events handler not found")
        body = handler_match.group(1)
        self.assertIn("AB_ADMIN_PASSWORD", body,
                      msg="download_ab_events must guard on the password")
        self.assertIn("return", body,
                      msg="download_ab_events must early-return when password is wrong")


def _evt(session_id, group, event_type, *, success, sec):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "ab_group": group,
        "event_type": event_type,
        "clean_action": "handle_missing",
        "dataset_key": "builtin_sleep",
        "columns_count": 1,
        "success": "" if success is None else str(success),
        "seconds_since_session_start": sec,
        "details": "",
    }


if __name__ == "__main__":
    unittest.main()
