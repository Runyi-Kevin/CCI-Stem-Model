"""Unit and integration tests (stdlib unittest; also runnable with pytest)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stem_model import analysis as A, config as C, scoring as S, sensitivity as X, validation as V  # noqa: E402


def make_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    att = rng.integers(4, 11, n)
    return pd.DataFrame({
        "participant_id": [f"STEM-{i:03d}" for i in range(1, n + 1)],
        "team_id": [f"TEAM-{i % 6:02d}" for i in range(n)],
        "sessions_attended": att, "sessions_on_time": [rng.integers(1, a + 1) for a in att],
        "pre_test": rng.integers(20, 70, n), "post_test": rng.integers(30, 100, n),
        "participation": rng.integers(30, 100, n), "teamwork": rng.integers(30, 100, n),
        "group_discussion": rng.integers(20, 100, n)})


class TestWeights(unittest.TestCase):
    def test_baseline_sums_to_one(self):
        self.assertAlmostEqual(sum(C.BASELINE_WEIGHTS.values()), 1.0)
        for w in C.SCENARIOS.values():
            self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_invalid_weights_rejected(self):
        bad = dict(C.BASELINE_WEIGHTS, knowledge_gain=0.5)
        with self.assertRaises(ValueError):
            S.validate_weights(bad)
        with self.assertRaises(ValueError):
            S.validate_weights({"teamwork": 1.0})
        with self.assertRaises(ValueError):
            S.validate_weights(dict(C.BASELINE_WEIGHTS, teamwork=-0.15, group_discussion=0.45))


class TestScoring(unittest.TestCase):
    def row(self, **kw):
        base = dict(participant_id="STEM-001", team_id="T", sessions_attended=8, sessions_on_time=8,
                    pre_test=45, post_test=43, participation=79, teamwork=67, group_discussion=60)
        base.update(kw)
        return pd.DataFrame([base])

    def test_hand_calculated_composite_decline_case(self):
        t = S.build_scored_table(self.row())          # gain floored to 0
        self.assertAlmostEqual(t.composite[0], 48.7, places=6)
        self.assertEqual(t.band[0], "Needs Improvement")

    def test_hand_calculated_composite_high_case(self):
        t = S.build_scored_table(self.row(sessions_attended=10, sessions_on_time=9, pre_test=52, post_test=86,
                                          participation=84, teamwork=81, group_discussion=80))
        self.assertAlmostEqual(t.composite[0], 79.6, places=6)
        self.assertEqual(t.band[0], "Very Good")

    def test_pre_test_ceiling_and_zero_attendance_do_not_divide_by_zero(self):
        t = S.compute_indicators(self.row(pre_test=100, post_test=100, sessions_attended=0, sessions_on_time=0))
        self.assertEqual(t.knowledge_gain[0], 0.0)
        self.assertEqual(t.attendance_punctuality[0], 0.0)

    def test_perfect_participant_scores_100(self):
        t = S.build_scored_table(self.row(sessions_attended=10, sessions_on_time=10, pre_test=0, post_test=100,
                                          participation=100, teamwork=100, group_discussion=100))
        self.assertAlmostEqual(t.composite[0], 100.0)

    def test_band_boundaries_inclusive_lower(self):
        s = pd.Series([49.99, 50, 59.99, 60, 70, 80, 100])
        self.assertEqual(list(S.assign_band(s)), ["Needs Improvement", "Fair", "Fair", "Good", "Very Good", "Excellent", "Excellent"])

    def test_ranking_ties_share_best_rank(self):
        self.assertEqual(list(S.rank_scores(pd.Series([90, 80, 80, 70]))), [1, 2, 2, 4])

    def test_scores_bounded(self):
        t = S.build_scored_table(make_df())
        self.assertTrue(t.composite.between(0, 100).all())


class TestValidation(unittest.TestCase):
    def test_clean_data_passes(self):
        _, chk = V.validate(make_df(), expected_n=60)
        self.assertFalse((chk.Status == "FAIL").any())

    def test_duplicate_ids_fail(self):
        d = make_df(); d.loc[1, "participant_id"] = d.loc[0, "participant_id"]
        with self.assertRaises(V.DataValidationError):
            V.validate(d, expected_n=60)

    def test_on_time_exceeding_attended_fails(self):
        d = make_df(); d.loc[0, "sessions_on_time"] = d.loc[0, "sessions_attended"] + 1
        with self.assertRaises(V.DataValidationError):
            V.validate(d, expected_n=60)

    def test_out_of_range_and_missing_fail(self):
        d = make_df(); d.loc[0, "teamwork"] = 140
        with self.assertRaises(V.DataValidationError):
            V.validate(d, expected_n=60)
        d = make_df(); d.loc[3, "pre_test"] = np.nan
        with self.assertRaises(V.DataValidationError):
            V.validate(d, expected_n=60)

    def test_non_numeric_text_detected(self):
        d = make_df().astype({"post_test": object}); d.loc[2, "post_test"] = "abc"
        _, chk = V.validate(d, expected_n=60, strict=False)
        self.assertEqual(chk.loc[chk.Check == "Numeric data types", "Status"].iloc[0], "FAIL")

    def test_wrong_row_count_fails(self):
        with self.assertRaises(V.DataValidationError):
            V.validate(make_df(10), expected_n=60)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            V.load_raw("does_not_exist.xlsx")


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = S.build_scored_table(make_df(120, 1))

    def test_pca_variance_sums_to_100_and_is_deterministic(self):
        a, b = A.pca_analysis(self.sc), A.pca_analysis(self.sc)
        self.assertAlmostEqual(a["variance"]["Variance %"].sum(), 100.0)
        pd.testing.assert_frame_equal(a["loadings"], b["loadings"])

    def test_kmo_in_unit_interval(self):
        self.assertTrue(0 < A.bartlett_kmo(self.sc[C.INDICATORS])["KMO (overall)"] < 1)

    def test_holm_adjustment_monotone_and_bounded(self):
        p = np.array([0.01, 0.04, 0.03, 0.5]); adj = A.holm(p)
        self.assertTrue((adj >= p).all() and (adj <= 1).all())

    def test_kmeans_labels_complete_and_reproducible(self):
        Z = A.pca_analysis(self.sc)["Z"]
        c1, c2 = A.fit_clusters(self.sc, Z, 3), A.fit_clusters(self.sc, Z, 3)
        self.assertEqual(set(c1["labels"]), {1, 2, 3}); np.testing.assert_array_equal(c1["labels"], c2["labels"])
        means = c1["profile"]["composite"].to_numpy()
        self.assertTrue((np.diff(means) > 0).all())        # labels ordered by performance level


class TestSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = S.build_scored_table(make_df(100, 2))

    def test_baseline_scenario_is_identity(self):
        r = X.scenario_analysis(self.sc).iloc[0]
        self.assertAlmostEqual(r["Spearman rho"], 1.0); self.assertEqual(r["Max |rank shift|"], 0)

    def test_oat_weights_valid_and_rho_bounded(self):
        o = X.oat_analysis(self.sc)
        self.assertTrue(o["Spearman rho"].between(-1, 1).all()); self.assertFalse(o.empty)

    def test_dirichlet_draws_sum_to_one(self):
        mc = X.monte_carlo(self.sc)
        np.testing.assert_allclose(mc["weights"].sum(axis=1), 1.0)

    def test_zero_noise_limit_recovers_ranking(self):
        C.NOISE_SDS, old = [0], C.NOISE_SDS
        try:
            self.assertAlmostEqual(X.noise_robustness(self.sc)["Mean Spearman rho"][0], 1.0)
        finally:
            C.NOISE_SDS = old

    def test_reproducibility_check(self):
        df = make_df(50, 3)
        self.assertTrue(X.reproducibility_check(lambda: S.build_scored_table(df))["identical"])


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_on_real_input_if_present(self):
        src = ROOT / "data" / "STEM_Participant_Data_450_Synthetic.xlsx"
        if not src.exists():
            self.skipTest("input workbook not present")
        from stem_model import pipeline
        saved = (C.MC_DRAWS, C.NOISE_REPS, C.BOOTSTRAPS)
        C.MC_DRAWS, C.NOISE_REPS, C.BOOTSTRAPS = 100, 10, 5      # keep the test fast
        try:
            with tempfile.TemporaryDirectory() as d:
                rc = pipeline.main(["--input", str(src), "--output", d])
                self.assertEqual(rc, 0)
                for f in ("STEM_Model_Results.xlsx", "Technical_Report.md", "run_manifest.json", "scored_participants.csv"):
                    self.assertTrue((Path(d) / f).exists(), f)
        finally:
            C.MC_DRAWS, C.NOISE_REPS, C.BOOTSTRAPS = saved

    def test_pipeline_returns_error_code_on_missing_input(self):
        from stem_model import pipeline
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(pipeline.main(["--input", "nope.xlsx", "--output", d]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
