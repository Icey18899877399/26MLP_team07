import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.isolation_forest_optimized import OptimizedIsolationForestScratch


class OptimizedIsolationForestScratchTests(unittest.TestCase):
    def test_oblique_outliers_rank_above_diagonal_inliers(self):
        inliers = [[value / 10.0, value / 10.0] for value in range(-20, 21)]
        outliers = [[-1.5, 1.5], [-1.0, 1.0], [1.0, -1.0], [1.5, -1.5]]
        X = inliers + outliers

        model = OptimizedIsolationForestScratch(
            n_estimators=120,
            max_samples=32,
            contamination=len(outliers) / len(X),
            max_features=2,
            n_split_candidates=4,
            random_state=23,
        ).fit(X)

        inlier_mean = sum(model.scores_[: len(inliers)]) / len(inliers)
        outlier_mean = sum(model.scores_[len(inliers) :]) / len(outliers)
        self.assertGreater(outlier_mean, inlier_mean)
        self.assertGreaterEqual(model.labels_[-4:].count(-1), 3)

    def test_feature_subsampling_is_recorded_for_every_tree(self):
        X = [[float(i), float(i % 2), float(i % 3), float(i % 5)] for i in range(20)]
        model = OptimizedIsolationForestScratch(
            n_estimators=12,
            max_samples=12,
            max_features=0.5,
            random_state=31,
        ).fit(X)

        self.assertEqual(model.max_features_, 2)
        self.assertEqual(len(model.estimators_features_), 12)
        self.assertTrue(
            all(len(features) == 2 for features in model.estimators_features_)
        )
        self.assertTrue(
            all(len(tree.normal) == 2 for tree in model.estimators_)
        )
        self.assertTrue(
            all(
                isinstance(feature, int) and isinstance(weight, float)
                for tree in model.estimators_
                for feature, weight in tree.normal
            )
        )

    def test_extra_attempts_do_not_replace_an_already_valid_random_split(self):
        X = [[float(x), float(y)] for x in range(4) for y in range(4)]
        one_attempt = OptimizedIsolationForestScratch(
            n_estimators=10,
            max_samples=16,
            max_depth=1,
            max_features=2,
            n_split_candidates=1,
            random_state=5,
        ).fit(X)
        five_attempts = OptimizedIsolationForestScratch(
            n_estimators=10,
            max_samples=16,
            max_depth=1,
            max_features=2,
            n_split_candidates=5,
            random_state=5,
        ).fit(X)

        self.assertEqual(one_attempt.scores_, five_attempts.scores_)

    def test_fixed_seed_reproduces_optimized_scores(self):
        X = [[float(i), float((i * 7) % 11)] for i in range(24)]
        settings = dict(
            n_estimators=30,
            max_samples=16,
            contamination=0.15,
            max_features=1.0,
            n_split_candidates=3,
            random_state=47,
        )

        first = OptimizedIsolationForestScratch(**settings).fit(X)
        second = OptimizedIsolationForestScratch(**settings).fit(X)

        self.assertEqual(first.scores_, second.scores_)
        self.assertEqual(first.labels_, second.labels_)

    def test_decision_function_is_positive_for_normal_predictions(self):
        X = [[-0.2], [-0.1], [0.0], [0.1], [0.2], [6.0]]
        model = OptimizedIsolationForestScratch(
            n_estimators=80,
            max_samples=6,
            contamination=1.0 / 6.0,
            random_state=17,
        ).fit(X)

        decisions = model.decision_function([[0.0], [6.0]])

        self.assertGreater(decisions[0], 0.0)
        self.assertLessEqual(decisions[1], 0.0)

    def test_equal_scores_at_the_contamination_boundary_remain_normal(self):
        X = [[2.0, 2.0] for _ in range(8)]
        model = OptimizedIsolationForestScratch(
            n_estimators=8,
            max_samples=8,
            contamination=0.25,
            random_state=29,
        ).fit(X)

        self.assertEqual(model.labels_, [1] * 8)
        self.assertEqual(model.predict(X), [1] * 8)

    def test_constructor_rejects_invalid_optimization_parameters(self):
        with self.assertRaises(ValueError):
            OptimizedIsolationForestScratch(max_features=0)
        with self.assertRaises(ValueError):
            OptimizedIsolationForestScratch(max_features=1.5)
        with self.assertRaises(ValueError):
            OptimizedIsolationForestScratch(n_split_candidates=0)

    def test_optimized_example_runs_and_prints_one_detected_anomaly(self):
        script = PROJECT_ROOT / "examples" / "isolation_forest_optimized_minimal.py"

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[1, 1, 1, 1, 1, 1, -1]")


if __name__ == "__main__":
    unittest.main()
