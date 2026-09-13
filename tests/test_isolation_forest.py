import math
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.isolation_forest import IsolationForestScratch, average_path_length


class IsolationForestScratchTests(unittest.TestCase):
    def test_average_path_length_matches_the_isolation_forest_formula(self):
        self.assertEqual(average_path_length(1), 0.0)
        self.assertEqual(average_path_length(2), 1.0)
        self.assertAlmostEqual(average_path_length(3), 5.0 / 3.0)

    def test_extreme_sample_receives_the_largest_anomaly_score(self):
        X = [
            [-0.2, 0.1], [0.0, 0.0], [0.1, -0.1], [0.2, 0.2],
            [-0.1, -0.2], [0.15, 0.05], [-0.05, 0.15], [0.05, -0.15],
            [8.0, 8.0],
        ]

        model = IsolationForestScratch(
            n_estimators=120,
            max_samples=9,
            contamination=1.0 / 9.0,
            random_state=7,
        ).fit(X)

        self.assertEqual(max(range(len(X)), key=model.scores_.__getitem__), 8)
        self.assertEqual(model.predict([[0.0, 0.0], [8.0, 8.0]]), [1, -1])

    def test_fit_predict_marks_the_requested_number_of_unique_top_scores(self):
        X = [[float(value), float(value * value)] for value in range(10)]
        model = IsolationForestScratch(
            n_estimators=80,
            max_samples=10,
            contamination=0.2,
            random_state=11,
        )

        labels = model.fit_predict(X)

        self.assertEqual(labels.count(-1), 2)
        self.assertEqual(labels, model.labels_)
        self.assertIsNot(labels, model.labels_)

    def test_equal_scores_at_the_contamination_boundary_remain_normal(self):
        X = [[1.0, 1.0] for _ in range(10)]

        model = IsolationForestScratch(
            n_estimators=10,
            max_samples=10,
            contamination=0.1,
            random_state=13,
        ).fit(X)

        self.assertEqual(model.scores_, [0.5] * 10)
        self.assertEqual(model.labels_, [1] * 10)
        self.assertEqual(model.predict(X), [1] * 10)
        self.assertEqual(model.decision_function(X), [0.0] * 10)

    def test_fixed_seed_reproduces_trees_scores_and_predictions(self):
        X = [[float(value), float(value % 3)] for value in range(12)]
        first = IsolationForestScratch(
            n_estimators=25,
            max_samples=8,
            random_state=19,
        ).fit(X)
        second = IsolationForestScratch(
            n_estimators=25,
            max_samples=8,
            random_state=19,
        ).fit(X)

        self.assertEqual(first.scores_, second.scores_)
        self.assertEqual(first.predict(X), second.predict(X))
        self.assertEqual(first.threshold_, second.threshold_)

    def test_sample_limit_sets_the_default_tree_depth(self):
        model = IsolationForestScratch(
            n_estimators=3,
            max_samples=5,
            random_state=3,
        ).fit([[float(value)] for value in range(12)])

        self.assertEqual(model.max_samples_, 5)
        self.assertEqual(model.max_depth_, math.ceil(math.log2(5)))

    def test_predict_checks_fitted_state_and_feature_count(self):
        model = IsolationForestScratch()
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

        model.fit([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

    def test_constructor_rejects_invalid_core_parameters(self):
        with self.assertRaises(ValueError):
            IsolationForestScratch(n_estimators=0)
        with self.assertRaises(ValueError):
            IsolationForestScratch(max_samples=0)
        with self.assertRaises(ValueError):
            IsolationForestScratch(contamination=0.0)
        with self.assertRaises(ValueError):
            IsolationForestScratch(contamination=0.6)

    def test_models_package_exports_both_isolation_forest_versions(self):
        from Models import IsolationForestScratch as ExportedBasic
        from Models import OptimizedIsolationForestScratch as ExportedOptimized

        self.assertIs(ExportedBasic, IsolationForestScratch)
        self.assertEqual(
            ExportedOptimized.__name__,
            "OptimizedIsolationForestScratch",
        )

    def test_minimal_example_runs_and_prints_one_detected_anomaly(self):
        script = PROJECT_ROOT / "examples" / "isolation_forest_minimal.py"

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
