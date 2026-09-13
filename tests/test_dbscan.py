import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.dbscan import DBSCANScratch


class DBSCANScratchTests(unittest.TestCase):
    def test_finds_two_clusters_and_one_noise_sample(self):
        X = [
            [0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1],
            [5.0, 5.0], [5.0, 5.1], [5.1, 5.0], [5.1, 5.1],
            [10.0, 10.0],
        ]

        model = DBSCANScratch(eps=0.25, min_samples=3).fit(X)

        self.assertEqual(model.labels_, [0, 0, 0, 0, 1, 1, 1, 1, -1])
        self.assertEqual(model.n_clusters_, 2)
        self.assertEqual(model.core_sample_indices_, list(range(8)))
        self.assertEqual(model.components_, X[:8])

    def test_min_samples_counts_the_sample_itself(self):
        labels = DBSCANScratch(eps=0.1, min_samples=1).fit_predict([[0.0], [2.0]])

        self.assertEqual(labels, [0, 1])

    def test_an_early_noise_sample_can_become_a_border_sample(self):
        X = [[0.0], [0.15], [0.3]]

        model = DBSCANScratch(eps=0.16, min_samples=3).fit(X)

        self.assertEqual(model.labels_, [0, 0, 0])
        self.assertEqual(model.core_sample_indices_, [1])

    def test_fit_predict_returns_a_copy_of_the_learned_labels(self):
        model = DBSCANScratch(eps=0.25, min_samples=2)

        labels = model.fit_predict([[0.0], [0.1], [5.0], [5.1]])

        self.assertEqual(labels, model.labels_)
        self.assertIsNot(labels, model.labels_)

    def test_predict_assigns_to_the_nearest_reachable_core_sample(self):
        X = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]]
        model = DBSCANScratch(eps=0.25, min_samples=2).fit(X)

        predicted = model.predict([[0.05], [5.05], [10.0]])

        self.assertEqual(predicted, [0, 1, -1])

    def test_predict_breaks_equal_distance_ties_by_training_order(self):
        X = [[-0.1], [1.1], [0.0], [1.2]]
        model = DBSCANScratch(eps=0.7, min_samples=2).fit(X)

        self.assertEqual(model.predict([[0.55]]), [1])

    def test_all_noise_is_a_valid_fitted_model(self):
        model = DBSCANScratch(eps=0.1, min_samples=2).fit([[0.0], [1.0]])

        self.assertEqual(model.labels_, [-1, -1])
        self.assertEqual(model.core_sample_indices_, [])
        self.assertEqual(model.components_, [])
        self.assertEqual(model.n_clusters_, 0)
        self.assertEqual(model.predict([[0.0]]), [-1])

    def test_constructor_exposes_tuning_parameters(self):
        model = DBSCANScratch(eps=0.75, min_samples=7)

        self.assertEqual(model.eps, 0.75)
        self.assertEqual(model.min_samples, 7)

    def test_rejects_invalid_parameters_and_input_shapes(self):
        with self.assertRaises(ValueError):
            DBSCANScratch(eps=0.0)
        with self.assertRaises(ValueError):
            DBSCANScratch(eps=float("nan"))
        with self.assertRaises(ValueError):
            DBSCANScratch(min_samples=0)
        with self.assertRaises(ValueError):
            DBSCANScratch().fit([])
        with self.assertRaises(ValueError):
            DBSCANScratch().fit([[0.0], [1.0, 2.0]])

    def test_predict_checks_fitted_state_and_feature_count(self):
        model = DBSCANScratch()
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

        model.fit([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

    def test_models_package_exports_both_dbscan_versions(self):
        from Models import DBSCANScratch as ExportedBasic
        from Models import OptimizedDBSCANScratch as ExportedOptimized

        self.assertIs(ExportedBasic, DBSCANScratch)
        self.assertEqual(ExportedOptimized.__name__, "OptimizedDBSCANScratch")

    def test_minimal_example_runs_and_prints_labels(self):
        script = PROJECT_ROOT / "examples" / "dbscan_minimal.py"

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[0, 0, 0, 1, 1, 1, -1]")


if __name__ == "__main__":
    unittest.main()
