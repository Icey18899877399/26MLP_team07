import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.kmeans import KMeansScratch


class KMeansScratchTests(unittest.TestCase):
    def test_finds_the_means_of_two_separated_clusters(self):
        X = [[0.0], [2.0], [10.0], [12.0]]

        model = KMeansScratch(n_clusters=2, max_iter=50, random_state=4).fit(X)

        self.assertEqual(sorted(center[0] for center in model.cluster_centers_), [1.0, 11.0])
        self.assertAlmostEqual(model.inertia_, 4.0)

    def test_predict_reuses_the_fitted_centers(self):
        X = [[0.0], [1.0], [9.0], [10.0]]
        model = KMeansScratch(n_clusters=2, random_state=2).fit(X)

        predicted = model.predict(X)

        self.assertEqual(predicted, model.labels_)
        self.assertEqual(model.predict([[0.2]])[0], model.labels_[0])
        self.assertEqual(model.predict([[9.8]])[0], model.labels_[-1])

    def test_fit_predict_returns_the_learned_labels(self):
        X = [[0.0], [1.0], [9.0], [10.0]]
        model = KMeansScratch(n_clusters=2, random_state=3)

        labels = model.fit_predict(X)

        self.assertEqual(labels, model.labels_)

    def test_fixed_seed_reproduces_centers_labels_and_history(self):
        X = [[0.0, 0.0], [0.5, 0.2], [5.0, 5.0], [5.2, 4.8]]

        first = KMeansScratch(n_clusters=2, random_state=9).fit(X)
        second = KMeansScratch(n_clusters=2, random_state=9).fit(X)

        self.assertEqual(first.cluster_centers_, second.cluster_centers_)
        self.assertEqual(first.labels_, second.labels_)
        self.assertEqual(first.inertia_history_, second.inertia_history_)

    def test_constructor_exposes_basic_tuning_parameters(self):
        model = KMeansScratch(n_clusters=4, max_iter=80, random_state=12)

        self.assertEqual(model.n_clusters, 4)
        self.assertEqual(model.max_iter, 80)
        self.assertEqual(model.random_state, 12)

    def test_rejects_invalid_training_matrix_and_cluster_count(self):
        with self.assertRaises(ValueError):
            KMeansScratch(n_clusters=2).fit([])
        with self.assertRaises(ValueError):
            KMeansScratch(n_clusters=2).fit([[0.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            KMeansScratch(n_clusters=3).fit([[0.0], [1.0]])

    def test_predict_checks_fitted_state_and_feature_count(self):
        model = KMeansScratch(n_clusters=2)
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

        model.fit([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

    def test_models_package_exports_both_kmeans_versions(self):
        from Models import KMeansScratch as ExportedBasic
        from Models import OptimizedKMeansScratch as ExportedOptimized

        self.assertIs(ExportedBasic, KMeansScratch)
        self.assertEqual(ExportedOptimized.__name__, "OptimizedKMeansScratch")

    def test_minimal_example_runs_and_prints_labels(self):
        script = PROJECT_ROOT / "examples" / "kmeans_minimal.py"

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[0, 0, 1, 1]")


if __name__ == "__main__":
    unittest.main()
