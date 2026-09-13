import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.kmeans_optimized import OptimizedKMeansScratch


class OptimizedKMeansScratchTests(unittest.TestCase):
    def test_standardization_uses_training_statistics_and_reports_original_centers(self):
        X = [[0.0, 0.0], [2.0, 20.0], [100.0, 1000.0], [102.0, 1020.0]]

        model = OptimizedKMeansScratch(
            n_clusters=2,
            n_init=3,
            standardize=True,
            random_state=5,
        ).fit(X)

        self.assertEqual(model.means_, [51.0, 510.0])
        centers = sorted(model.cluster_centers_)
        for actual, expected in zip(centers, [[1.0, 10.0], [101.0, 1010.0]]):
            for value, target in zip(actual, expected):
                self.assertAlmostEqual(value, target)

    def test_multiple_initializations_keep_the_lowest_inertia(self):
        X = [[0.0], [1.0], [4.0], [5.0], [10.0], [11.0]]

        model = OptimizedKMeansScratch(
            n_clusters=3,
            init="random",
            n_init=6,
            standardize=False,
            random_state=8,
        ).fit(X)

        self.assertEqual(len(model.run_inertias_), 6)
        self.assertEqual(model.inertia_, min(model.run_inertias_))

    def test_kmeans_plus_plus_is_reproducible_with_a_fixed_seed(self):
        X = [[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.2, 5.1], [10.0, 0.0], [10.2, 0.1]]

        first = OptimizedKMeansScratch(n_clusters=3, n_init=4, random_state=17).fit(X)
        second = OptimizedKMeansScratch(n_clusters=3, n_init=4, random_state=17).fit(X)

        self.assertEqual(first.cluster_centers_, second.cluster_centers_)
        self.assertEqual(first.labels_, second.labels_)
        self.assertEqual(first.run_inertias_, second.run_inertias_)

    def test_empty_cluster_is_reset_to_a_sample_away_from_the_occupied_mean(self):
        model = OptimizedKMeansScratch(
            n_clusters=2,
            n_init=1,
            standardize=False,
        )

        centers = model._recompute_centers(
            [[0.0], [5.0], [10.0]],
            labels=[0, 0, 0],
        )

        self.assertEqual(centers[0], [5.0])
        self.assertIn(centers[1], ([0.0], [10.0]))

    def test_tolerance_stops_after_centers_stabilize(self):
        X = [[0.0], [1.0], [9.0], [10.0]]

        model = OptimizedKMeansScratch(
            n_clusters=2,
            n_init=1,
            max_iter=100,
            tol=1e-8,
            standardize=False,
            random_state=2,
        ).fit(X)

        self.assertLess(model.n_iter_, model.max_iter)
        self.assertEqual(len(model.inertia_history_), model.n_iter_)

    def test_iteration_limit_keeps_centers_labels_predictions_and_inertia_consistent(self):
        X = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2]]
        model = OptimizedKMeansScratch(
            n_clusters=2,
            n_init=1,
            max_iter=1,
            tol=0.0,
            standardize=False,
            random_state=10,
        ).fit(X)

        public_labels = [
            min(
                range(2),
                key=lambda cluster: sum(
                    (value - center) ** 2
                    for value, center in zip(row, model.cluster_centers_[cluster])
                ),
            )
            for row in X
        ]
        public_inertia = sum(
            sum(
                (value - center) ** 2
                for value, center in zip(row, model.cluster_centers_[label])
            )
            for row, label in zip(X, public_labels)
        )

        self.assertEqual(model.labels_, public_labels)
        self.assertEqual(model.predict(X), public_labels)
        self.assertAlmostEqual(model.inertia_, public_inertia)
        self.assertAlmostEqual(model.inertia_history_[-1], model.inertia_)

    def test_kmeans_plus_plus_never_reselects_a_zero_weight_center(self):
        class ZeroThresholdGenerator:
            @staticmethod
            def randrange(_limit):
                return 0

            @staticmethod
            def random():
                return 0.0

        model = OptimizedKMeansScratch(n_clusters=2)

        centers = model._initialize_kmeans_plus_plus(
            [[0.0], [10.0], [20.0]],
            ZeroThresholdGenerator(),
        )

        self.assertEqual(centers[0], [0.0])
        self.assertNotEqual(centers[1], centers[0])

    def test_duplicate_heavy_data_recovers_empty_clusters_without_state_mismatch(self):
        X = [[0.0], [0.0], [0.0], [10.0], [10.0]]

        model = OptimizedKMeansScratch(
            n_clusters=3,
            n_init=2,
            standardize=False,
            random_state=4,
        ).fit(X)

        self.assertEqual(len(model.cluster_centers_), 3)
        self.assertEqual(model.predict(X), model.labels_)
        self.assertAlmostEqual(model.inertia_history_[-1], model.inertia_)

    def test_rejects_invalid_parameters_and_input_shapes(self):
        with self.assertRaises(ValueError):
            OptimizedKMeansScratch(init="typo")
        with self.assertRaises(ValueError):
            OptimizedKMeansScratch(n_init=0)
        with self.assertRaises(ValueError):
            OptimizedKMeansScratch(max_iter=0)
        with self.assertRaises(ValueError):
            OptimizedKMeansScratch(tol=-1.0)
        with self.assertRaises(ValueError):
            OptimizedKMeansScratch(n_clusters=2).fit([[0.0], [1.0, 2.0]])

    def test_predict_checks_fitted_state_and_feature_count_without_changing_statistics(self):
        model = OptimizedKMeansScratch(n_clusters=2)
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

        model.fit([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0], [10.0, 10.0]])
        means_before = list(model.means_)
        scales_before = list(model.scales_)
        with self.assertRaises(ValueError):
            model.predict([[1.0]])

        self.assertEqual(model.means_, means_before)
        self.assertEqual(model.scales_, scales_before)

    def test_fit_predict_matches_predict_on_training_rows(self):
        X = [[0.0], [1.0], [9.0], [10.0]]
        model = OptimizedKMeansScratch(n_clusters=2, n_init=2, random_state=6)

        labels = model.fit_predict(X)

        self.assertEqual(labels, model.labels_)
        self.assertEqual(model.predict(X), model.labels_)

    def test_constructor_exposes_optimization_parameters(self):
        model = OptimizedKMeansScratch(
            n_clusters=5,
            init="random",
            n_init=7,
            max_iter=90,
            tol=1e-5,
            standardize=False,
            random_state=21,
        )

        self.assertEqual(model.n_clusters, 5)
        self.assertEqual(model.init, "random")
        self.assertEqual(model.n_init, 7)
        self.assertEqual(model.max_iter, 90)
        self.assertEqual(model.tol, 1e-5)
        self.assertFalse(model.standardize)
        self.assertEqual(model.random_state, 21)

    def test_optimized_example_runs_and_prints_labels(self):
        script = PROJECT_ROOT / "examples" / "kmeans_optimized_minimal.py"

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
