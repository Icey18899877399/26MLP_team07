import subprocess
import sys
import unittest
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.dbscan_optimized import OptimizedDBSCANScratch, _KDTree
from Models.dbscan import DBSCANScratch


class OptimizedDBSCANScratchTests(unittest.TestCase):
    def test_kd_tree_radius_query_matches_brute_force(self):
        X = [[0.0, 0.0], [0.1, 0.2], [0.3, 0.1], [1.0, 1.0], [2.0, 2.0]]
        tree = _KDTree(X, leaf_size=2)
        query = [0.15, 0.1]
        radius_squared = 0.25 ** 2
        expected = [
            index
            for index, row in enumerate(X)
            if sum((left - right) ** 2 for left, right in zip(row, query))
            <= radius_squared
        ]

        self.assertEqual(tree.query_radius(query, radius_squared), expected)

    def test_kd_tree_matches_brute_force_across_random_queries(self):
        generator = random.Random(19)
        for feature_count in (1, 2, 4):
            X = [
                [generator.uniform(-3.0, 3.0) for _ in range(feature_count)]
                for _ in range(35)
            ]
            for leaf_size in (1, 3, 8):
                tree = _KDTree(X, leaf_size=leaf_size)
                for radius in (0.1, 0.8, 2.5):
                    query = [
                        generator.uniform(-3.0, 3.0)
                        for _ in range(feature_count)
                    ]
                    radius_squared = radius ** 2
                    expected = [
                        index
                        for index, row in enumerate(X)
                        if sum(
                            (left - right) ** 2
                            for left, right in zip(row, query)
                        )
                        <= radius_squared
                    ]
                    self.assertEqual(
                        tree.query_radius(query, radius_squared),
                        expected,
                    )

    def test_kd_tree_and_brute_force_produce_the_same_clustering(self):
        X = [
            [0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1],
            [5.0, 5.0], [5.0, 5.1], [5.1, 5.0], [5.1, 5.1],
            [10.0, 10.0],
        ]

        tree_model = OptimizedDBSCANScratch(
            eps=0.25, min_samples=3, algorithm="kd_tree", standardize=False,
        ).fit(X)
        brute_model = OptimizedDBSCANScratch(
            eps=0.25, min_samples=3, algorithm="brute", standardize=False,
        ).fit(X)

        self.assertEqual(tree_model.labels_, brute_model.labels_)
        self.assertEqual(tree_model.core_sample_indices_, brute_model.core_sample_indices_)
        self.assertEqual(tree_model.labels_, [0, 0, 0, 0, 1, 1, 1, 1, -1])

    def test_both_optimized_searches_match_the_basic_algorithm(self):
        generator = random.Random(31)
        X = [
            [generator.uniform(-2.0, 2.0), generator.uniform(-2.0, 2.0)]
            for _ in range(40)
        ]

        for eps, min_samples in ((0.35, 2), (0.7, 4), (1.2, 7)):
            expected = DBSCANScratch(eps=eps, min_samples=min_samples).fit(X)
            for algorithm in ("kd_tree", "brute"):
                actual = OptimizedDBSCANScratch(
                    eps=eps,
                    min_samples=min_samples,
                    algorithm=algorithm,
                    standardize=False,
                ).fit(X)
                self.assertEqual(actual.labels_, expected.labels_)
                self.assertEqual(
                    actual.core_sample_indices_,
                    expected.core_sample_indices_,
                )

    def test_standardization_uses_training_statistics_and_keeps_raw_components(self):
        X = [[0.0, 0.0], [2.0, 20.0], [100.0, 1000.0], [102.0, 1020.0]]

        model = OptimizedDBSCANScratch(
            eps=0.2,
            min_samples=2,
            standardize=True,
        ).fit(X)

        self.assertEqual(model.means_, [51.0, 510.0])
        self.assertEqual(model.components_, X)
        self.assertEqual(model.n_clusters_, 2)

    def test_zero_variance_feature_uses_unit_scale(self):
        model = OptimizedDBSCANScratch(
            eps=2.0,
            min_samples=1,
            standardize=True,
        ).fit([[1.0, 4.0], [1.0, 6.0]])

        self.assertEqual(model.scales_[0], 1.0)

    def test_predict_uses_training_statistics_without_changing_them(self):
        X = [[0.0, 0.0], [0.0, 1.0], [10.0, 100.0], [10.0, 101.0]]
        model = OptimizedDBSCANScratch(
            eps=0.1,
            min_samples=2,
            standardize=True,
        ).fit(X)
        means_before = list(model.means_)
        scales_before = list(model.scales_)

        predicted = model.predict([[0.0, 0.5], [10.0, 100.5], [5.0, 50.0]])

        self.assertEqual(predicted, [0, 1, -1])
        self.assertEqual(model.means_, means_before)
        self.assertEqual(model.scales_, scales_before)

    def test_all_noise_is_a_valid_fitted_model(self):
        model = OptimizedDBSCANScratch(
            eps=0.1,
            min_samples=2,
            standardize=False,
        ).fit([[0.0], [1.0]])

        self.assertEqual(model.labels_, [-1, -1])
        self.assertEqual(model.core_sample_indices_, [])
        self.assertEqual(model.components_, [])
        self.assertEqual(model.n_clusters_, 0)
        self.assertEqual(model.predict([[0.0]]), [-1])

    def test_predict_tie_rule_matches_the_basic_version(self):
        X = [[-0.1], [1.1], [0.0], [1.2]]
        basic = DBSCANScratch(eps=0.7, min_samples=2).fit(X)
        optimized = OptimizedDBSCANScratch(
            eps=0.7,
            min_samples=2,
            standardize=False,
        ).fit(X)

        self.assertEqual(optimized.predict([[0.55]]), basic.predict([[0.55]]))

    def test_fit_predict_returns_the_learned_labels(self):
        X = [[0.0], [0.1], [5.0], [5.1]]
        model = OptimizedDBSCANScratch(
            eps=0.25, min_samples=2, standardize=False,
        )

        labels = model.fit_predict(X)

        self.assertEqual(labels, model.labels_)

    def test_constructor_exposes_optimization_parameters(self):
        model = OptimizedDBSCANScratch(
            eps=0.8,
            min_samples=6,
            algorithm="brute",
            leaf_size=12,
            standardize=False,
        )

        self.assertEqual(model.eps, 0.8)
        self.assertEqual(model.min_samples, 6)
        self.assertEqual(model.algorithm, "brute")
        self.assertEqual(model.leaf_size, 12)
        self.assertFalse(model.standardize)

    def test_rejects_invalid_parameters_and_input_shapes(self):
        with self.assertRaises(ValueError):
            OptimizedDBSCANScratch(algorithm="typo")
        with self.assertRaises(ValueError):
            OptimizedDBSCANScratch(leaf_size=0)
        with self.assertRaises(ValueError):
            OptimizedDBSCANScratch(eps=-1.0)
        with self.assertRaises(ValueError):
            OptimizedDBSCANScratch(min_samples=0)
        with self.assertRaises(ValueError):
            OptimizedDBSCANScratch().fit([[0.0], [1.0, 2.0]])

    def test_predict_checks_fitted_state_and_feature_count(self):
        model = OptimizedDBSCANScratch()
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

        model.fit([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

    def test_optimized_example_runs_and_prints_labels(self):
        script = PROJECT_ROOT / "examples" / "dbscan_optimized_minimal.py"

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
