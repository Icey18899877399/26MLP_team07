import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.knn_optimized import OptimizedKNNScratch


class OptimizedKNNScratchTests(unittest.TestCase):
    def test_standardization_uses_training_statistics(self):
        model = OptimizedKNNScratch(n_neighbors=1, standardize=True).fit(
            [[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]],
            [0, 1, 1],
        )

        self.assertEqual(model.means, [2.0, 20.0])
        transformed = model._transform([[2.0, 20.0]])[0]
        self.assertAlmostEqual(transformed[0], 0.0)
        self.assertAlmostEqual(transformed[1], 0.0)

    def test_distance_weighting_can_outvote_a_more_distant_majority(self):
        X = [[0.0], [2.0], [3.0]]
        y = [0, 1, 1]
        uniform = OptimizedKNNScratch(
            n_neighbors=3,
            weights="uniform",
            standardize=False,
        ).fit(X, y)
        weighted = OptimizedKNNScratch(
            n_neighbors=3,
            weights="distance",
            standardize=False,
        ).fit(X, y)

        self.assertEqual(uniform.predict([[0.1]]), [1])
        self.assertEqual(weighted.predict([[0.1]]), [0])

    def test_minkowski_parameter_changes_the_selected_neighbor(self):
        X = [[0.0, 3.0], [2.0, 2.0]]
        y = ["L1 neighbor", "L2 neighbor"]
        manhattan = OptimizedKNNScratch(
            n_neighbors=1,
            p=1,
            standardize=False,
        ).fit(X, y)
        euclidean = OptimizedKNNScratch(
            n_neighbors=1,
            p=2,
            standardize=False,
        ).fit(X, y)

        self.assertEqual(manhattan.predict([[0.0, 0.0]]), ["L1 neighbor"])
        self.assertEqual(euclidean.predict([[0.0, 0.0]]), ["L2 neighbor"])

    def test_exact_matches_receive_all_distance_weight(self):
        model = OptimizedKNNScratch(
            n_neighbors=3,
            weights="distance",
            standardize=False,
        ).fit([[0.0], [0.0], [2.0]], [0, 1, 1])

        probabilities = model.predict_proba([[0.0]])[0]

        self.assertEqual(probabilities, [0.5, 0.5])
        self.assertEqual(model.predict([[0.0]]), [0])

    def test_exposes_optimization_parameters_for_tuning(self):
        model = OptimizedKNNScratch(
            n_neighbors=9,
            p=1,
            weights="uniform",
            standardize=False,
        )

        self.assertEqual(model.n_neighbors, 9)
        self.assertEqual(model.p, 1)
        self.assertEqual(model.weights, "uniform")
        self.assertFalse(model.standardize)


if __name__ == "__main__":
    unittest.main()
