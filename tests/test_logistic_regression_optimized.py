import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.logistic_regression_optimized import OptimizedLogisticRegressionScratch


class OptimizedLogisticRegressionScratchTests(unittest.TestCase):
    def test_standardization_keeps_large_input_probabilities_finite(self):
        X = [[-1_000_000.0], [-500_000.0], [500_000.0], [1_000_000.0]]
        y = [0, 0, 1, 1]
        model = OptimizedLogisticRegressionScratch(max_iter=500).fit(X, y)

        for negative, positive in model.predict_proba(X):
            self.assertTrue(math.isfinite(negative))
            self.assertTrue(math.isfinite(positive))
            self.assertAlmostEqual(negative + positive, 1.0, places=12)

    def test_l2_regularization_reduces_weight_magnitude(self):
        X = [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
        y = [0, 0, 0, 1, 1, 1]
        plain = OptimizedLogisticRegressionScratch(max_iter=1_000, l2=0.0, tol=0.0).fit(X, y)
        regularized = OptimizedLogisticRegressionScratch(max_iter=1_000, l2=1.0, tol=0.0).fit(X, y)

        self.assertLess(abs(regularized.weights[0]), abs(plain.weights[0]))

    def test_balanced_weights_recover_the_rare_overlapping_class(self):
        X = [[0.0]] * 8 + [[1.0], [1.0], [1.0]]
        y = [0] * 10 + [1]
        plain = OptimizedLogisticRegressionScratch(max_iter=3_000, tol=1e-12).fit(X, y)
        balanced = OptimizedLogisticRegressionScratch(
            max_iter=3_000,
            tol=1e-12,
            class_weight="balanced",
        ).fit(X, y)

        self.assertEqual(plain.predict([[1.0]]), [0])
        self.assertEqual(balanced.predict([[1.0]]), [1])

    def test_early_stopping_records_less_than_the_iteration_limit(self):
        model = OptimizedLogisticRegressionScratch(max_iter=5_000, tol=1e-4).fit(
            [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
            [0, 0, 0, 1, 1, 1],
        )

        self.assertLess(model.n_iter, model.max_iter)
        self.assertEqual(model.n_iter, len(model.loss_history))


if __name__ == "__main__":
    unittest.main()
