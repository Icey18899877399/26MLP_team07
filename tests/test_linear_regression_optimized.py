import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.linear_regression_optimized import OptimizedLinearRegressionScratch


class OptimizedLinearRegressionScratchTests(unittest.TestCase):
    def test_default_settings_converge_when_target_scale_is_large(self):
        X = [[float(index)] for index in range(-50, 50)]
        y = [1_000.0 * row[0] + 100.0 for row in X]

        model = OptimizedLinearRegressionScratch().fit(X, y)
        predictions = model.predict(X)
        mse = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, y)
        ) / len(y)

        self.assertLess(mse, 1.0)

    def test_standardization_handles_features_with_different_scales(self):
        X = [
            [1_000_000.0, -2.0],
            [2_000_000.0, -1.0],
            [3_000_000.0, 0.0],
            [4_000_000.0, 1.0],
            [5_000_000.0, 2.0],
            [6_000_000.0, -2.0],
        ]
        y = [0.000003 * row[0] - 2.0 * row[1] + 1.0 for row in X]

        model = OptimizedLinearRegressionScratch(
            learning_rate=0.05,
            max_iter=3_000,
            batch_size=None,
            tol=1e-12,
        ).fit(X, y)

        predictions = model.predict(X)
        self.assertTrue(all(math.isfinite(value) for value in predictions))
        for prediction, target in zip(predictions, y):
            self.assertAlmostEqual(prediction, target, places=3)

    def test_l2_regularization_reduces_weight_magnitude(self):
        X = [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
        y = [-9.0, -6.0, -3.0, 3.0, 6.0, 9.0]

        plain = OptimizedLinearRegressionScratch(
            learning_rate=0.03,
            max_iter=2_000,
            batch_size=None,
            standardize=False,
            l2=0.0,
            tol=0.0,
        ).fit(X, y)
        regularized = OptimizedLinearRegressionScratch(
            learning_rate=0.03,
            max_iter=2_000,
            batch_size=None,
            standardize=False,
            l2=1.0,
            tol=0.0,
        ).fit(X, y)

        self.assertLess(abs(regularized.weights[0]), abs(plain.weights[0]))

    def test_early_stopping_records_the_actual_iteration_count(self):
        X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
        y = [-3.0, -1.0, 1.0, 3.0, 5.0]

        model = OptimizedLinearRegressionScratch(
            max_iter=5_000,
            batch_size=None,
            tol=1e-7,
        ).fit(X, y)

        self.assertLess(model.n_iter, model.max_iter)
        self.assertEqual(model.n_iter, len(model.loss_history))

    def test_seed_reproduces_mini_batch_training(self):
        X = [[float(index), float(index % 3)] for index in range(30)]
        y = [1.5 * row[0] - 0.7 * row[1] + 2.0 for row in X]
        parameters = {
            "learning_rate": 0.02,
            "max_iter": 300,
            "batch_size": 7,
            "tol": 0.0,
            "random_state": 19,
        }

        first = OptimizedLinearRegressionScratch(**parameters).fit(X, y)
        second = OptimizedLinearRegressionScratch(**parameters).fit(X, y)

        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.bias, second.bias)
        self.assertEqual(first.predict(X), second.predict(X))


if __name__ == "__main__":
    unittest.main()
