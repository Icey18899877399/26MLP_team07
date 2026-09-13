import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.linear_regression import LinearRegressionScratch
from Models.linear_regression_optimized import OptimizedLinearRegressionScratch


class LinearRegressionScratchTests(unittest.TestCase):
    def test_learns_a_line_with_slope_and_intercept(self):
        X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
        y = [-3.0, -1.0, 1.0, 3.0, 5.0]

        model = LinearRegressionScratch(
            learning_rate=0.1,
            max_iter=1_000,
        ).fit(X, y)

        self.assertAlmostEqual(model.weights[0], 2.0, places=5)
        self.assertAlmostEqual(model.bias, 1.0, places=5)
        self.assertAlmostEqual(model.predict([[3.0]])[0], 7.0, places=5)

    def test_learns_multiple_feature_coefficients(self):
        X = [
            [-2.0, 1.0],
            [-1.0, -2.0],
            [0.0, 2.0],
            [1.0, -1.0],
            [2.0, 0.5],
            [3.0, 3.0],
        ]
        y = [3.0 * row[0] - 2.0 * row[1] + 0.5 for row in X]

        model = LinearRegressionScratch(
            learning_rate=0.05,
            max_iter=2_000,
        ).fit(X, y)

        self.assertAlmostEqual(model.weights[0], 3.0, places=4)
        self.assertAlmostEqual(model.weights[1], -2.0, places=4)
        self.assertAlmostEqual(model.bias, 0.5, places=4)

    def test_constructor_exposes_basic_tuning_parameters(self):
        model = LinearRegressionScratch(learning_rate=0.03, max_iter=250)

        self.assertEqual(model.learning_rate, 0.03)
        self.assertEqual(model.max_iter, 250)

    def test_models_package_exports_both_linear_regressors(self):
        import Models

        self.assertIs(Models.LinearRegressionScratch, LinearRegressionScratch)
        self.assertIs(
            Models.OptimizedLinearRegressionScratch,
            OptimizedLinearRegressionScratch,
        )


if __name__ == "__main__":
    unittest.main()
