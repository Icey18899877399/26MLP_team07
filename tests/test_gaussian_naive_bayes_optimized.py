import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.gaussian_naive_bayes_optimized import OptimizedGaussianNaiveBayesScratch


class OptimizedGaussianNaiveBayesScratchTests(unittest.TestCase):
    def test_log_space_keeps_high_dimensional_probabilities_finite(self):
        model = OptimizedGaussianNaiveBayesScratch().fit(
            [[0.0] * 800, [0.1] * 800, [5.0] * 800, [5.1] * 800],
            [0, 0, 1, 1],
        )

        probabilities = model.predict_proba([[0.05] * 800])[0]

        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertEqual(model.predict([[0.05] * 800]), [0])

    def test_variance_smoothing_handles_constant_features(self):
        model = OptimizedGaussianNaiveBayesScratch(var_smoothing=1e-6).fit(
            [[1.0, 0.0], [1.0, 0.2], [1.0, 3.0], [1.0, 3.2]],
            [0, 0, 1, 1],
        )

        probabilities = model.predict_proba([[1.0, 0.1]])[0]

        self.assertGreater(model.epsilon, 0.0)
        self.assertTrue(all(math.isfinite(value) for value in probabilities))

    def test_custom_class_prior_changes_an_ambiguous_prediction(self):
        X = [[-1.0], [1.0], [-1.0], [1.0]]
        y = [0, 0, 1, 1]
        model = OptimizedGaussianNaiveBayesScratch(
            class_prior={0: 0.1, 1: 0.9}
        ).fit(X, y)

        self.assertEqual(model.predict([[0.0]]), [1])
        self.assertEqual(model.class_priors, [0.1, 0.9])

    def test_partial_fit_matches_one_shot_fit(self):
        X = [[0.0, 1.0], [0.4, 1.2], [4.8, 3.0], [5.2, 3.2]]
        y = [0, 0, 1, 1]
        one_shot = OptimizedGaussianNaiveBayesScratch().fit(X, y)
        incremental = OptimizedGaussianNaiveBayesScratch()

        incremental.partial_fit(X[:2], y[:2], classes=[0, 1])
        incremental.partial_fit(X[2:], y[2:])

        one_shot_probabilities = one_shot.predict_proba([[0.2, 1.1], [5.0, 3.1]])
        incremental_probabilities = incremental.predict_proba(
            [[0.2, 1.1], [5.0, 3.1]]
        )
        for expected_row, actual_row in zip(
            one_shot_probabilities,
            incremental_probabilities,
        ):
            for expected, actual in zip(expected_row, actual_row):
                self.assertAlmostEqual(actual, expected, places=12)

    def test_exposes_optimization_parameters_for_tuning(self):
        model = OptimizedGaussianNaiveBayesScratch(
            var_smoothing=1e-7,
            class_prior={0: 0.25, 1: 0.75},
        )

        self.assertEqual(model.var_smoothing, 1e-7)
        self.assertEqual(model.class_prior, {0: 0.25, 1: 0.75})


if __name__ == "__main__":
    unittest.main()
