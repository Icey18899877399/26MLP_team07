import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.gaussian_naive_bayes import GaussianNaiveBayesScratch


class GaussianNaiveBayesScratchTests(unittest.TestCase):
    def test_learns_a_small_multiclass_example(self):
        X = [[0.0], [0.4], [4.8], [5.2], [9.8], [10.2]]
        y = ["left", "left", "middle", "middle", "right", "right"]

        model = GaussianNaiveBayesScratch().fit(X, y)

        self.assertEqual(
            model.predict([[0.2], [5.0], [10.0]]),
            ["left", "middle", "right"],
        )

    def test_learns_priors_means_and_population_variances(self):
        model = GaussianNaiveBayesScratch().fit(
            [[0.0], [2.0], [8.0], [12.0]],
            [0, 0, 1, 1],
        )

        self.assertEqual(model.classes, [0, 1])
        self.assertEqual(model.class_priors, [0.5, 0.5])
        self.assertEqual(model.means, [[1.0], [10.0]])
        self.assertEqual(model.variances, [[1.0], [4.0]])

    def test_predict_proba_follows_class_order_and_sums_to_one(self):
        model = GaussianNaiveBayesScratch().fit(
            [[-2.0], [-1.0], [1.0], [2.0]],
            [0, 0, 1, 1],
        )

        probabilities = model.predict_proba([[0.0]])[0]

        self.assertEqual(len(probabilities), 2)
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)

    def test_variance_floor_is_available_for_tuning(self):
        model = GaussianNaiveBayesScratch(variance_floor=1e-6)

        self.assertEqual(model.variance_floor, 1e-6)


if __name__ == "__main__":
    unittest.main()
