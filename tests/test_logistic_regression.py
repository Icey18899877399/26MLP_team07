import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.logistic_regression import LogisticRegressionScratch


class LogisticRegressionScratchTests(unittest.TestCase):
    def test_learns_the_smallest_binary_example(self):
        X = [[-2.0], [-1.0], [1.0], [2.0]]
        y = [0, 0, 1, 1]

        model = LogisticRegressionScratch(
            learning_rate=0.2,
            max_iter=1_000,
            threshold=0.5,
        ).fit(X, y)

        self.assertEqual(model.predict(X), y)

    def test_constructor_exposes_basic_tuning_parameters(self):
        model = LogisticRegressionScratch(
            learning_rate=0.05,
            max_iter=250,
            threshold=0.7,
        )

        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.max_iter, 250)
        self.assertEqual(model.threshold, 0.7)

    def test_predict_proba_returns_two_class_probabilities(self):
        model = LogisticRegressionScratch(max_iter=500).fit(
            [[-2.0], [-1.0], [1.0], [2.0]],
            [0, 0, 1, 1],
        )

        probabilities = model.predict_proba([[0.0]])[0]

        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
