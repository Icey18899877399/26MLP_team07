import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.knn import KNNScratch


class KNNScratchTests(unittest.TestCase):
    def test_predicts_the_nearest_majority_class(self):
        X = [[0.0], [1.0], [4.0], [5.0]]
        y = [0, 0, 1, 1]
        model = KNNScratch(n_neighbors=3).fit(X, y)

        self.assertEqual(model.predict([[0.5], [4.5]]), [0, 1])

    def test_supports_multiclass_labels(self):
        X = [[0.0], [5.0], [10.0]]
        y = ["left", "middle", "right"]
        model = KNNScratch(n_neighbors=1).fit(X, y)

        self.assertEqual(model.predict([[0.2], [5.1], [9.8]]), y)

    def test_predict_proba_follows_the_learned_class_order(self):
        X = [[0.0], [1.0], [4.0], [5.0]]
        y = [0, 0, 1, 1]
        model = KNNScratch(n_neighbors=3).fit(X, y)

        probabilities = model.predict_proba([[0.5]])[0]

        self.assertEqual(model.classes, [0, 1])
        self.assertAlmostEqual(probabilities[0], 2 / 3)
        self.assertAlmostEqual(probabilities[1], 1 / 3)

    def test_exposes_neighbor_count_for_tuning(self):
        model = KNNScratch(n_neighbors=7)

        self.assertEqual(model.n_neighbors, 7)


if __name__ == "__main__":
    unittest.main()

