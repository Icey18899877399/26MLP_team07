import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.cart_decision_tree import CARTClassifierScratch


class CARTClassifierScratchTests(unittest.TestCase):
    def test_finds_the_midpoint_split_for_a_binary_problem(self):
        model = CARTClassifierScratch().fit(
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 1, 1],
        )

        self.assertEqual(model.predict([[0.5], [2.5]]), [0, 1])
        self.assertEqual(model.root.feature_index, 0)
        self.assertEqual(model.root.threshold, 1.5)

    def test_supports_multiclass_classification(self):
        X = [[0.0], [0.5], [3.0], [3.5], [6.0], [6.5]]
        y = ["left", "left", "middle", "middle", "right", "right"]

        model = CARTClassifierScratch().fit(X, y)

        self.assertEqual(
            model.predict([[0.2], [3.2], [6.2]]),
            ["left", "middle", "right"],
        )

    def test_predict_proba_uses_leaf_class_frequencies(self):
        model = CARTClassifierScratch(max_depth=0).fit(
            [[0.0], [1.0], [2.0]],
            ["A", "A", "B"],
        )

        probabilities = model.predict_proba([[100.0]])[0]

        self.assertEqual(model.classes, ["A", "B"])
        self.assertAlmostEqual(probabilities[0], 2 / 3)
        self.assertAlmostEqual(probabilities[1], 1 / 3)

    def test_depth_limit_controls_tree_size(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [0, 0, 1, 1, 0, 0]
        stump = CARTClassifierScratch(max_depth=1).fit(X, y)
        full_tree = CARTClassifierScratch(max_depth=None).fit(X, y)

        self.assertEqual(stump.tree_depth_, 1)
        self.assertEqual(stump.n_leaves_, 2)
        self.assertGreater(full_tree.n_leaves_, stump.n_leaves_)

    def test_feature_importance_identifies_the_informative_feature(self):
        model = CARTClassifierScratch().fit(
            [[0.0, 4.0], [1.0, 4.0], [2.0, 4.0], [3.0, 4.0]],
            [0, 0, 1, 1],
        )

        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertEqual(model.feature_importances_, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
