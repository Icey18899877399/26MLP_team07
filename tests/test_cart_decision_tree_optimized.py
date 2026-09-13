import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.cart_decision_tree_optimized import OptimizedCARTClassifierScratch


class OptimizedCARTClassifierScratchTests(unittest.TestCase):
    def test_sorted_scan_finds_the_best_midpoint(self):
        model = OptimizedCARTClassifierScratch().fit(
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 1, 1],
        )

        self.assertEqual(model.root.feature_index, 0)
        self.assertEqual(model.root.threshold, 1.5)
        self.assertEqual(model.predict([[0.5], [2.5]]), [0, 1])

    def test_min_samples_leaf_blocks_an_isolated_leaf(self):
        X = [[0.0], [1.0], [2.0], [100.0]]
        y = [0, 0, 0, 1]
        unrestricted = OptimizedCARTClassifierScratch(min_samples_leaf=1).fit(X, y)
        restricted = OptimizedCARTClassifierScratch(min_samples_leaf=2).fit(X, y)

        self.assertEqual(unrestricted.predict([[100.0]]), [1])
        self.assertEqual(restricted.predict([[100.0]]), [0])

    def test_min_impurity_decrease_blocks_a_weak_split(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0, 0, 1, 1]
        unrestricted = OptimizedCARTClassifierScratch(
            min_impurity_decrease=0.0
        ).fit(X, y)
        restricted = OptimizedCARTClassifierScratch(
            min_impurity_decrease=0.6
        ).fit(X, y)

        self.assertEqual(unrestricted.n_leaves_, 2)
        self.assertEqual(restricted.n_leaves_, 1)

    def test_balanced_class_weight_equalizes_root_probabilities(self):
        model = OptimizedCARTClassifierScratch(
            max_depth=0,
            class_weight="balanced",
        ).fit(
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 0, 1],
        )

        probabilities = model.predict_proba([[1.5]])[0]

        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertAlmostEqual(probabilities[1], 0.5)

    def test_custom_class_weight_changes_leaf_prediction(self):
        plain = OptimizedCARTClassifierScratch(max_depth=0).fit(
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 0, 1],
        )
        weighted = OptimizedCARTClassifierScratch(
            max_depth=0,
            class_weight={0: 1.0, 1: 5.0},
        ).fit(
            [[0.0], [1.0], [2.0], [3.0]],
            [0, 0, 0, 1],
        )

        self.assertEqual(plain.predict([[1.5]]), [0])
        self.assertEqual(weighted.predict([[1.5]]), [1])

    def test_cost_complexity_pruning_reduces_leaf_count(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [0, 0, 1, 1, 0, 0]
        unpruned = OptimizedCARTClassifierScratch(ccp_alpha=0.0).fit(X, y)
        pruned = OptimizedCARTClassifierScratch(ccp_alpha=1.0).fit(X, y)

        self.assertLess(pruned.n_leaves_, unpruned.n_leaves_)
        self.assertEqual(pruned.n_leaves_, 1)

    def test_feature_sampling_is_reproducible(self):
        X = [
            [0.0, 0.0, 3.0],
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        y = [0, 0, 1, 1]
        first = OptimizedCARTClassifierScratch(
            max_features=1,
            random_state=7,
        ).fit(X, y)
        second = OptimizedCARTClassifierScratch(
            max_features=1,
            random_state=7,
        ).fit(X, y)

        self.assertEqual(first.root.feature_index, second.root.feature_index)
        self.assertEqual(first.predict(X), second.predict(X))

    def test_feature_importances_are_normalized_after_pruning(self):
        model = OptimizedCARTClassifierScratch(ccp_alpha=0.0).fit(
            [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
            [0, 0, 1, 1],
        )

        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertGreater(max(model.feature_importances_), 0.0)


if __name__ == "__main__":
    unittest.main()
