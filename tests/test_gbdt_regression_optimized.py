import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.gbdt_regression_optimized import OptimizedGBDTRegressorScratch


class OptimizedGBDTRegressorScratchTests(unittest.TestCase):
    def test_large_validation_fraction_still_leaves_training_data(self):
        model = OptimizedGBDTRegressorScratch(
            n_estimators=2,
            validation_fraction=0.9,
            n_iter_no_change=1,
            random_state=2,
        )

        model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])

        self.assertGreaterEqual(model.n_estimators_, 1)

    def test_optimized_model_learns_a_nonlinear_signal(self):
        X = [[float(value), float(value % 3)] for value in range(-15, 16)]
        y = [0.5 * row[0] ** 2 - 2.0 * row[1] for row in X]
        mean = sum(y) / len(y)
        baseline_mse = sum((target - mean) ** 2 for target in y) / len(y)

        model = OptimizedGBDTRegressorScratch(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=2,
            random_state=7,
        ).fit(X, y)
        predictions = model.predict(X)
        mse = sum((prediction - target) ** 2 for prediction, target in zip(predictions, y)) / len(y)

        self.assertLess(mse, baseline_mse * 0.03)

    def test_subsampling_is_reproducible_with_a_fixed_seed(self):
        X = [[float(value), float(value % 4)] for value in range(30)]
        y = [row[0] + row[1] for row in X]
        parameters = {
            "n_estimators": 8,
            "subsample": 0.6,
            "max_features": 0.5,
            "random_state": 19,
        }

        first = OptimizedGBDTRegressorScratch(**parameters).fit(X, y)
        second = OptimizedGBDTRegressorScratch(**parameters).fit(X, y)

        self.assertEqual(first.sample_indices_, second.sample_indices_)
        self.assertEqual(first.predict(X), second.predict(X))
        self.assertTrue(all(len(indices) == 18 for indices in first.sample_indices_))

    def test_subsample_indices_refer_only_to_original_training_rows(self):
        X = [[float(value)] for value in range(12)]
        y = [float(value * value) for value in range(12)]

        model = OptimizedGBDTRegressorScratch(
            n_estimators=2,
            subsample=0.5,
            validation_fraction=0.25,
            n_iter_no_change=5,
            random_state=23,
        ).fit(X, y)

        training = set(model.training_indices_)
        validation = set(model.validation_indices_)
        sampled = {index for stage in model.sample_indices_ for index in stage}
        self.assertEqual(training | validation, set(range(12)))
        self.assertFalse(training & validation)
        self.assertTrue(sampled <= training)

    def test_l2_regularization_reduces_the_first_tree_update(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 0.0, 10.0, 10.0]

        unregularized = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            l2_regularization=0.0,
            random_state=3,
        ).fit(X, y)
        regularized = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            l2_regularization=20.0,
            random_state=3,
        ).fit(X, y)

        unregularized_change = max(abs(value - 5.0) for value in unregularized.predict(X))
        regularized_change = max(abs(value - 5.0) for value in regularized.predict(X))
        self.assertLess(regularized_change, unregularized_change)

    def test_l2_regularization_participates_in_split_selection(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 1.0, 2.0, 4.0]

        unregularized = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            l2_regularization=0.0,
        ).fit(X, y)
        regularized = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            l2_regularization=10.0,
        ).fit(X, y)

        self.assertEqual(unregularized.estimators_[0].root.threshold, 2.5)
        self.assertEqual(regularized.estimators_[0].root.threshold, 1.5)

    def test_validation_early_stopping_keeps_only_the_best_rounds(self):
        X = [[float(value)] for value in range(40)]
        y = [3.0] * 40

        model = OptimizedGBDTRegressorScratch(
            n_estimators=50,
            validation_fraction=0.2,
            n_iter_no_change=3,
            tol=1e-12,
            random_state=11,
        ).fit(X, y)

        self.assertLess(model.n_estimators_, 50)
        self.assertEqual(model.n_estimators_, 1)
        self.assertEqual(model.n_estimators_, len(model.estimators_))
        self.assertEqual(model.n_estimators_, len(model.validation_loss_))
        self.assertEqual(model.predict([[100.0]])[0], 3.0)

    def test_leaf_size_and_minimum_gain_can_block_splits(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 0.0, 0.0, 10.0]

        leaf_limited = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=2,
            min_samples_leaf=2,
        ).fit(X, y)
        gain_limited = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=2,
            min_impurity_decrease=100.0,
        ).fit(X, y)

        root = leaf_limited.estimators_[0].root
        self.assertGreaterEqual(root.left.sample_count, 2)
        self.assertGreaterEqual(root.right.sample_count, 2)
        self.assertTrue(gain_limited.estimators_[0].root.is_leaf)

    def test_feature_subsampling_changes_the_available_split(self):
        X = [[float(value), 1.0] for value in range(8)]
        y = [0.0] * 4 + [10.0] * 4

        all_features = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            max_features=None,
            random_state=0,
        ).fit(X, y)
        one_sampled_feature = OptimizedGBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            max_features=1,
            random_state=0,
        ).fit(X, y)

        self.assertEqual(all_features.predict(X), y)
        self.assertEqual(one_sampled_feature.predict(X), [5.0] * len(y))

    def test_staged_prediction_finishes_at_the_regular_prediction(self):
        X = [[float(value)] for value in range(10)]
        y = [row[0] ** 2 for row in X]
        model = OptimizedGBDTRegressorScratch(
            n_estimators=6,
            random_state=5,
        ).fit(X, y)

        stages = list(model.staged_predict(X))

        self.assertEqual(len(stages), model.n_estimators_)
        self.assertEqual(stages[-1], model.predict(X))

    def test_feature_importances_are_normalized(self):
        X = [[float(value), float(value % 2)] for value in range(30)]
        y = [5.0 * row[0] for row in X]

        model = OptimizedGBDTRegressorScratch(
            n_estimators=20,
            max_depth=2,
            max_features=None,
            random_state=13,
        ).fit(X, y)

        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertGreater(model.feature_importances_[0], model.feature_importances_[1])


if __name__ == "__main__":
    unittest.main()
