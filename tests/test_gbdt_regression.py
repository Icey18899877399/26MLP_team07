import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.gbdt_regression import GBDTRegressorScratch


class GBDTRegressorScratchTests(unittest.TestCase):
    def test_one_stump_fits_a_two_level_residual_pattern(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 0.0, 10.0, 10.0]

        model = GBDTRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
        ).fit(X, y)

        self.assertEqual(model.predict(X), y)
        self.assertEqual(len(model.estimators_), 1)
        self.assertAlmostEqual(model.init_prediction_, 5.0)

    def test_additional_boosting_rounds_reduce_training_error(self):
        X = [[float(value)] for value in range(-4, 5)]
        y = [row[0] ** 2 for row in X]

        one_tree = GBDTRegressorScratch(
            n_estimators=1,
            learning_rate=0.2,
            max_depth=2,
        ).fit(X, y)
        twelve_trees = GBDTRegressorScratch(
            n_estimators=12,
            learning_rate=0.2,
            max_depth=2,
        ).fit(X, y)

        self.assertLess(twelve_trees.train_loss_[-1], one_tree.train_loss_[-1])
        self.assertLess(twelve_trees.train_loss_[-1], 1.0)

    def test_feature_importance_finds_the_only_informative_feature(self):
        X = [[float(value), 1.0] for value in range(12)]
        y = [0.0] * 6 + [10.0] * 6

        model = GBDTRegressorScratch(
            n_estimators=3,
            learning_rate=0.0,
            max_depth=1,
        ).fit(X, y)

        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertGreater(model.feature_importances_[0], model.feature_importances_[1])

    def test_models_package_exports_both_gbdt_regressors(self):
        import Models
        from Models.gbdt_regression_optimized import OptimizedGBDTRegressorScratch

        self.assertIs(Models.GBDTRegressorScratch, GBDTRegressorScratch)
        self.assertIs(
            Models.OptimizedGBDTRegressorScratch,
            OptimizedGBDTRegressorScratch,
        )


if __name__ == "__main__":
    unittest.main()
