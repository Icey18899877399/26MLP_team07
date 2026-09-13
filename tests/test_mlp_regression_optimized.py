import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.mlp_regression_optimized import OptimizedMLPRegressorScratch


def rmse(actual, predicted):
    return math.sqrt(
        sum((target - estimate) ** 2 for target, estimate in zip(actual, predicted))
        / len(actual)
    )


class OptimizedMLPRegressorScratchTests(unittest.TestCase):
    def test_learns_a_nonlinear_signal_better_than_the_mean(self):
        X = [[value / 10.0, (value % 4) / 3.0] for value in range(-30, 31)]
        y = [1.5 * row[0] ** 2 - 2.0 * row[1] + 0.5 for row in X]
        mean = sum(y) / len(y)
        mean_rmse = rmse(y, [mean] * len(y))

        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(16, 8),
            activation="tanh",
            learning_rate=0.02,
            max_iter=1_500,
            batch_size=16,
            validation_fraction=0.2,
            n_iter_no_change=100,
            random_state=11,
        ).fit(X, y)

        self.assertLess(rmse(y, model.predict(X)), mean_rmse * 0.25)

    def test_standardization_handles_very_different_feature_scales(self):
        X = [[float(value), float(value) * 1_000_000.0] for value in range(-12, 13)]
        y = [2.0 * row[0] + 3.0 for row in X]

        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(8,),
            activation="tanh",
            learning_rate=0.02,
            max_iter=800,
            batch_size=None,
            validation_fraction=0.2,
            n_iter_no_change=80,
            random_state=3,
        ).fit(X, y)
        predictions = model.predict(X)

        self.assertTrue(all(math.isfinite(value) for value in predictions))
        self.assertLess(rmse(y, predictions), 1.0)

    def test_validation_split_and_standardizer_use_training_rows_only(self):
        X = [[float(value)] for value in range(10)]
        y = [float(value) for value in range(10)]

        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(4,),
            max_iter=5,
            batch_size=None,
            validation_fraction=0.3,
            n_iter_no_change=2,
            random_state=13,
        ).fit(X, y)

        expected_mean = sum(X[index][0] for index in model.training_indices_) / len(model.training_indices_)
        self.assertAlmostEqual(model.feature_means_[0], expected_mean)
        self.assertFalse(set(model.training_indices_) & set(model.validation_indices_))
        self.assertEqual(
            set(model.training_indices_) | set(model.validation_indices_),
            set(range(len(X))),
        )

    def test_early_stopping_restores_the_best_epoch(self):
        X = [[float(value)] for value in range(30)]
        y = [4.0] * len(X)

        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(6,),
            learning_rate=0.01,
            max_iter=200,
            batch_size=None,
            validation_fraction=0.2,
            n_iter_no_change=5,
            tol=1e-8,
            random_state=17,
        ).fit(X, y)

        self.assertLess(model.n_iter_, model.max_iter)
        self.assertLessEqual(model.best_iteration_, model.n_iter_)
        self.assertEqual(model.n_iter_, len(model.validation_loss_))
        self.assertLess(rmse(y, model.predict(X)), 0.05)

    def test_tolerance_does_not_discard_strictly_better_parameters(self):
        X = [[-1.0], [0.0], [1.0]]
        y = [0.0, 1.0, 2.0]

        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(2,),
            activation="tanh",
            learning_rate=0.01,
            max_iter=10,
            batch_size=None,
            tol=1.0,
            validation_fraction=0.0,
            n_iter_no_change=3,
            random_state=42,
        ).fit(X, y)
        restored_loss = (
            sum(
                (target - prediction) ** 2
                for target, prediction in zip(y, model.predict(X))
            )
            / len(y)
            / (model.target_scale_ ** 2)
        )

        self.assertEqual(model.n_iter_, len(model.validation_loss_))
        self.assertGreater(model.n_iter_, 1)
        self.assertAlmostEqual(restored_loss, min(model.validation_loss_), places=12)

    def test_positive_validation_fraction_keeps_both_partitions_nonempty(self):
        for fraction in (0.2, 0.9):
            with self.subTest(fraction=fraction):
                model = OptimizedMLPRegressorScratch(
                    hidden_layer_sizes=(2,),
                    max_iter=1,
                    validation_fraction=fraction,
                    random_state=5,
                ).fit([[0.0], [1.0]], [0.0, 1.0])

                self.assertEqual(len(model.training_indices_), 1)
                self.assertEqual(len(model.validation_indices_), 1)

    def test_single_sample_is_used_for_training_without_validation(self):
        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(2,),
            max_iter=1,
            validation_fraction=0.9,
            random_state=5,
        ).fit([[2.0]], [7.0])

        self.assertEqual(model.training_indices_, [0])
        self.assertEqual(model.validation_indices_, [])

    def test_fixed_seed_reproduces_minibatch_training(self):
        X = [[float(value), float(value % 3)] for value in range(24)]
        y = [0.5 * row[0] - row[1] for row in X]
        parameters = {
            "hidden_layer_sizes": (7, 4),
            "activation": "relu",
            "learning_rate": 0.005,
            "max_iter": 40,
            "batch_size": 6,
            "validation_fraction": 0.25,
            "n_iter_no_change": 10,
            "random_state": 23,
        }

        first = OptimizedMLPRegressorScratch(**parameters).fit(X, y)
        second = OptimizedMLPRegressorScratch(**parameters).fit(X, y)

        self.assertEqual(first.weights_, second.weights_)
        self.assertEqual(first.biases_, second.biases_)
        self.assertEqual(first.predict(X), second.predict(X))

    def test_multiple_hidden_layers_define_parameter_shapes(self):
        model = OptimizedMLPRegressorScratch(
            hidden_layer_sizes=(5, 3),
            max_iter=1,
            validation_fraction=0.0,
            random_state=2,
        ).fit([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0])

        self.assertEqual(
            [(len(layer), len(layer[0])) for layer in model.weights_],
            [(2, 5), (5, 3), (3, 1)],
        )


if __name__ == "__main__":
    unittest.main()
