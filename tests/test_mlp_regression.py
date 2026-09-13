import math
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.mlp_regression import MLPRegressorScratch


def rmse(actual, predicted):
    return math.sqrt(
        sum((target - estimate) ** 2 for target, estimate in zip(actual, predicted))
        / len(actual)
    )


class MLPRegressorScratchTests(unittest.TestCase):
    def test_minimal_examples_run_directly_and_print_one_prediction(self):
        scripts = (
            PROJECT_ROOT / "examples" / "mlp_regression_minimal.py",
            PROJECT_ROOT / "examples" / "mlp_regression_optimized_minimal.py",
        )

        for script in scripts:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = completed.stdout.strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(math.isfinite(float(lines[0])))

    def test_learns_a_small_nonlinear_curve(self):
        X = [[value / 4.0] for value in range(-8, 9)]
        y = [row[0] ** 2 for row in X]

        model = MLPRegressorScratch(
            hidden_size=12,
            learning_rate=0.03,
            max_iter=4_000,
            random_state=7,
        ).fit(X, y)

        self.assertLess(rmse(y, model.predict(X)), 0.20)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_fixed_seed_reproduces_parameters_and_predictions(self):
        X = [[-1.0], [0.0], [1.0], [2.0]]
        y = [1.0, 0.0, 1.0, 4.0]
        parameters = {
            "hidden_size": 5,
            "learning_rate": 0.02,
            "max_iter": 100,
            "random_state": 19,
        }

        first = MLPRegressorScratch(**parameters).fit(X, y)
        second = MLPRegressorScratch(**parameters).fit(X, y)

        self.assertEqual(first.input_weights_, second.input_weights_)
        self.assertEqual(first.output_weights_, second.output_weights_)
        self.assertEqual(first.predict(X), second.predict(X))

    def test_constructor_preserves_tuning_parameters(self):
        model = MLPRegressorScratch(
            hidden_size=6,
            learning_rate=0.04,
            max_iter=321,
            random_state=5,
        )

        self.assertEqual(model.hidden_size, 6)
        self.assertEqual(model.learning_rate, 0.04)
        self.assertEqual(model.max_iter, 321)
        self.assertEqual(model.random_state, 5)

    def test_models_package_exports_both_mlp_regressors(self):
        import Models
        from Models.mlp_regression_optimized import OptimizedMLPRegressorScratch

        self.assertIs(Models.MLPRegressorScratch, MLPRegressorScratch)
        self.assertIs(
            Models.OptimizedMLPRegressorScratch,
            OptimizedMLPRegressorScratch,
        )


if __name__ == "__main__":
    unittest.main()
