import importlib
import math
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OneClassSVMModuleContractTests(unittest.TestCase):
    def test_basic_module_exposes_the_scratch_model(self):
        try:
            module = importlib.import_module("Models.one_class_svm")
        except ModuleNotFoundError:
            self.fail("Models.one_class_svm is missing")

        self.assertTrue(hasattr(module, "OneClassSVMScratch"))

    def test_basic_model_exposes_the_training_and_inference_api(self):
        module = importlib.import_module("Models.one_class_svm")
        model = module.OneClassSVMScratch()

        for method in (
            "fit",
            "score_samples",
            "decision_function",
            "predict",
            "fit_predict",
        ):
            self.assertTrue(callable(getattr(model, method, None)), method)

    def test_constructor_exposes_basic_tuning_parameters(self):
        module = importlib.import_module("Models.one_class_svm")
        try:
            model = module.OneClassSVMScratch(
                nu=0.2,
                learning_rate=0.75,
                max_iter=120,
                tol=1e-7,
            )
        except TypeError as error:
            self.fail(str(error))

        self.assertEqual(model.nu, 0.2)
        self.assertEqual(model.learning_rate, 0.75)
        self.assertEqual(model.max_iter, 120)
        self.assertEqual(model.tol, 1e-7)

    def test_linear_model_rejects_a_point_on_the_opposite_side_of_origin(self):
        module = importlib.import_module("Models.one_class_svm")
        train_X = [[1.5], [1.8], [2.0], [2.2], [2.5]]
        model = module.OneClassSVMScratch(
            nu=0.2,
            learning_rate=1.0,
            max_iter=1200,
            tol=1e-8,
        ).fit(train_X)

        predictions = model.predict(train_X + [[-3.0]])

        self.assertEqual(len(predictions), 6)
        self.assertGreaterEqual(predictions[:5].count(1), 4)
        self.assertEqual(predictions[-1], -1)

    def test_dual_coefficients_obey_sum_and_box_constraints(self):
        module = importlib.import_module("Models.one_class_svm")
        X = [[1.0], [1.5], [2.0], [2.5], [3.0]]
        model = module.OneClassSVMScratch(
            nu=0.4,
            learning_rate=1.0,
            max_iter=800,
            tol=1e-8,
        ).fit(X)

        alphas = getattr(model, "alphas_", [])
        upper = 1.0 / (model.nu * len(X))
        self.assertEqual(len(alphas), len(X))
        self.assertAlmostEqual(sum(alphas), 1.0, places=7)
        self.assertTrue(all(-1e-10 <= value <= upper + 1e-8 for value in alphas))
        self.assertEqual(
            model.support_indices_,
            [index for index, value in enumerate(alphas) if value > 1e-8],
        )

    def test_nu_one_offset_satisfies_the_capped_support_kkt_bound(self):
        module = importlib.import_module("Models.one_class_svm")
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        model = module.OneClassSVMScratch(nu=1.0, max_iter=20).fit(X)
        training_scores = model.score_samples(X)

        self.assertGreaterEqual(model.offset_, max(training_scores) - 1e-10)

    def test_backtracking_never_accepts_an_objective_increase(self):
        module = importlib.import_module("Models.one_class_svm")
        model = module.OneClassSVMScratch(
            nu=0.5,
            learning_rate=1e100,
            max_iter=3,
            tol=0.0,
        ).fit([[1.0], [2.0], [3.0]])

        self.assertTrue(
            all(
                later <= earlier + 1e-14
                for earlier, later in zip(
                    model.objective_history_, model.objective_history_[1:]
                )
            )
        )

    def test_score_and_decision_function_differ_by_the_learned_offset(self):
        module = importlib.import_module("Models.one_class_svm")
        model = module.OneClassSVMScratch(nu=0.25, max_iter=600).fit(
            [[1.0], [1.4], [1.8], [2.2]]
        )
        query = [[1.5], [-2.0]]
        scores = model.score_samples(query)
        decisions = model.decision_function(query)

        self.assertEqual(len(scores), 2)
        self.assertEqual(len(decisions), 2)
        for score, decision in zip(scores, decisions):
            self.assertAlmostEqual(score - model.offset_, decision, places=10)

    def test_fit_predict_returns_a_copy_of_training_labels(self):
        module = importlib.import_module("Models.one_class_svm")
        model = module.OneClassSVMScratch(nu=0.2, max_iter=500)

        labels = model.fit_predict([[1.0], [1.2], [1.4], [1.6], [1.8]])

        self.assertEqual(labels, model.labels_)
        self.assertIsNot(labels, model.labels_)

    def test_rejects_invalid_parameters_and_input(self):
        module = importlib.import_module("Models.one_class_svm")
        invalid_settings = (
            {"nu": 0.0},
            {"nu": 1.1},
            {"learning_rate": 0.0},
            {"max_iter": 0},
            {"tol": -1.0},
            {"learning_rate": math.inf},
            {"learning_rate": math.nan},
            {"tol": math.inf},
            {"tol": math.nan},
        )
        for settings in invalid_settings:
            with self.assertRaises(ValueError):
                module.OneClassSVMScratch(**settings)

        model = module.OneClassSVMScratch()
        with self.assertRaises(ValueError):
            model.predict([[1.0]])
        with self.assertRaises(ValueError):
            model.fit([])
        with self.assertRaises(ValueError):
            model.fit([[1.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            model.fit([[math.inf]])
        model.fit([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            model.predict([[1.0, 2.0]])

    def test_models_package_exports_both_versions(self):
        models = importlib.import_module("Models")

        self.assertTrue(hasattr(models, "OneClassSVMScratch"))
        self.assertTrue(hasattr(models, "OptimizedOneClassSVMScratch"))

    def test_minimal_example_runs_and_marks_the_opposite_point_as_anomaly(self):
        script = PROJECT_ROOT / "examples" / "one_class_svm_minimal.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[1, 1, 1, 1, 1, -1]")


if __name__ == "__main__":
    unittest.main()
