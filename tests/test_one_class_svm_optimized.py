import importlib
import math
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OptimizedOneClassSVMModuleContractTests(unittest.TestCase):
    def test_optimized_module_exposes_the_scratch_model(self):
        try:
            module = importlib.import_module("Models.one_class_svm_optimized")
        except ModuleNotFoundError:
            self.fail("Models.one_class_svm_optimized is missing")

        self.assertTrue(hasattr(module, "OptimizedOneClassSVMScratch"))

    def test_optimized_model_exposes_the_training_and_inference_api(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        model = module.OptimizedOneClassSVMScratch()

        for method in (
            "fit",
            "score_samples",
            "decision_function",
            "predict",
            "fit_predict",
        ):
            self.assertTrue(callable(getattr(model, method, None)), method)

    def test_constructor_exposes_kernel_and_optimization_parameters(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        try:
            model = module.OptimizedOneClassSVMScratch(
                nu=0.15,
                kernel="rbf",
                gamma=0.8,
                standardize=True,
                max_iter=240,
                tol=1e-7,
            )
        except TypeError as error:
            self.fail(str(error))

        self.assertEqual(model.nu, 0.15)
        self.assertEqual(model.kernel, "rbf")
        self.assertEqual(model.gamma, 0.8)
        self.assertTrue(model.standardize)
        self.assertEqual(model.max_iter, 240)
        self.assertEqual(model.tol, 1e-7)

    def test_rbf_kernel_learns_a_non_linear_ring_boundary(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        ring = [
            [math.cos(index * math.pi / 12.0), math.sin(index * math.pi / 12.0)]
            for index in range(24)
        ]
        model = module.OptimizedOneClassSVMScratch(
            nu=0.1,
            kernel="rbf",
            gamma=2.0,
            standardize=False,
            max_iter=1200,
            tol=1e-7,
        ).fit(ring)

        ring_predictions = model.predict(ring)
        novel_predictions = model.predict([[0.0, 0.0], [3.0, 3.0]])

        self.assertGreaterEqual(ring_predictions.count(1), 20)
        self.assertEqual(novel_predictions, [-1, -1])

    def test_standardization_uses_training_statistics_only(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        X = [[10.0, 900.0], [12.0, 1000.0], [14.0, 1100.0]]
        model = module.OptimizedOneClassSVMScratch(
            nu=1.0 / 3.0,
            gamma="scale",
            standardize=True,
            max_iter=300,
        ).fit(X)

        means_before = list(getattr(model, "means_", []))
        scales_before = list(getattr(model, "scales_", []))
        model.score_samples([[1000.0, -5000.0]])

        self.assertEqual(means_before, [12.0, 1000.0])
        self.assertAlmostEqual(scales_before[0], math.sqrt(8.0 / 3.0))
        self.assertAlmostEqual(scales_before[1], math.sqrt(20000.0 / 3.0))
        self.assertEqual(model.means_, means_before)
        self.assertEqual(model.scales_, scales_before)
        self.assertAlmostEqual(model.gamma_, 0.5)

    def test_scale_gamma_is_invariant_to_translation_without_standardization(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        original = module.OptimizedOneClassSVMScratch(
            gamma="scale",
            standardize=False,
            max_iter=20,
        ).fit([[0.0], [1.0], [2.0]])
        shifted = module.OptimizedOneClassSVMScratch(
            gamma="scale",
            standardize=False,
            max_iter=20,
        ).fit([[1000.0], [1001.0], [1002.0]])

        self.assertAlmostEqual(original.gamma_, 1.5)
        self.assertAlmostEqual(shifted.gamma_, original.gamma_)

    def test_kernel_cache_and_support_only_inference_state_are_recorded(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        X = [[value / 5.0, (value % 3) / 3.0] for value in range(12)]
        model = module.OptimizedOneClassSVMScratch(
            nu=0.25,
            gamma=1.0,
            standardize=False,
            max_iter=500,
        ).fit(X)

        kernel_matrix = getattr(model, "kernel_matrix_", [])
        self.assertEqual(len(kernel_matrix), len(X))
        self.assertTrue(all(len(row) == len(X) for row in kernel_matrix))
        self.assertEqual(len(model.support_vectors_), len(model.support_alphas_))
        self.assertEqual(
            model.support_indices_,
            [index for index, alpha in enumerate(model.alphas_) if alpha > 1e-8],
        )
        self.assertLessEqual(model.n_iter_, model.max_iter)
        self.assertGreaterEqual(len(model.objective_history_), 1)

    def test_frank_wolfe_keeps_the_dual_feasible(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        X = [[float(value), float(value % 2)] for value in range(10)]
        model = module.OptimizedOneClassSVMScratch(
            nu=0.2,
            kernel="linear",
            standardize=False,
            max_iter=400,
            tol=1e-8,
        ).fit(X)

        upper = 1.0 / (model.nu * len(X))
        self.assertAlmostEqual(sum(model.alphas_), 1.0, places=8)
        self.assertTrue(
            all(-1e-10 <= alpha <= upper + 1e-8 for alpha in model.alphas_)
        )

    def test_linear_kernel_rejects_mean_centering_that_collapses_the_model(self):
        module = importlib.import_module("Models.one_class_svm_optimized")

        with self.assertRaises(ValueError):
            module.OptimizedOneClassSVMScratch(
                kernel="linear",
                standardize=True,
            )

    def test_nu_one_offset_satisfies_the_capped_support_kkt_bound(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        model = module.OptimizedOneClassSVMScratch(
            nu=1.0,
            kernel="linear",
            standardize=False,
            max_iter=20,
        ).fit(X)

        self.assertGreaterEqual(
            model.offset_, max(model.score_samples(X)) - 1e-10
        )

    def test_rejects_invalid_kernel_parameters_and_input(self):
        module = importlib.import_module("Models.one_class_svm_optimized")
        invalid_settings = (
            {"nu": 0.0},
            {"kernel": "polynomial"},
            {"gamma": 0.0},
            {"standardize": "yes"},
            {"max_iter": 0},
            {"tol": -1.0},
            {"gamma": math.inf},
            {"gamma": math.nan},
            {"tol": math.inf},
            {"tol": math.nan},
        )
        for settings in invalid_settings:
            with self.assertRaises(ValueError):
                module.OptimizedOneClassSVMScratch(**settings)

        model = module.OptimizedOneClassSVMScratch()
        with self.assertRaises(ValueError):
            model.decision_function([[0.0]])
        with self.assertRaises(ValueError):
            model.fit([])
        model.fit([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            model.predict([[0.0]])

    def test_optimized_example_runs_and_rejects_distant_points(self):
        script = PROJECT_ROOT / "examples" / "one_class_svm_optimized_minimal.py"
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
