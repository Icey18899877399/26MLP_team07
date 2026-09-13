"""One-Class SVM 可视化的行为契约测试。"""

import importlib
import math
import tempfile
import unittest
from pathlib import Path


class VisualizationAvailabilityTests(unittest.TestCase):
    def test_visualization_module_is_available(self):
        try:
            module = importlib.import_module("visualization.one_class_svm_figures")
        except ModuleNotFoundError:
            self.fail("visualization.one_class_svm_figures is not implemented")
        self.assertTrue(hasattr(module, "FIGURE_FILENAMES"))


class MetricAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viz = importlib.import_module("visualization.one_class_svm_figures")

    def test_ranking_metrics_match_hand_checked_example(self):
        result = self.viz.binary_ranking_curves(
            [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
        )
        self.assertAlmostEqual(result["roc_auc"], 0.75, places=12)
        self.assertAlmostEqual(result["average_precision"], 5.0 / 6.0, places=12)
        self.assertEqual(result["fpr"][0], 0.0)
        self.assertEqual(result["tpr"][-1], 1.0)

    def test_tied_scores_reduce_to_chance_ranking(self):
        result = self.viz.binary_ranking_curves([1, 0, 1, 0], [2.0] * 4)
        self.assertAlmostEqual(result["roc_auc"], 0.5, places=12)
        self.assertAlmostEqual(result["average_precision"], 0.5, places=12)

    def test_intrinsic_zero_threshold_marks_negative_decisions_as_anomalies(self):
        result = self.viz.threshold_metrics(
            [0, 1, 1, 0], [0.3, -0.2, 0.0, -0.1]
        )
        self.assertEqual(result["confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertAlmostEqual(result["f1"], 0.5, places=12)

    def test_stratified_split_is_complete_disjoint_and_reproducible(self):
        labels = [0] * 30 + [1] * 10
        first = self.viz.stratified_split(labels, seed=17)
        second = self.viz.stratified_split(labels, seed=17)
        self.assertEqual(first, second)
        combined = first["train"] + first["validation"] + first["test"]
        self.assertEqual(sorted(combined), list(range(40)))
        self.assertEqual(len(combined), len(set(combined)))
        self.assertGreater(sum(labels[i] for i in first["validation"]), 0)
        self.assertGreater(sum(labels[i] for i in first["test"]), 0)

    def test_basic_scaler_changes_scale_without_mean_centering(self):
        train = [[10.0, 100.0], [14.0, 104.0]]
        scales = self.viz.fit_scale_only(train)
        transformed = self.viz.transform_scale_only([[12.0, 102.0]], scales)
        self.assertEqual(scales, [2.0, 2.0])
        self.assertEqual(transformed, [[6.0, 51.0]])

    def test_clean_reference_selection_never_contains_anomalies(self):
        labels = [0, 1, 0, 0, 1, 0]
        selected = self.viz.select_clean_reference(
            [0, 1, 2, 3, 4, 5], labels, limit=3, seed=9
        )
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(labels[index] == 0 for index in selected))

    def test_evaluation_cap_preserves_the_source_prevalence(self):
        labels = [0] * 90 + [1] * 10
        selected = self.viz.stratified_subsample(
            list(range(100)), labels, max_total=50, seed=5
        )
        self.assertEqual(len(selected), 50)
        self.assertEqual(sum(labels[index] for index in selected), 5)

    def test_support_categories_follow_box_constraint(self):
        categories = self.viz.support_categories(
            [0.0, 0.1, 0.5, 0.50000000001], cap=0.5, tolerance=1e-8
        )
        self.assertEqual(categories, ["zero", "free", "capped", "capped"])

    def test_effective_support_fraction_ignores_numerical_dust(self):
        fraction = self.viz.effective_support_fraction(
            [0.00001, 0.02, 0.499, 0.5], cap=0.5, relative_tolerance=0.001
        )
        self.assertAlmostEqual(fraction, 0.75, places=12)

    def test_relative_objective_does_not_force_the_last_point_to_zero(self):
        self.assertEqual(self.viz.relative_objective([4.0, 2.0, 1.0]), [1.0, 0.5, 0.25])

    def test_injection_count_targets_final_contamination_fraction(self):
        self.assertEqual(self.viz.injection_count(90, 0.10), 10)


class ExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viz = importlib.import_module("visualization.one_class_svm_figures")
        cls.stage_root = Path(__file__).resolve().parents[1]
        cls.report = cls.viz.run_experiment(
            cls.stage_root / "data" / "anomaly", quick=True
        )

    def test_real_datasets_load_with_expected_shapes(self):
        cardio = self.viz.load_npz_dataset(
            self.stage_root / "data" / "anomaly" / "6_cardio.npz"
        )
        mammography = self.viz.load_npz_dataset(
            self.stage_root / "data" / "anomaly" / "23_mammography.npz"
        )
        self.assertEqual((len(cardio["X"]), len(cardio["X"][0])), (1831, 21))
        self.assertEqual(sum(cardio["y"]), 176)
        self.assertEqual((len(mammography["X"]), len(mammography["X"][0])), (11183, 6))
        self.assertEqual(sum(mammography["y"]), 260)

    def test_report_preserves_evaluation_integrity(self):
        protocol = self.report["protocol"]
        self.assertEqual(protocol["backend"], "Python")
        self.assertEqual(protocol["output_format"], "PNG only")
        self.assertEqual(protocol["font"], "Times New Roman")
        self.assertEqual(protocol["threshold"], "intrinsic decision = 0")
        self.assertEqual(protocol["tuning_split"], "validation")
        self.assertFalse(protocol["test_used_for_tuning"])
        self.assertEqual(
            protocol["timing"],
            "warm-up, alternating method order, median and interquartile range",
        )
        self.assertIn("normal-only", protocol["training_policy"])
        self.assertEqual(protocol["effective_support_threshold"], "alpha > 0.01C")
        self.assertEqual(
            protocol["evaluation_sampling"],
            "stratified cap preserving validation/test prevalence",
        )

    def test_boundary_evidence_includes_the_ocsvm_origin(self):
        self.assertEqual(self.report["mechanism"]["origin"], [0.0, 0.0])

    def test_report_uses_six_distinct_figure_palettes(self):
        self.assertEqual(set(self.report["figures"]), set(self.viz.FIGURE_FILENAMES))
        signatures = [tuple(colors) for colors in self.report["palettes"].values()]
        self.assertEqual(len(signatures), 6)
        self.assertEqual(len(set(signatures)), 6)

    def test_benchmark_covers_two_datasets_and_uncertainty_summaries(self):
        benchmark = self.report["benchmark"]
        self.assertEqual(set(benchmark), {"Cardio", "Mammography"})
        for dataset in benchmark.values():
            self.assertGreaterEqual(dataset["n_seeds"], 2)
            self.assertEqual(
                dataset["basic"]["train_normal_cap"],
                dataset["optimized"]["train_normal_cap"],
            )
            for method in ("basic", "optimized"):
                self.assertIn("mean", dataset[method]["auprc"])
                self.assertIn("std", dataset[method]["auprc"])
                self.assertIn("median", dataset[method]["fit_time_ms"])
                self.assertIn("q1", dataset[method]["fit_time_ms"])
                self.assertIn("q3", dataset[method]["fit_time_ms"])

    def test_runtime_and_robustness_statistics_match_their_labels(self):
        runtime = self.report["runtime"]
        for method in ("basic", "optimized"):
            medians = runtime[f"{method}_time_ms"]
            q1 = runtime[f"{method}_time_q1_ms"]
            q3 = runtime[f"{method}_time_q3_ms"]
            self.assertTrue(all(low <= middle <= high for low, middle, high in zip(q1, medians, q3)))
        robustness = self.report["robustness"]
        self.assertTrue(
            all(actual + 1e-12 >= nu for actual, nu in zip(
                robustness["actual_support_fractions"], robustness["nu_values"]
            ))
        )
        self.assertEqual(robustness["actual_contamination"], robustness["contamination"])

    def test_clean_training_ids_and_test_ids_are_auditable(self):
        for dataset in self.report["audit"].values():
            self.assertTrue(set(dataset["validation_ids"]).isdisjoint(dataset["test_ids"]))
            self.assertTrue(all(label == 0 for label in dataset["selected_train_labels"]))

    def test_renderer_creates_exactly_six_high_resolution_png_files(self):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment check
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.viz.render_all(self.report, tmp)
            names = sorted(path.name for path in paths)
            self.assertEqual(names, sorted(self.viz.FIGURE_FILENAMES))
            self.assertEqual(
                sorted(path.name for path in Path(tmp).iterdir()), names
            )
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreaterEqual(image.width, 1200)
                    self.assertGreaterEqual(image.height, 700)
                    self.assertTrue(math.isclose(image.info["dpi"][0], 300, rel_tol=0.02))


if __name__ == "__main__":
    unittest.main()
