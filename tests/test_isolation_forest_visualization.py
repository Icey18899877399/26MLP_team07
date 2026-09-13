import math
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.isolation_forest_figures import (
    FIGURE_FILENAMES,
    PALETTES,
    PLOT_STYLE,
    binary_classification_metrics,
    binary_ranking_curves,
    fit_standardizer,
    generate_oblique_data,
    load_npz_dataset,
    render_all,
    run_experiment,
    stratified_split,
    transform_with_standardizer,
)


class IsolationForestVisualizationCalculationTests(unittest.TestCase):
    def test_ranking_curves_match_a_hand_checked_example(self):
        y_true = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.35, 0.8]

        curves = binary_ranking_curves(y_true, scores)

        self.assertAlmostEqual(curves["roc_auc"], 0.75)
        self.assertAlmostEqual(curves["average_precision"], 5.0 / 6.0)
        self.assertEqual(curves["roc_fpr"][0], 0.0)
        self.assertEqual(curves["roc_tpr"][-1], 1.0)

    def test_equal_scores_reduce_to_chance_ranking(self):
        curves = binary_ranking_curves([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])

        self.assertAlmostEqual(curves["roc_auc"], 0.5)
        self.assertAlmostEqual(curves["average_precision"], 0.5)

    def test_threshold_metrics_use_anomaly_as_the_positive_class(self):
        metrics = binary_classification_metrics(
            [0, 0, 1, 1],
            [1, -1, -1, 1],
        )

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)

    def test_standardizer_uses_training_statistics_only(self):
        train = [[0.0, 10.0], [2.0, 14.0]]
        validation = [[100.0, 1000.0]]

        means, scales = fit_standardizer(train)
        transformed = transform_with_standardizer(validation, means, scales)

        self.assertEqual(means, [1.0, 12.0])
        self.assertEqual(scales, [1.0, 2.0])
        self.assertEqual(transformed, [[99.0, 494.0]])

    def test_stratified_split_is_complete_disjoint_and_reproducible(self):
        X = [[float(index)] for index in range(20)]
        y = [0] * 15 + [1] * 5

        first = stratified_split(X, y, seed=17)
        second = stratified_split(X, y, seed=17)

        all_indices = (
            first["train_indices"]
            + first["validation_indices"]
            + first["test_indices"]
        )
        self.assertEqual(sorted(all_indices), list(range(20)))
        self.assertEqual(len(set(all_indices)), 20)
        self.assertEqual(first, second)
        self.assertGreater(sum(first["train_y"]), 0)
        self.assertGreater(sum(first["validation_y"]), 0)
        self.assertGreater(sum(first["test_y"]), 0)

    def test_oblique_generator_is_reproducible_and_labels_requested_outliers(self):
        first_X, first_y = generate_oblique_data(
            n_inliers=40,
            n_outliers=6,
            seed=29,
        )
        second_X, second_y = generate_oblique_data(
            n_inliers=40,
            n_outliers=6,
            seed=29,
        )

        self.assertEqual(first_X, second_X)
        self.assertEqual(first_y, second_y)
        self.assertEqual(len(first_X), 46)
        self.assertEqual(sum(first_y), 6)


class IsolationForestVisualizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cardio_path = PROJECT_ROOT / "data" / "anomaly" / "6_cardio.npz"
        cls.mammography_path = (
            PROJECT_ROOT / "data" / "anomaly" / "23_mammography.npz"
        )
        cls.report = run_experiment(
            cardio_path=cls.cardio_path,
            mammography_path=cls.mammography_path,
            seed=13,
            quick=True,
        )

    def test_npz_loader_reads_features_and_binary_labels(self):
        X, y = load_npz_dataset(self.cardio_path)

        self.assertEqual(len(X), 1831)
        self.assertEqual(len(X[0]), 21)
        self.assertEqual(sum(y), 176)

    def test_reduced_experiment_contains_all_six_evidence_layers(self):
        self.assertEqual(
            set(self.report["figures"]),
            {
                "mechanism",
                "score_landscape",
                "score_distribution",
                "ranking_curves",
                "parameter_sensitivity",
                "benchmark",
            },
        )
        self.assertEqual(
            self.report["figure_contract"]["archetype"],
            "quantitative grid with a mechanism-led opening",
        )

    def test_tuning_report_discloses_validation_only_selection(self):
        tuning = self.report["figures"]["parameter_sensitivity"]

        self.assertEqual(tuning["selection_split"], "validation")
        self.assertFalse(tuning["test_used_for_tuning"])
        self.assertGreater(len(tuning["grid"]), 1)
        self.assertIn("max_samples", tuning["selected"])
        self.assertIn("max_features", tuning["selected"])
        self.assertIn("baseline_selected", tuning)
        self.assertEqual(
            tuning["n_estimators"],
            self.report["figures"]["ranking_curves"]["n_estimators"],
        )
        self.assertEqual(
            tuning["selection_rule"],
            "maximize validation AUPRC; deterministic complexity tie-break",
        )
        self.assertNotIn("fit_time", tuning["selection_rule"])

    def test_parameter_sensitivity_uses_repeated_timing_summaries(self):
        tuning = self.report["figures"]["parameter_sensitivity"]

        self.assertEqual(
            tuning["timing_protocol"],
            "warm-up; rotated configuration order; median and IQR",
        )
        for row in tuning["baseline_grid"] + tuning["grid"] + tuning["attempts"]:
            self.assertIn("fit_time_median", row)
            self.assertIn("fit_time_q1", row)
            self.assertIn("fit_time_q3", row)
            self.assertLessEqual(row["fit_time_q1"], row["fit_time_median"])
            self.assertLessEqual(row["fit_time_median"], row["fit_time_q3"])

    def test_cardio_tuning_records_never_enter_the_benchmark_test_partition(self):
        tuning_indices = set(
            self.report["figures"]["parameter_sensitivity"][
                "selection_record_indices"
            ]
        )
        benchmark_indices = set(
            self.report["figures"]["benchmark"]["datasets"]["Cardio"][
                "test_indices"
            ]
        )

        self.assertTrue(tuning_indices.isdisjoint(benchmark_indices))
        self.assertEqual(
            self.report["figures"]["benchmark"]["tuning_test_overlap_count"],
            0,
        )

    def test_score_distribution_exposes_each_training_calibrated_threshold(self):
        distribution = self.report["figures"]["score_distribution"]

        self.assertEqual(
            distribution["threshold_calibration"],
            "oracle training prevalence",
        )
        for method in ("Isolation Forest", "Extended IF"):
            self.assertIsInstance(
                distribution["methods"][method]["model_threshold"],
                float,
            )

    def test_benchmark_reports_both_datasets_metrics_and_repeated_seeds(self):
        benchmark = self.report["figures"]["benchmark"]

        self.assertEqual(set(benchmark["datasets"]), {"Cardio", "Mammography"})
        for dataset in benchmark["datasets"].values():
            self.assertGreaterEqual(len(dataset["seeds"]), 2)
            for method in ("Isolation Forest", "Extended IF"):
                self.assertIn("roc_auc_mean", dataset["methods"][method])
                self.assertIn("average_precision_mean", dataset["methods"][method])
                self.assertIn("fit_time_mean", dataset["methods"][method])
                self.assertIn("fit_time_median", dataset["methods"][method])
                self.assertIn("fit_time_q1", dataset["methods"][method])
                self.assertIn("fit_time_q3", dataset["methods"][method])
        self.assertEqual(
            benchmark["timing_protocol"],
            "warm-up; alternating fit order; one scoring pass; median and IQR across model seeds",
        )

    def test_renderer_creates_exactly_six_png_files_without_layout_warnings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                paths = render_all(self.report, output)

            self.assertEqual({path.name for path in paths}, set(FIGURE_FILENAMES))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(FIGURE_FILENAMES),
            )
            self.assertTrue(all(path.suffix == ".png" for path in paths))
            from PIL import Image

            for path in paths:
                with Image.open(path) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertGreater(image.height, 600)
                    self.assertAlmostEqual(image.info["dpi"][0], 300.0, delta=0.2)
            layout_messages = [
                str(item.message)
                for item in caught
                if "layout" in str(item.message).lower()
            ]
            self.assertEqual(layout_messages, [])

    def test_uses_times_new_roman_and_six_distinct_palettes(self):
        self.assertEqual(PLOT_STYLE["font.family"], "serif")
        self.assertEqual(PLOT_STYLE["font.serif"], ["Times New Roman"])
        self.assertEqual(len(PALETTES), 6)
        self.assertEqual(len(set(PALETTES.values())), 6)

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "isolation_forest_figures.py"

        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--cardio", completed.stdout)
        self.assertIn("--mammography", completed.stdout)
        self.assertIn("--output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
