import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.linear_regression_figures import (  # noqa: E402
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    PLOT_STYLE,
    build_parser,
    build_optimized_model,
    kfold_indices,
    load_concrete_csv,
    pearson_correlation,
    regression_metrics,
    render_six_figures,
    split_indices,
    stop_annotation,
    qq_reference,
)


def sample_report():
    true = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    return {
        "data": {
            "target_values": [8.0, 11.0, 16.0, 22.0, 31.0, 42.0, 55.0],
            "feature_names": ["Cement", "Water", "Age", "Fly ash"],
            "correlations": [0.52, -0.29, 0.34, -0.08],
        },
        "convergence": {
            "iterations": [1, 10, 50, 100, 250, 500],
            "base_train": [900, 410, 230, 190, 170, 164],
            "base_validation": [940, 450, 260, 220, 205, 202],
            "optimized_train": [720, 240, 150, 130, 124, 123],
            "optimized_validation": [760, 270, 175, 155, 149, 148],
            "early_stop": 250,
        },
        "regularization": {
            "l2_values": [0.0, 0.001, 0.01, 0.1, 1.0],
            "cv_mean": [12.8, 12.4, 11.9, 12.1, 14.0],
            "cv_std": [1.2, 1.0, 0.8, 0.9, 1.4],
            "selected_l2": 0.01,
            "coefficient_names": ["Cement", "Water", "Age", "Fly ash"],
            "coefficient_paths": [
                [0.42, 0.41, 0.38, 0.29, 0.12],
                [-0.31, -0.30, -0.27, -0.19, -0.08],
                [0.25, 0.25, 0.23, 0.18, 0.09],
                [-0.08, -0.08, -0.07, -0.05, -0.02],
            ],
        },
        "prediction": {
            "true": true,
            "Base": {
                "predicted": [12.0, 18.0, 34.0, 37.0, 47.0, 55.0],
                "metrics": {"rmse": 3.51, "mae": 3.17, "r2": 0.95},
            },
            "Optimized": {
                "predicted": [11.0, 20.0, 31.0, 39.0, 50.0, 59.0],
                "metrics": {"rmse": 0.91, "mae": 0.67, "r2": 0.997},
            },
        },
        "residuals": {
            "Base": {
                "fitted": [12.0, 18.0, 34.0, 37.0, 47.0, 55.0],
                "residuals": [-2.0, 2.0, -4.0, 3.0, 3.0, 5.0],
                "qq_expected": [-1.3, -0.7, -0.2, 0.2, 0.7, 1.3],
                "qq_observed": [-4.0, -2.0, 2.0, 3.0, 3.0, 5.0],
            },
            "Optimized": {
                "fitted": [11.0, 20.0, 31.0, 39.0, 50.0, 59.0],
                "residuals": [-1.0, 0.0, -1.0, 1.0, 0.0, 1.0],
                "qq_expected": [-1.3, -0.7, -0.2, 0.2, 0.7, 1.3],
                "qq_observed": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            },
        },
        "benchmark": {
            "methods": ["Mean predictor", "Base", "Optimized"],
            "rmse": [17.1, 3.51, 0.91],
            "mae": [14.3, 3.17, 0.67],
            "r2": [0.0, 0.95, 0.997],
            "cv_rmse": {
                "Mean predictor": [17.5, 16.9, 17.2, 16.7, 17.4],
                "Base": [3.8, 3.4, 3.5, 3.2, 3.7],
                "Optimized": [1.0, 0.9, 0.8, 0.95, 0.85],
            },
            "fit_ms": [0.02, 48.0, 19.0],
        },
    }


class LinearRegressionVisualizationCalculationTests(unittest.TestCase):
    def test_optimized_model_factory_matches_tuned_and_final_pipeline(self):
        model = build_optimized_model(l2=0.01, seed=7)

        self.assertEqual(model.batch_size, 32)
        self.assertEqual(model.max_iter, 1_000)
        self.assertEqual(model.tol, 1e-10)

    def test_regression_metrics_match_hand_calculation(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(1 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.5)

    def test_pearson_correlation_preserves_direction(self):
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [6, 4, 2]), -1.0)

    def test_split_indices_are_disjoint_complete_and_reproducible(self):
        first_train, first_test = split_indices(20, test_fraction=0.2, seed=7)
        second_train, second_test = split_indices(20, test_fraction=0.2, seed=7)

        self.assertEqual((first_train, first_test), (second_train, second_test))
        self.assertFalse(set(first_train) & set(first_test))
        self.assertEqual(set(first_train) | set(first_test), set(range(20)))

    def test_kfold_validation_indices_cover_each_sample_once(self):
        folds = kfold_indices(17, folds=5, seed=11)

        flattened = [index for fold in folds for index in fold]
        self.assertEqual(sorted(flattened), list(range(17)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_stop_annotation_distinguishes_early_stop_from_iteration_limit(self):
        self.assertEqual(stop_annotation(240, 1_000), "Early stop = 240")
        self.assertEqual(stop_annotation(1_000, 1_000), "Run limit = 1000")

    def test_qq_reference_uses_each_residual_distribution(self):
        low, high = qq_reference([-1.0, 0.0, 1.0], -1.0, 1.0)

        self.assertEqual(low, (-1.0, -1.0))
        self.assertEqual(high, (1.0, 1.0))


class LinearRegressionVisualizationContractTests(unittest.TestCase):
    def test_default_data_path_exists_and_loads_concrete_dataset(self):
        default_path = build_parser().parse_args([]).data

        features, targets = load_concrete_csv(default_path)

        self.assertEqual(len(features), 1_030)
        self.assertEqual(len(targets), 1_030)
        self.assertTrue(all(len(row) == 8 for row in features))

    def test_uses_times_new_roman_and_six_distinct_palettes(self):
        self.assertEqual(PLOT_STYLE["font.family"], "serif")
        self.assertEqual(PLOT_STYLE["font.serif"][0], "Times New Roman")
        self.assertEqual(len(FIGURE_PALETTES), 6)
        self.assertEqual(len({tuple(colors) for colors in FIGURE_PALETTES.values()}), 6)

    def test_renderer_exports_only_six_png_files_without_layout_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                paths = render_six_figures(sample_report(), output_dir)

            self.assertEqual({path.name for path in paths}, set(FIGURE_FILENAMES))
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(FIGURE_FILENAMES))
            self.assertTrue(all(path.suffix.lower() == ".png" for path in paths))
            self.assertTrue(all(path.stat().st_size > 1_000 for path in paths))
            warning_text = " ".join(str(item.message).lower() for item in captured)
            self.assertNotIn("layout", warning_text)

    def test_renderer_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            note = output_dir / "notes.txt"
            note.write_text("keep", encoding="utf-8")

            render_six_figures(sample_report(), output_dir)

            self.assertEqual(note.read_text(encoding="utf-8"), "keep")

    def test_comparison_panels_remain_in_single_named_images(self):
        expected = {
            "02_optimization_convergence.png",
            "04_actual_vs_predicted.png",
            "05_residual_diagnostics.png",
            "06_final_benchmark_comparison.png",
        }
        self.assertTrue(expected.issubset(set(FIGURE_FILENAMES)))

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "linear_regression_figures.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--data", completed.stdout)
        self.assertIn("--output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
