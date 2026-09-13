import math
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.gaussian_naive_bayes_figures import (
    FIGURE_PALETTES,
    _stability_report,
    classification_metrics,
    probability_curves,
    render_six_figures,
)


class GaussianNaiveBayesVisualizationTests(unittest.TestCase):
    def _sample_report(self):
        return {
            "feature_fit": {
                "feature_name": "Mean radius",
                "values": {"Benign": [9.0, 10.0, 11.0], "Malignant": [15.0, 16.0, 17.0]},
                "grid": [8.0, 12.0, 18.0],
                "densities": {
                    "Benign": [0.02, 0.20, 0.00],
                    "Malignant": [0.00, 0.03, 0.16],
                },
                "means": {"Benign": 10.0, "Malignant": 16.0},
            },
            "surface": {
                "x_values": [9.0, 17.0],
                "y_values": [10.0, 30.0],
                "probabilities": [[0.05, 0.70], [0.20, 0.95]],
                "train_x": [[10.0, 12.0], [16.0, 28.0]],
                "train_y": [0, 1],
                "feature_names": ["Mean radius", "Mean texture"],
            },
            "assumptions": {
                "qq": {
                    "Benign": {"theoretical": [-1.0, 0.0, 1.0], "observed": [-0.9, 0.0, 1.1]},
                    "Malignant": {"theoretical": [-1.0, 0.0, 1.0], "observed": [-1.2, 0.1, 0.9]},
                },
                "correlations": {
                    "Benign": [[1.0, 0.2], [0.2, 1.0]],
                    "Malignant": [[1.0, 0.5], [0.5, 1.0]],
                },
                "feature_names": ["Radius", "Texture"],
            },
            "benchmark": {
                "confusion": {
                    "Base": [[6, 2], [2, 4]],
                    "Optimized": [[7, 1], [1, 5]],
                },
                "metrics": {
                    "Majority": {"accuracy": 0.57, "balanced_accuracy": 0.50, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "Base": {"accuracy": 0.71, "balanced_accuracy": 0.71, "precision": 0.67, "recall": 0.67, "f1": 0.67},
                    "Optimized": {"accuracy": 0.86, "balanced_accuracy": 0.85, "precision": 0.83, "recall": 0.83, "f1": 0.83},
                },
            },
            "diagnostics": {
                "Base": {
                    "roc": {"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.7, 1.0], "auc": 0.75},
                    "pr": {"recall": [0.0, 0.7, 1.0], "precision": [1.0, 0.8, 0.43], "ap": 0.72},
                    "calibration": {"mean_probability": [0.1, 0.5, 0.9], "positive_rate": [0.0, 0.6, 1.0]},
                    "brier": 0.18,
                },
                "Optimized": {
                    "roc": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.9, 1.0], "auc": 0.91},
                    "pr": {"recall": [0.0, 0.9, 1.0], "precision": [1.0, 0.9, 0.43], "ap": 0.88},
                    "calibration": {"mean_probability": [0.1, 0.5, 0.9], "positive_rate": [0.1, 0.5, 0.9]},
                    "brier": 0.10,
                },
            },
            "optimization": {
                "smoothing": {"values": [1e-9, 1e-5], "scores": [0.72, 0.84], "best": 1e-5},
                "priors": {"values": [0.3, 0.5, 0.7], "precision": [0.90, 0.82, 0.70], "recall": [0.60, 0.78, 0.91], "best": 0.5},
                "stability": {"dimensions": [10, 100, 500], "base_finite": [1.0, 1.0, 0.0], "optimized_finite": [1.0, 1.0, 1.0], "underflow_dimension": 500},
                "incremental": {"samples_seen": [50, 100, 200], "balanced_accuracy": [0.70, 0.78, 0.84], "batch_balanced_accuracy": 0.85},
            },
        }

    def test_metrics_and_probability_curves_match_hand_checked_values(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8]

        metrics = classification_metrics(y_true, y_pred)
        curves = probability_curves(y_true, scores, n_bins=2)

        self.assertEqual(metrics["confusion"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(curves["roc"]["auc"], 0.75)
        self.assertAlmostEqual(curves["pr"]["ap"], 5 / 6)
        self.assertAlmostEqual(curves["brier"], 0.158125)

    def test_log_domain_remains_finite_after_direct_product_underflows(self):
        stability = _stability_report()

        self.assertIn(0.0, stability["base_finite"])
        self.assertEqual(stability["optimized_finite"], [1.0] * len(stability["dimensions"]))

    def test_script_can_run_directly_from_the_command_line(self):
        script = PROJECT_ROOT / "visualization" / "gaussian_naive_bayes_figures.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_renderer_exports_exactly_six_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = render_six_figures(self._sample_report(), output_dir)
            files = [path for path in output_dir.iterdir() if path.is_file()]

            self.assertEqual(len(paths), 6)
            self.assertEqual(len(files), 6)
            self.assertTrue(all(path.suffix == ".png" for path in files))

    def test_renderer_keeps_model_comparisons_in_shared_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = render_six_figures(self._sample_report(), Path(directory))

            self.assertIn("04_benchmark_performance.png", {path.name for path in paths})
            self.assertIn("05_probability_diagnostics.png", {path.name for path in paths})
            self.assertIn("06_optimization_analysis.png", {path.name for path in paths})

    def test_renderer_finishes_without_layout_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                render_six_figures(self._sample_report(), Path(directory))

            layout_warnings = [
                warning
                for warning in caught
                if "layout" in str(warning.message).lower()
            ]
            self.assertEqual(layout_warnings, [])

    def test_uses_times_new_roman_and_six_distinct_palettes(self):
        self.assertEqual(plt.rcParams["font.family"], ["serif"])
        self.assertEqual(plt.rcParams["font.serif"][0], "Times New Roman")
        self.assertEqual(len(FIGURE_PALETTES), 6)
        self.assertEqual(len({tuple(colors) for colors in FIGURE_PALETTES.values()}), 6)


if __name__ == "__main__":
    unittest.main()
