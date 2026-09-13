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

from visualization.random_forest_figures import (  # noqa: E402
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    PLOT_STYLE,
    RANK_CMAP_COLORS,
    classification_metrics,
    pairwise_disagreement,
    probability_curves,
    render_six_figures,
)
from visualization import random_forest_figures as rf_figures  # noqa: E402


def _sample_report():
    return {
        "bootstrap": {
            "membership": [
                [2, 1, 0, 1, 0, 2, 1, 0],
                [0, 1, 2, 0, 1, 1, 0, 2],
                [1, 0, 1, 2, 1, 0, 2, 1],
                [2, 1, 0, 1, 2, 1, 0, 1],
            ],
            "sample_labels": ["B", "B", "M", "B", "M", "M", "B", "M"],
            "oob_counts": [1, 2, 1, 2, 1, 1, 2, 1],
            "example_tree_probabilities": [0.08, 0.72, 0.83, 0.67, 0.22, 0.91, 0.76, 0.88],
            "example_probability": 0.634,
            "example_true": 1,
            "example_scope": "CV validation",
        },
        "convergence": {
            "tree_counts": [1, 5, 10, 20, 40],
            "cv_mean": [0.78, 0.84, 0.87, 0.89, 0.90],
            "cv_std": [0.08, 0.04, 0.025, 0.018, 0.015],
            "oob_mean": [0.76, 0.83, 0.86, 0.885, 0.895],
            "oob_std": [0.09, 0.05, 0.03, 0.02, 0.016],
            "coverage_mean": [0.36, 0.88, 0.98, 1.0, 1.0],
            "selected_count": 20,
        },
        "hyperparameters": {
            "depth_labels": ["2", "4", "6", "None"],
            "feature_labels": ["25%", "sqrt", "all"],
            "score_matrix": [
                [0.84, 0.86, 0.85],
                [0.88, 0.91, 0.90],
                [0.87, 0.90, 0.89],
                [0.85, 0.88, 0.87],
            ],
            "leaf_matrix": [[3, 4, 4], [8, 10, 11], [13, 16, 19], [20, 25, 31]],
            "selected": [1, 1],
        },
        "diversity": {
            "disagreement": [0.18, 0.23, 0.31, 0.27, 0.35, 0.21],
            "error_correlation": [0.22, 0.15, 0.05, 0.11, -0.02, 0.18],
            "individual_scores": [0.78, 0.81, 0.76, 0.84, 0.80, 0.79, 0.83, 0.77],
            "ensemble_score": 0.90,
            "tree_counts": [1, 2, 3, 4, 5, 6, 7, 8],
            "cumulative_scores": [0.78, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90],
        },
        "importance": {
            "feature_names": ["Worst perimeter", "Worst concave points", "Worst radius", "Mean texture"],
            "Base": {"mean": [0.35, 0.27, 0.20, 0.10], "std": [0.08, 0.07, 0.05, 0.03]},
            "Optimized": {"mean": [0.32, 0.30, 0.19, 0.11], "std": [0.05, 0.04, 0.04, 0.02]},
            "rank_matrix": [[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 4, 3], [1, 2, 3, 4], [2, 1, 3, 4]],
        },
        "benchmark": {
            "Base": {
                "confusion": [[66, 5], [4, 38]],
                "metrics": {"balanced_accuracy": 0.917, "f1": 0.894, "auc": 0.965, "ap": 0.951},
                "roc": {"x": [0, 0.08, 0.2, 1], "y": [0, 0.83, 0.95, 1]},
                "pr": {"x": [0, 0.75, 1], "y": [1, 0.94, 0.37]},
            },
            "Optimized": {
                "confusion": [[68, 3], [3, 39]],
                "metrics": {"balanced_accuracy": 0.943, "f1": 0.929, "auc": 0.978, "ap": 0.970},
                "roc": {"x": [0, 0.04, 0.13, 1], "y": [0, 0.88, 0.98, 1]},
                "pr": {"x": [0, 0.8, 1], "y": [1, 0.97, 0.37]},
            },
            "runtime": {
                "sizes": [80, 160, 320],
                "base_ms": [40, 130, 520],
                "optimized_serial_ms": [22, 65, 205],
                "optimized_parallel_ms": [35, 48, 115],
            },
        },
    }


class RandomForestVisualizationCalculationTests(unittest.TestCase):
    def test_metrics_and_probability_curves_match_hand_calculation(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.10, 0.40, 0.35, 0.80]

        metrics = classification_metrics(y_true, y_pred)
        curves = probability_curves(y_true, scores)

        self.assertEqual(metrics["confusion"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(curves["roc"]["auc"], 0.75)
        self.assertAlmostEqual(curves["pr"]["ap"], 5 / 6)

    def test_pairwise_disagreement_is_fraction_of_different_predictions(self):
        self.assertAlmostEqual(
            pairwise_disagreement([0, 1, 1, 0], [0, 0, 1, 1]),
            0.5,
        )

    def test_probability_curves_handle_tied_scores_without_row_order_bias(self):
        first = probability_curves([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.2])
        second = probability_curves([0, 1, 0, 1], [0.8, 0.8, 0.2, 0.2])

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["roc"]["auc"], 0.5)
        self.assertAlmostEqual(first["pr"]["ap"], 0.5)


class RandomForestVisualizationContractTests(unittest.TestCase):
    def test_dense_sample_labels_are_limited_to_twelve_ticks(self):
        positions = rf_figures._sample_tick_positions(36, maximum_ticks=12)
        self.assertLessEqual(len(positions), 12)
        self.assertEqual(positions[0], 0)

    def test_runtime_axis_uses_log_scale_and_names_startup_cost(self):
        figure, axis = plt.subplots()
        rf_figures._configure_runtime_axis(axis)
        self.assertEqual(axis.get_yscale(), "log")
        self.assertIn("startup", axis.get_ylabel().lower())
        self.assertIn("synthetic", axis.get_title().lower())
        self.assertIn("12 features", axis.get_xlabel().lower())
        plt.close(figure)

    def test_rank_heatmap_uses_a_colorblind_safe_sequential_scale(self):
        self.assertEqual(RANK_CMAP_COLORS, ("#F7FBFF", "#6BAED6", "#08306B"))

    def test_has_six_distinct_optimized_palettes_and_times_font(self):
        self.assertEqual(len(FIGURE_PALETTES), 6)
        self.assertEqual(len({tuple(colors) for colors in FIGURE_PALETTES.values()}), 6)
        self.assertEqual(PLOT_STYLE["font.family"], "serif")
        self.assertEqual(PLOT_STYLE["font.serif"][0], "Times New Roman")

    def test_renderer_creates_only_the_six_png_files_without_layout_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                paths = render_six_figures(_sample_report(), output_dir)

            self.assertEqual({path.name for path in paths}, set(FIGURE_FILENAMES))
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(FIGURE_FILENAMES))
            self.assertTrue(all(path.suffix.lower() == ".png" for path in paths))
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))
            warning_text = " ".join(str(item.message).lower() for item in captured)
            self.assertNotIn("layout", warning_text)

    def test_renderer_preserves_unrelated_files_in_a_custom_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            unrelated = output_dir / "notes.txt"
            unrelated.write_text("keep me", encoding="utf-8")

            render_six_figures(_sample_report(), output_dir)

            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")

    def test_comparisons_remain_in_single_named_images(self):
        expected = {
            "03_hyperparameter_response.png",
            "04_tree_diversity_and_ensemble_gain.png",
            "05_feature_importance_stability.png",
            "06_final_benchmark_comparison.png",
        }
        self.assertTrue(expected.issubset(set(FIGURE_FILENAMES)))

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "random_forest_figures.py"
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
