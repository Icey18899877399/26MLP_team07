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

from visualization.cart_decision_tree_figures import (  # noqa: E402
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    _tree_dict,
    classification_metrics,
    probability_curves,
    render_six_figures,
    split_gain_curve,
)
from visualization import cart_decision_tree_figures as cart_figures  # noqa: E402


def _leaf(prediction, probabilities, impurity, sample_count):
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "impurity": impurity,
        "sample_count": sample_count,
        "feature_index": None,
        "threshold": None,
        "left": None,
        "right": None,
    }


def _sample_report():
    base_tree = {
        "prediction": 0,
        "probabilities": [0.5, 0.5],
        "impurity": 0.5,
        "sample_count": 4,
        "feature_index": 0,
        "threshold": 1.5,
        "left": _leaf(0, [1.0, 0.0], 0.0, 2),
        "right": _leaf(1, [0.0, 1.0], 0.0, 2),
    }
    optimized_tree = {
        **base_tree,
        "threshold": 1.4,
        "probabilities": [0.55, 0.45],
    }
    return {
        "split_gain": {
            "feature_name": "Feature A",
            "values": {"Benign": [0.0, 1.0], "Malignant": [2.0, 3.0]},
            "thresholds": [0.5, 1.5, 2.5],
            "gains": [1.0 / 6.0, 0.5, 1.0 / 6.0],
            "selected_threshold": 1.5,
        },
        "trees": {
            "Base": base_tree,
            "Optimized": optimized_tree,
            "metadata": {
                "Base": {"depth": 1, "leaves": 2},
                "Optimized": {"depth": 1, "leaves": 2},
            },
            "feature_names": ["Feature A", "Feature B"],
        },
        "boundaries": {
            "x_values": [0.0, 0.5, 1.0],
            "y_values": [0.0, 0.5, 1.0],
            "Base": [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            "Optimized": [[0, 0, 0], [0, 0, 1], [0, 1, 1]],
            "train_x": [[0.1, 0.2], [0.3, 0.4], [0.8, 0.7], [0.9, 0.9]],
            "train_y": [0, 0, 1, 1],
            "feature_names": ["Feature A", "Feature B"],
        },
        "pruning": {
            "alphas": [0.0, 0.01, 0.1],
            "score_mean": [0.80, 0.85, 0.75],
            "score_std": [0.02, 0.01, 0.03],
            "leaves_mean": [8.0, 4.0, 1.0],
            "depth_mean": [5.0, 3.0, 0.0],
            "best_alpha": 0.01,
        },
        "diagnostics": {
            "Base": {
                "confusion": [[6, 2], [2, 4]],
                "balanced_accuracy": 0.708,
                "roc": {"x": [0.0, 0.2, 1.0], "y": [0.0, 0.8, 1.0], "auc": 0.82},
                "pr": {"x": [0.0, 0.7, 1.0], "y": [1.0, 0.8, 0.4], "ap": 0.78},
            },
            "Optimized": {
                "confusion": [[7, 1], [1, 5]],
                "balanced_accuracy": 0.854,
                "roc": {"x": [0.0, 0.1, 1.0], "y": [0.0, 0.9, 1.0], "auc": 0.91},
                "pr": {"x": [0.0, 0.8, 1.0], "y": [1.0, 0.9, 0.4], "ap": 0.88},
            },
        },
        "optimization": {
            "runtime": {
                "sizes": [50, 100, 200],
                "base_ms": [1.0, 4.0, 16.0],
                "optimized_ms": [0.8, 1.8, 5.0],
            },
            "ablation": [
                {"name": "Base config", "balanced_accuracy": 0.78, "leaves": 12.0},
                {"name": "+ Fast scan", "balanced_accuracy": 0.78, "leaves": 12.0},
                {"name": "+ Pre-pruning", "balanced_accuracy": 0.82, "leaves": 7.0},
                {"name": "+ Class weights", "balanced_accuracy": 0.84, "leaves": 7.0},
                {"name": "+ CCP pruning", "balanced_accuracy": 0.85, "leaves": 4.0},
            ],
            "importance": {
                "feature_names": ["Feature A", "Feature B", "Feature C"],
                "Base": {"mean": [0.60, 0.30, 0.10], "std": [0.10, 0.05, 0.02]},
                "Optimized": {"mean": [0.70, 0.20, 0.10], "std": [0.05, 0.03, 0.02]},
            },
        },
    }


class TestCartVisualizationCalculations(unittest.TestCase):
    def test_tree_export_reorders_probabilities_by_binary_label(self):
        class Node:
            prediction = 1
            probabilities = [0.8, 0.2]
            impurity = 0.32
            sample_count = 10
            feature_index = None
            threshold = None
            left = None
            right = None

        exported = _tree_dict(Node(), classes=[1, 0])
        self.assertEqual(exported["probabilities"], [0.2, 0.8])

    def test_split_gain_curve_matches_hand_calculation(self):
        result = split_gain_curve([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1])
        self.assertEqual(result["thresholds"], [0.5, 1.5, 2.5])
        for actual, expected in zip(result["gains"], [1 / 6, 0.5, 1 / 6]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["selected_threshold"], 1.5)

    def test_metrics_and_curves_match_hand_calculation(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.10, 0.40, 0.35, 0.80]
        metrics = classification_metrics(y_true, y_pred)
        curves = probability_curves(y_true, scores)
        self.assertEqual(metrics["confusion"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(curves["roc"]["auc"], 0.75)
        self.assertAlmostEqual(curves["pr"]["ap"], 5.0 / 6.0)


class TestCartVisualizationContract(unittest.TestCase):
    def test_threshold_annotation_uses_axes_coordinates(self):
        figure, axis = plt.subplots()
        annotation = cart_figures._annotate_selected_threshold(axis, 104.0, "#123456")
        self.assertIs(annotation.get_transform(), axis.transAxes)
        self.assertGreaterEqual(annotation.get_position()[1], 0.0)
        self.assertLessEqual(annotation.get_position()[1], 1.0)
        plt.close(figure)

    def test_six_distinct_palettes_and_times_font(self):
        self.assertEqual(len(FIGURE_PALETTES), 6)
        self.assertEqual(len({tuple(colors) for colors in FIGURE_PALETTES.values()}), 6)
        self.assertEqual(cart_figures.PLOT_STYLE["font.family"], "serif")
        self.assertEqual(cart_figures.PLOT_STYLE["font.serif"][0], "Times New Roman")

    def test_renderer_creates_exactly_six_png_files_without_layout_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                paths = render_six_figures(_sample_report(), output_dir)
            self.assertEqual({path.name for path in paths}, set(FIGURE_FILENAMES))
            self.assertEqual(
                {path.name for path in output_dir.iterdir() if path.is_file()},
                set(FIGURE_FILENAMES),
            )
            self.assertTrue(all(path.suffix.lower() == ".png" for path in paths))
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))
            warning_text = " ".join(str(item.message).lower() for item in captured)
            self.assertNotIn("layout", warning_text)

    def test_comparisons_are_kept_in_their_named_single_images(self):
        expected = {
            "02_tree_structure_comparison.png",
            "03_decision_boundary_comparison.png",
            "05_test_set_diagnostics.png",
            "06_optimization_evidence.png",
        }
        self.assertTrue(expected.issubset(set(FIGURE_FILENAMES)))

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "cart_decision_tree_figures.py"
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
