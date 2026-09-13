import csv
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.logistic_regression_figures import (
    classification_metrics,
    precision_recall_curve,
    render_six_figures,
    roc_curve,
)


class LogisticRegressionVisualizationTests(unittest.TestCase):
    def test_metrics_and_curves_match_a_hand_checked_example(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8]

        metrics = classification_metrics(y_true, y_pred)
        _, _, auc = roc_curve(y_true, scores)
        _, _, average_precision = precision_recall_curve(y_true, scores)

        self.assertEqual(metrics["confusion"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(auc, 0.75)
        self.assertAlmostEqual(average_precision, 5 / 6)

    def _sample_report(self):
        return {
            "class_counts": {"Benign": 3, "Malignant": 1},
            "loss_series": {
                "Base configuration": [0.69, 0.62, 0.58],
                "Full optimized": [0.69, 0.51, 0.42],
            },
            "early_stop_iteration": 3,
            "confusion": [[2, 1], [0, 1]],
            "roc": {"fpr": [0.0, 0.0, 1.0], "tpr": [0.0, 1.0, 1.0], "auc": 1.0},
            "pr": {"recall": [0.0, 1.0], "precision": [1.0, 1.0], "ap": 1.0},
            "pr_comparison": {
                "Base configuration": {
                    "recall": [0.0, 0.5, 1.0],
                    "precision": [1.0, 0.5, 0.5],
                    "ap": 0.5,
                },
                "Full optimized": {
                    "recall": [0.0, 1.0],
                    "precision": [1.0, 1.0],
                    "ap": 1.0,
                },
            },
            "ablation": [
                {"name": "Base", "accuracy": 0.75, "f1": 0.67},
                {"name": "Full", "accuracy": 1.0, "f1": 1.0},
            ],
        }

    def test_default_renderer_exports_only_six_png_files(self):
        report = self._sample_report()

        with tempfile.TemporaryDirectory() as directory:
            paths = render_six_figures(report, Path(directory))
            graphic_files = [
                path
                for path in Path(directory).iterdir()
                if path.suffix in {".png", ".svg", ".pdf"}
            ]

            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.suffix == ".png" for path in paths))
            self.assertEqual(len(graphic_files), 6)
            self.assertTrue(all(path.suffix == ".png" for path in graphic_files))

    def test_renderer_keeps_both_pr_models_in_one_figure_source_table(self):
        report = self._sample_report()

        with tempfile.TemporaryDirectory() as directory:
            render_six_figures(report, Path(directory))
            source_path = Path(directory) / "source_data" / "05_precision_recall_curve.csv"
            with source_path.open("r", encoding="utf-8-sig", newline="") as file:
                series = {row["series"] for row in csv.DictReader(file)}

            self.assertEqual(series, {"Base configuration", "Full optimized"})

    def test_publication_font_is_times_new_roman(self):
        self.assertEqual(plt.rcParams["font.family"], ["serif"])
        self.assertEqual(plt.rcParams["font.serif"][0], "Times New Roman")

    def test_renderer_exports_six_source_tables(self):
        report = self._sample_report()

        with tempfile.TemporaryDirectory() as directory:
            render_six_figures(report, Path(directory))
            csv_paths = sorted((Path(directory) / "source_data").glob("*.csv"))

            self.assertEqual(len(csv_paths), 6)


if __name__ == "__main__":
    unittest.main()
