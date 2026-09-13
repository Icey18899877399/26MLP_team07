import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.knn_figures import (
    FIGURE_PALETTES,
    mean_and_std,
    render_six_figures,
    select_k_full_sort,
    select_k_heap,
    stratified_k_folds,
)


class KNNVisualizationTests(unittest.TestCase):
    def _sample_report(self):
        return {
            "metadata": {"folds": 2, "seed": 42},
            "neighborhood": {
                "train_x": [[-1.0, -1.0], [0.0, 0.5], [1.0, 1.0]],
                "train_y": [0, 0, 1],
                "query": [0.2, 0.3],
                "query_label": 0,
                "predicted_label": 0,
                "neighbors": [
                    {"index": 1, "distance": 0.28, "weight": 3.54, "label": 0},
                    {"index": 2, "distance": 1.06, "weight": 0.94, "label": 1},
                ],
                "feature_names": ["Feature A (standardized)", "Feature B (standardized)"],
            },
            "boundaries": {
                "x_values": [-1.0, 1.0],
                "y_values": [-1.0, 1.0],
                "train_x": [[-1.0, -1.0], [1.0, 1.0]],
                "train_y": [0, 1],
                "feature_names": ["Feature A (standardized)", "Feature B (standardized)"],
                "panels": [
                    {"k": 1, "predictions": [[0, 1], [0, 1]]},
                    {"k": 5, "predictions": [[0, 0], [1, 1]]},
                    {"k": 15, "predictions": [[0, 0], [0, 1]]},
                ],
            },
            "validation": {
                "k_values": [1, 3],
                "series": {
                    "Base accuracy": {"mean": [0.80, 0.85], "std": [0.02, 0.01]},
                    "Base F1": {"mean": [0.76, 0.82], "std": [0.03, 0.02]},
                    "Optimized accuracy": {"mean": [0.88, 0.91], "std": [0.02, 0.01]},
                    "Optimized F1": {"mean": [0.86, 0.90], "std": [0.02, 0.01]},
                },
                "best_k": 3,
            },
            "diagnostics": {
                "Base": {
                    "roc": {"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.80},
                    "pr": {"recall": [0.0, 0.8, 1.0], "precision": [1.0, 0.75, 0.5], "ap": 0.75},
                },
                "Optimized": {
                    "roc": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.9, 1.0], "auc": 0.90},
                    "pr": {"recall": [0.0, 0.9, 1.0], "precision": [1.0, 0.9, 0.5], "ap": 0.88},
                },
            },
            "confusion": {
                "Base": [[8, 2], [2, 6]],
                "Optimized": [[9, 1], [1, 7]],
            },
            "optimization": [
                {"name": "Base", "accuracy": 0.80, "f1": 0.76},
                {"name": "+ Scale", "accuracy": 0.88, "f1": 0.86},
                {"name": "Full", "accuracy": 0.91, "f1": 0.90},
            ],
            "runtime": {
                "train_sizes": [100, 500],
                "full_sort_ms": [0.3, 1.8],
                "heap_ms": [0.2, 0.9],
                "selections": 20,
                "repeats": 3,
            },
        }

    def test_stratified_folds_cover_every_sample_once(self):
        targets = [0, 0, 0, 0, 1, 1, 1, 1]

        folds = stratified_k_folds(targets, n_splits=2, seed=7)

        self.assertEqual(sorted(index for fold in folds for index in fold), list(range(8)))
        self.assertEqual([sorted(targets[index] for index in fold) for fold in folds], [[0, 0, 1, 1]] * 2)

    def test_mean_and_std_matches_a_hand_checked_sample(self):
        mean, std = mean_and_std([1.0, 2.0, 3.0])

        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, math.sqrt(2.0 / 3.0))

    def test_full_sort_and_heap_select_the_same_k_distances(self):
        distances = [4.0, 1.0, 3.0, 2.0, 0.5]

        self.assertEqual(select_k_full_sort(distances, 3), [0.5, 1.0, 2.0])
        self.assertEqual(select_k_heap(distances, 3), [0.5, 1.0, 2.0])

    def test_renderer_exports_six_png_files_and_six_source_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = render_six_figures(self._sample_report(), output_dir)
            graphics = [path for path in output_dir.iterdir() if path.suffix in {".png", ".svg", ".pdf"}]
            tables = list((output_dir / "source_data").glob("*.csv"))

            self.assertEqual(len(paths), 6)
            self.assertEqual(len(graphics), 6)
            self.assertTrue(all(path.suffix == ".png" for path in graphics))
            self.assertEqual(len(tables), 6)

    def test_comparison_source_tables_contain_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            render_six_figures(self._sample_report(), output_dir)

            with (output_dir / "source_data" / "04_roc_pr_comparison.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                models = {row["model"] for row in csv.DictReader(file)}

            self.assertEqual(models, {"Base", "Optimized"})

    def test_uses_times_new_roman_and_six_distinct_palettes(self):
        self.assertEqual(plt.rcParams["font.family"], ["serif"])
        self.assertEqual(plt.rcParams["font.serif"][0], "Times New Roman")
        self.assertEqual(len(FIGURE_PALETTES), 6)
        self.assertEqual(len({tuple(colors) for colors in FIGURE_PALETTES.values()}), 6)


if __name__ == "__main__":
    unittest.main()
