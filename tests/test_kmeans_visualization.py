import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.kmeans_figures import (
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    adjusted_rand_index,
    normalized_mutual_information,
    purity_score,
    render_six_figures,
    run_experiment,
    silhouette_score,
    standardize_features,
)


class KMeansVisualizationTests(unittest.TestCase):
    def _sample_report(self):
        return {
            "metadata": {
                "sample_count": 6,
                "feature_count": 2,
                "n_clusters": 3,
                "seed": 42,
                "stability_runs": 3,
                "benchmark_runs": 3,
            },
            "cluster_map": {
                "x": [1.0, 1.2, 5.0, 5.2, 9.0, 9.2],
                "y": [1.0, 1.1, 5.0, 5.1, 1.0, 1.1],
                "labels": [0, 0, 1, 1, 2, 2],
                "centers": [[1.1, 1.05], [5.1, 5.05], [9.1, 1.05]],
                "feature_names": ["Area", "Kernel groove length"],
                "inertia": 0.12,
                "silhouette": 0.91,
            },
            "trajectory": {
                "x": [1.0, 1.2, 5.0, 5.2, 9.0, 9.2],
                "y": [1.0, 1.1, 5.0, 5.1, 1.0, 1.1],
                "labels": [0, 0, 1, 1, 2, 2],
                "feature_names": ["Area", "Kernel groove length"],
                "paths": [
                    [[1.5, 1.4], [1.2, 1.1], [1.1, 1.05]],
                    [[4.5, 4.2], [5.0, 4.9], [5.1, 5.05]],
                    [[8.5, 1.5], [9.0, 1.2], [9.1, 1.05]],
                ],
            },
            "convergence": {
                "base_history": [18.0, 8.0, 5.0],
                "optimized_history": [11.0, 4.5, 4.0],
                "base_label": "Random initialization (1 start)",
                "optimized_label": "K-Means++ (1 start)",
            },
            "selection": {
                "k_values": [2, 3, 4],
                "inertia": [12.0, 5.0, 4.2],
                "silhouette": [0.61, 0.82, 0.70],
                "selected_k": 3,
                "reference_k": 3,
            },
            "stability": {
                "methods": ["Random", "K-Means++", "K-Means++ (n_init=10)"],
                "values": {
                    "Random": [7.0, 6.0, 8.0],
                    "K-Means++": [5.4, 5.1, 5.2],
                    "K-Means++ (n_init=10)": [4.9, 4.8, 4.9],
                },
            },
            "benchmark": {
                "methods": ["Base K-Means", "Optimized K-Means"],
                "runs": {
                    "Base K-Means": [
                        {"seed": 0, "silhouette": 0.70, "purity": 0.82, "ari": 0.72, "nmi": 0.75, "fit_ms": 0.8},
                        {"seed": 1, "silhouette": 0.72, "purity": 0.84, "ari": 0.74, "nmi": 0.77, "fit_ms": 0.9},
                    ],
                    "Optimized K-Means": [
                        {"seed": 0, "silhouette": 0.81, "purity": 0.91, "ari": 0.86, "nmi": 0.88, "fit_ms": 2.0},
                        {"seed": 1, "silhouette": 0.82, "purity": 0.92, "ari": 0.87, "nmi": 0.89, "fit_ms": 2.1},
                    ],
                },
            },
        }

    def test_standardization_has_zero_mean_and_unit_population_variance(self):
        scaled, means, scales = standardize_features([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        self.assertEqual(means, [2.0, 20.0])
        self.assertAlmostEqual(scales[0], math.sqrt(2.0 / 3.0))
        for feature in range(2):
            column = [row[feature] for row in scaled]
            self.assertAlmostEqual(sum(column) / 3.0, 0.0)
            self.assertAlmostEqual(sum(value * value for value in column) / 3.0, 1.0)

    def test_external_cluster_scores_equal_one_for_a_perfect_partition(self):
        truth = [1, 1, 2, 2, 3, 3]
        predicted = [2, 2, 0, 0, 1, 1]

        self.assertEqual(purity_score(truth, predicted), 1.0)
        self.assertAlmostEqual(adjusted_rand_index(truth, predicted), 1.0)
        self.assertAlmostEqual(normalized_mutual_information(truth, predicted), 1.0)

    def test_degenerate_partition_metrics_follow_standard_boundary_values(self):
        self.assertAlmostEqual(adjusted_rand_index([0, 1, 2], [4, 5, 6]), 1.0)
        self.assertAlmostEqual(normalized_mutual_information([0, 0, 0], [2, 3, 4]), 0.0)
        self.assertAlmostEqual(normalized_mutual_information([0, 0, 0], [7, 7, 7]), 1.0)

    def test_silhouette_matches_a_hand_checked_two_cluster_example(self):
        score = silhouette_score([[0.0], [2.0], [10.0], [12.0]], [0, 0, 1, 1])

        self.assertAlmostEqual(score, 0.797979797979798)

    def test_renderer_exports_six_png_files_and_six_source_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = render_six_figures(self._sample_report(), output_dir)
            graphics = [path for path in output_dir.iterdir() if path.suffix.lower() in {".png", ".svg", ".pdf", ".tiff"}]
            tables = list((output_dir / "source_data").glob("*.csv"))

            self.assertEqual([path.name for path in paths], list(FIGURE_FILENAMES))
            self.assertEqual(len(graphics), 6)
            self.assertTrue(all(path.suffix.lower() == ".png" for path in graphics))
            self.assertEqual(len(tables), 6)

    def test_benchmark_source_table_contains_both_models_and_all_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            render_six_figures(self._sample_report(), output_dir)

            with (output_dir / "source_data" / "06_benchmark_comparison.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as source:
                rows = list(csv.DictReader(source))

            self.assertEqual({row["model"] for row in rows}, {"Base K-Means", "Optimized K-Means"})
            self.assertEqual(
                set(rows[0]),
                {"model", "seed", "silhouette", "purity", "ari", "nmi", "fit_ms"},
            )

    def test_source_tables_contain_every_plotted_value_and_scale_description(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            report = self._sample_report()
            render_six_figures(report, output_dir)

            with (output_dir / "source_data" / "01_cluster_distribution.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as source:
                cluster_rows = list(csv.DictReader(source))
            with (output_dir / "source_data" / "02_centroid_trajectory.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as source:
                trajectory_rows = list(csv.DictReader(source))

            self.assertTrue(all(row["clustering_space"] == "7-feature standardized" for row in cluster_rows))
            self.assertEqual(float(cluster_rows[0]["sse"]), report["cluster_map"]["inertia"])
            self.assertEqual(float(cluster_rows[0]["silhouette"]), report["cluster_map"]["silhouette"])
            self.assertEqual({row["type"] for row in trajectory_rows}, {"sample", "centroid"})
            self.assertTrue(all(row["display_space"] == "2-feature original scale" for row in trajectory_rows))

    def test_explicit_cluster_count_does_not_depend_on_reference_label_cardinality(self):
        rows = []
        for center in (0.0, 5.0, 10.0):
            for offset in (0.0, 0.1, 0.2, 0.3):
                features = [center + offset + 0.01 * index for index in range(7)]
                rows.append("\t".join(str(value) for value in features + [1]))
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "seeds_like.txt"
            data_path.write_text("\n".join(rows), encoding="utf-8")

            report = run_experiment(
                data_path,
                n_clusters=3,
                k_values=(2, 3),
                stability_runs=1,
                benchmark_runs=1,
            )

        self.assertEqual(report["metadata"]["n_clusters"], 3)
        self.assertEqual(report["metadata"]["reference_classes"], 1)
        self.assertEqual(len(report["cluster_map"]["centers"]), 3)

    def test_uses_times_new_roman_and_six_distinct_palettes(self):
        self.assertEqual(plt.rcParams["font.family"], ["serif"])
        self.assertEqual(plt.rcParams["font.serif"][0], "Times New Roman")
        self.assertEqual(len(FIGURE_PALETTES), 6)
        self.assertEqual(len({tuple(colors) for colors in FIGURE_PALETTES.values()}), 6)


if __name__ == "__main__":
    unittest.main()
