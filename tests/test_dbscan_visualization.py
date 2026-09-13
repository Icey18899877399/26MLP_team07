import subprocess
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

import matplotlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.dbscan_figures import (
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    adjusted_rand_index,
    coverage_score,
    find_knee,
    generate_two_moons,
    k_distances,
    normalized_mutual_information,
    render_six_figures,
    run_experiment,
    select_parameters,
    silhouette_score,
    _plot_benchmark,
    _plot_standardization,
)


DATA_PATH = PROJECT_ROOT / "data" / "clustering" / "seeds" / "seeds_dataset.txt"


class DBSCANVisualizationCalculationTests(unittest.TestCase):
    def test_two_moons_is_reproducible_and_contains_requested_noise(self):
        first_X, first_y = generate_two_moons(
            samples_per_moon=5,
            noise_samples=3,
            jitter=0.02,
            random_state=8,
        )
        second_X, second_y = generate_two_moons(
            samples_per_moon=5,
            noise_samples=3,
            jitter=0.02,
            random_state=8,
        )

        self.assertEqual(first_X, second_X)
        self.assertEqual(first_y, second_y)
        self.assertEqual(Counter(first_y), Counter({0: 5, 1: 5, -1: 3}))

    def test_k_distance_counts_the_sample_itself_like_dbscan(self):
        distances = k_distances([[0.0], [1.0], [3.0]], min_samples=2)

        self.assertEqual(distances, [1.0, 1.0, 2.0])

    def test_knee_finds_the_largest_departure_from_the_endpoint_line(self):
        index, value = find_knee([0.1, 0.2, 0.3, 1.0])

        self.assertEqual(index, 2)
        self.assertAlmostEqual(value, 0.3)

    def test_external_metrics_equal_one_for_a_perfect_partition(self):
        truth = [0, 0, 1, 1, -1]
        predicted = [4, 4, 7, 7, -1]

        self.assertAlmostEqual(adjusted_rand_index(truth, predicted), 1.0)
        self.assertAlmostEqual(normalized_mutual_information(truth, predicted), 1.0)

    def test_silhouette_excludes_noise_and_coverage_reports_it(self):
        X = [[0.0], [1.0], [4.0], [5.0], [100.0]]
        labels = [0, 0, 1, 1, -1]

        self.assertAlmostEqual(silhouette_score(X, labels), 0.746031746031746)
        self.assertAlmostEqual(coverage_score(labels), 0.8)

    def test_parameter_selection_uses_only_silhouette_times_coverage(self):
        selected = select_parameters(
            eps_values=[0.4, 0.6],
            min_samples_values=[5],
            cluster_counts=[[2, 3]],
            silhouette_scores=[[0.5, 0.5]],
            noise_rates=[[0.2, 0.2]],
        )

        self.assertEqual(selected, (0.4, 5))


class DBSCANVisualizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_experiment(
            DATA_PATH,
            samples_per_moon=24,
            noise_samples=6,
            min_samples=4,
            eps_values=(0.35, 0.55),
            min_samples_values=(3, 5),
            benchmark_sizes=(40, 80),
            benchmark_runs=1,
            random_state=12,
        )

    def test_reduced_experiment_contains_every_evidence_layer(self):
        self.assertEqual(
            set(self.report),
            {
                "metadata",
                "synthetic",
                "mechanism",
                "k_distance",
                "parameter_grid",
                "scaling",
                "benchmark",
            },
        )
        self.assertEqual(len(self.report["parameter_grid"]["cluster_counts"]), 2)
        self.assertEqual(len(self.report["parameter_grid"]["cluster_counts"][0]), 2)

    def test_unsupervised_selection_chooses_a_non_degenerate_grid_candidate(self):
        grid = self.report["parameter_grid"]
        scaling = self.report["scaling"]

        self.assertIn(grid["selected_eps"], grid["eps_values"])
        self.assertIn(grid["selected_min_samples"], grid["min_samples_values"])
        self.assertGreaterEqual(scaling["standardized_cluster_count"], 2)

    def test_scaling_panel_discloses_workflow_and_display_coordinates(self):
        scaling = self.report["scaling"]

        self.assertEqual(scaling["comparison"], "raw_knee_vs_scaled_tuned")
        self.assertEqual(scaling["display_coordinates"], "original")

    def test_scaling_panel_titles_do_not_overlap(self):
        figure = _plot_standardization(self.report)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        left_box = figure.axes[0].title.get_window_extent(renderer)
        right_box = figure.axes[1].title.get_window_extent(renderer)

        self.assertLess(left_box.x1, right_box.x0)

    def test_benchmark_title_discloses_scaling_and_tuning(self):
        figure = _plot_benchmark(self.report)

        self.assertIn("Scaling/tuning", figure._suptitle.get_text())
        self.assertIn("KD-tree", figure._suptitle.get_text())

    def test_runtime_benchmark_holds_cluster_semantics_constant(self):
        benchmark = self.report["benchmark"]

        self.assertEqual(benchmark["sizes"], [40, 80])
        self.assertEqual(benchmark["label_agreement"], [True, True])
        self.assertTrue(all(value >= 0.0 for value in benchmark["basic_seconds"]))
        self.assertTrue(all(value >= 0.0 for value in benchmark["kd_tree_seconds"]))
        self.assertEqual(len(benchmark["basic_std_seconds"]), 2)
        self.assertEqual(len(benchmark["kd_tree_std_seconds"]), 2)
        self.assertTrue(all(value >= 0.0 for value in benchmark["basic_std_seconds"]))
        self.assertTrue(all(value >= 0.0 for value in benchmark["kd_tree_std_seconds"]))

    def test_quality_summary_reports_accuracy_structure_and_coverage(self):
        quality = self.report["benchmark"]["quality"]

        self.assertEqual(
            quality["methods"],
            ["Basic DBSCAN (raw + knee)", "Optimized DBSCAN (scaled + tuned)"],
        )
        self.assertEqual(
            set(quality["metrics"]),
            {"ARI", "NMI", "Silhouette", "Coverage"},
        )
        for values in quality["metrics"].values():
            self.assertEqual(len(values), 2)
            self.assertTrue(all(-1.0 <= value <= 1.0 for value in values))

    def test_renderer_creates_six_png_files_without_layout_warnings(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            unrelated = output_dir / "keep.txt"
            unrelated.write_text("keep", encoding="utf-8")

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                render_six_figures(self.report, output_dir)

            generated = sorted(path.name for path in output_dir.glob("*.png"))
            self.assertEqual(generated, sorted(FIGURE_FILENAMES))
            self.assertTrue(unrelated.exists())
            for filename in FIGURE_FILENAMES:
                path = output_dir / filename
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertGreater(path.stat().st_size, 10_000)
            layout_messages = [
                str(item.message)
                for item in captured
                if "layout" in str(item.message).lower()
            ]
            self.assertEqual(layout_messages, [])

    def test_uses_times_new_roman_and_six_distinct_palettes(self):
        self.assertEqual(matplotlib.rcParams["font.family"], ["serif"])
        self.assertIn("Times New Roman", matplotlib.rcParams["font.serif"])
        self.assertEqual(set(FIGURE_PALETTES), set(FIGURE_FILENAMES))
        self.assertEqual(len(set(FIGURE_PALETTES.values())), 6)

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "dbscan_figures.py"

        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DBSCAN", completed.stdout)


if __name__ == "__main__":
    unittest.main()
