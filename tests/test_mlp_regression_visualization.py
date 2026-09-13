import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import csv
from pathlib import Path

from matplotlib.colors import to_rgba


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.mlp_regression_figures import (  # noqa: E402
    DISPLAY_BEST,
    FEATURE_NAMES,
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    PLOT_STYLE,
    _teaching_report,
    benchmark_box_positions,
    benchmark_cv_title,
    build_parser,
    kfold_indices,
    jitter_offsets,
    load_concrete_csv,
    partial_dependence,
    permutation_importance,
    regression_metrics,
    render_six_figures,
    run_experiment,
    split_indices,
    pipeline_gain_title,
    timing_title,
    tuning_colormap,
    tuning_experiment_label,
    robustness_title,
)


def sample_report():
    true = [8.0, 13.0, 18.0, 24.0, 31.0, 40.0]
    base_predictions = [11.0, 11.5, 20.5, 21.0, 28.0, 36.0]
    optimized_predictions = [8.5, 13.5, 17.5, 24.5, 30.5, 39.5]
    methods = ["Mean", "Linear", "Base MLP", DISPLAY_BEST]
    return {
        "mechanism": {
            "layer_sizes": [1, 8, 1],
            "activation": "tanh",
            "sample_x": [-2.0, -1.4, -0.8, 0.0, 0.7, 1.3, 2.0],
            "sample_y": [4.0, 1.5, -0.2, 0.0, 1.0, 2.8, 6.0],
            "stages": [
                {"epoch": 1, "grid": [-2, -1, 0, 1, 2], "predicted": [1.0, 0.8, 0.7, 0.8, 1.0], "rmse": 2.5},
                {"epoch": 40, "grid": [-2, -1, 0, 1, 2], "predicted": [3.0, 1.0, 0.1, 1.5, 4.5], "rmse": 0.9},
                {"epoch": 160, "grid": [-2, -1, 0, 1, 2], "predicted": [4.1, 0.8, 0.0, 1.8, 5.9], "rmse": 0.2},
            ],
        },
        "convergence": {
            "base_epochs": [1, 2, 3, 4, 5],
            "base_loss": [1.6, 1.2, 0.95, 0.82, 0.75],
            "optimized_epochs": [1, 2, 3, 4, 5],
            "optimized_train_loss": [1.2, 0.72, 0.43, 0.30, 0.24],
            "optimized_validation_loss": [1.3, 0.78, 0.48, 0.39, 0.42],
            "best_iteration": 4,
            "stop_iteration": 5,
        },
        "tuning": {
            "architectures": ["(8,)", "(16,)", "(16, 8)"],
            "learning_rates": [0.001, 0.005, 0.02],
            "cv_mean": {
                "tanh": [[8.8, 7.1, 7.8], [8.2, 6.5, 7.4], [7.9, 6.0, 7.2]],
                "relu": [[8.4, 6.8, 7.6], [7.7, 6.2, 7.0], [7.4, 5.8, 6.9]],
            },
            "selected_activation": "relu",
            "selected_architecture": "(16, 8)",
            "selected_learning_rate": 0.005,
            "folds": 5,
            "sample_count": 180,
        },
        "interpretation": {
            "feature_names": ["Cement", "Slag", "Fly ash", "Water", "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age"],
            "importances": [5.2, 1.4, 0.9, 3.8, 1.0, 0.5, 0.4, 6.1],
            "importance_std": [0.4, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1, 0.5],
            "partial_dependence": [
                {"feature_name": "Age", "grid": [1, 7, 14, 28, 56], "mean": [20, 25, 29, 35, 39], "low": [13, 17, 21, 27, 31], "high": [30, 34, 38, 44, 48]},
                {"feature_name": "Cement", "grid": [100, 200, 300, 400, 500], "mean": [20, 25, 32, 38, 41], "low": [12, 17, 23, 29, 32], "high": [29, 34, 42, 48, 51]},
            ],
        },
        "diagnostics": {
            "true": true,
            "Base MLP": {
                "predicted": base_predictions,
                "residuals": [a - b for a, b in zip(true, base_predictions)],
                "metrics": regression_metrics(true, base_predictions),
            },
            DISPLAY_BEST: {
                "predicted": optimized_predictions,
                "residuals": [a - b for a, b in zip(true, optimized_predictions)],
                "metrics": regression_metrics(true, optimized_predictions),
            },
        },
        "benchmark": {
            "methods": methods,
            "rmse": [12.0, 8.5, 7.2, 5.4],
            "mae": [9.7, 6.6, 5.6, 4.0],
            "r2": [0.0, 0.50, 0.64, 0.80],
            "cv_rmse": {
                "Mean": [11.5, 12.1, 12.6, 11.9, 12.4],
                "Linear": [8.0, 8.5, 9.0, 8.3, 8.8],
                "Base MLP": [6.8, 7.1, 7.6, 7.0, 7.4],
                DISPLAY_BEST: [5.1, 5.4, 5.7, 5.3, 5.5],
            },
            "seed_rmse": {
                "Base MLP": [7.0, 7.4, 7.2, 7.8, 7.1],
                DISPLAY_BEST: [5.2, 5.5, 5.4, 5.7, 5.3],
            },
            "fit_seconds": [0.0001, 0.01, 0.8, 1.3],
            "folds": 5,
            "seed_runs": 5,
            "timing_repeats": 3,
        },
    }


class MLPVisualizationCalculationTests(unittest.TestCase):
    def test_seed_jitter_supports_any_positive_run_count(self):
        offsets = jitter_offsets(6)

        self.assertEqual(len(offsets), 6)
        self.assertAlmostEqual(sum(offsets), 0.0)
        self.assertLess(min(offsets), 0.0)
        self.assertGreater(max(offsets), 0.0)

    def test_teaching_view_makes_nonlinear_learning_visibly_progressive(self):
        stages = _teaching_report()["stages"]

        self.assertLess(stages[-1]["rmse"], 0.5 * stages[0]["rmse"])

    def test_regression_metrics_match_hand_calculation(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(1 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.5)

    def test_split_indices_are_disjoint_complete_and_reproducible(self):
        first = split_indices(20, test_fraction=0.2, seed=7)
        second = split_indices(20, test_fraction=0.2, seed=7)

        self.assertEqual(first, second)
        train, test = first
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(20)))

    def test_kfold_validation_indices_cover_each_sample_once(self):
        folds = kfold_indices(17, folds=5, seed=11)

        flattened = [index for fold in folds for index in fold]
        self.assertEqual(sorted(flattened), list(range(17)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_partial_dependence_averages_over_observed_rows(self):
        class AdditiveModel:
            @staticmethod
            def predict(rows):
                return [sum(row) for row in rows]

        result = partial_dependence(AdditiveModel(), [[1.0, 10.0], [3.0, 20.0]], 0, [0.0, 2.0])

        self.assertEqual(result["mean"], [15.0, 17.0])
        self.assertEqual(result["low"], [10.0, 12.0])
        self.assertEqual(result["high"], [20.0, 22.0])

    def test_permutation_importance_identifies_the_predictive_feature(self):
        class FirstFeatureModel:
            @staticmethod
            def predict(rows):
                return [row[0] for row in rows]

        X = [[0.0, 8.0], [1.0, 6.0], [2.0, 4.0], [3.0, 2.0], [4.0, 0.0]]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]

        result = permutation_importance(FirstFeatureModel(), X, y, repeats=5, seed=3)

        self.assertGreater(result["mean"][0], 0.0)
        self.assertAlmostEqual(result["mean"][1], 0.0)
        self.assertEqual(len(result["std"]), 2)


class MLPVisualizationContractTests(unittest.TestCase):
    def test_tuning_colormap_uses_the_figure_specific_palette(self):
        palette = FIGURE_PALETTES[FIGURE_FILENAMES[2]]
        colormap = tuning_colormap(palette)

        self.assertEqual(colormap(0.0), to_rgba(palette[0]))
        self.assertEqual(colormap(1.0), to_rgba(palette[2]))

    def test_benchmark_titles_disclose_post_selection_and_run_counts(self):
        self.assertEqual(
            benchmark_cv_title(5),
            "Descriptive post-selection 5-fold RMSE\n(settings selected on overlapping training data)",
        )
        self.assertEqual(robustness_title(5), "Initialization robustness (5 seeds)")
        self.assertEqual(timing_title(3), "Median training cost (3 runs)")
        self.assertEqual(pipeline_gain_title(), "Full-pipeline gain")

    def test_math_text_uses_times_new_roman(self):
        self.assertEqual(PLOT_STYLE["mathtext.fontset"], "custom")
        self.assertEqual(PLOT_STYLE["mathtext.rm"], "Times New Roman")

    def test_tuning_label_discloses_the_training_subset_size(self):
        label = tuning_experiment_label({"folds": 5, "sample_count": 180})

        self.assertEqual(
            label,
            "Bounded 5-fold CV on representative outer-training subset (n=180)",
        )

    def test_benchmark_boxplots_share_zero_based_method_positions(self):
        self.assertEqual(benchmark_box_positions(["A", "B", "C", "D"]), [0, 1, 2, 3])

    def test_best_model_label_states_that_the_search_was_bounded(self):
        self.assertEqual(DISPLAY_BEST, "Best-grid MLP")

    def test_default_data_path_loads_the_concrete_dataset(self):
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
            self.assertNotIn("glyph", warning_text)

    def test_renderer_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            note = output_dir / "notes.txt"
            note.write_text("keep", encoding="utf-8")

            render_six_figures(sample_report(), output_dir)

            self.assertEqual(note.read_text(encoding="utf-8"), "keep")

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "mlp_regression_figures.py"
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

    def test_reduced_end_to_end_experiment_keeps_test_targets_in_diagnostics_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "tiny_concrete.csv"
            rows = []
            for index in range(40):
                features = [float(index + feature * 0.25) for feature in range(8)]
                target = 5.0 + 0.7 * index + 0.2 * (index % 4)
                rows.append([*features, target])
            with data_path.open("w", encoding="utf-8", newline="") as target_file:
                writer = csv.writer(target_file)
                writer.writerow([*FEATURE_NAMES, "Strength"])
                writer.writerows(rows)

            report = run_experiment(
                data_path,
                seed=7,
                folds=2,
                tuning_samples=12,
                tuning_max_iter=2,
                final_max_iter=2,
                seed_runs=1,
                timing_repeats=1,
            )

            _, test_indices = split_indices(40, test_fraction=0.2, seed=7)
            expected_test_targets = [rows[index][-1] for index in test_indices]
            self.assertEqual(report["diagnostics"]["true"], expected_test_targets)
            self.assertEqual(report["tuning"]["sample_count"], 12)
            self.assertTrue(all(len(values) == 2 for values in report["benchmark"]["cv_rmse"].values()))
            self.assertTrue(all(len(values) == 1 for values in report["benchmark"]["seed_rmse"].values()))

            perturbed_rows = [list(row) for row in rows]
            for index in test_indices:
                perturbed_rows[index] = [value + 10_000.0 for value in rows[index][:-1]] + [rows[index][-1] + 10_000.0]
            perturbed_path = Path(temp_dir) / "tiny_concrete_test_perturbed.csv"
            with perturbed_path.open("w", encoding="utf-8", newline="") as target_file:
                writer = csv.writer(target_file)
                writer.writerow([*FEATURE_NAMES, "Strength"])
                writer.writerows(perturbed_rows)

            perturbed_report = run_experiment(
                perturbed_path,
                seed=7,
                folds=2,
                tuning_samples=12,
                tuning_max_iter=2,
                final_max_iter=2,
                seed_runs=1,
                timing_repeats=1,
            )

            self.assertEqual(report["tuning"], perturbed_report["tuning"])
            self.assertEqual(report["convergence"], perturbed_report["convergence"])
            self.assertEqual(report["interpretation"], perturbed_report["interpretation"])
            self.assertEqual(report["benchmark"]["cv_rmse"], perturbed_report["benchmark"]["cv_rmse"])


if __name__ == "__main__":
    unittest.main()
