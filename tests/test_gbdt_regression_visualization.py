import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.gbdt_regression_figures import (  # noqa: E402
    FIGURE_FILENAMES,
    FIGURE_PALETTES,
    DISPLAY_BEST,
    PLOT_STYLE,
    _MeanRegressor,
    _linear_baseline,
    build_parser,
    kfold_indices,
    load_concrete_csv,
    heatmap_text_color,
    partial_dependence,
    regression_metrics,
    render_six_figures,
    split_indices,
    stage_staircase,
    staged_predictions,
    stopping_annotation,
)


def sample_report():
    true = [8.0, 13.0, 18.0, 24.0, 31.0, 40.0]
    base_predictions = [10.0, 12.0, 20.0, 22.0, 29.0, 37.0]
    optimized_predictions = [8.5, 13.5, 17.5, 24.5, 30.5, 39.5]
    return {
        "mechanism": {
            "feature_name": "Age",
            "sample_x": [1, 3, 7, 14, 28, 56],
            "sample_y": [12, 16, 22, 29, 36, 43],
            "stages": [
                {"round": 0, "edges": [1, 56], "predicted": [26], "rmse": 11.0},
                {"round": 1, "edges": [1, 10, 56], "predicted": [20, 34], "rmse": 7.0},
                {"round": 5, "edges": [1, 5, 20, 56], "predicted": [14, 28, 39], "rmse": 3.0},
                {"round": 20, "edges": [1, 3, 7, 14, 28, 56], "predicted": [12, 16, 22, 29, 36], "rmse": 0.5},
            ],
        },
        "convergence": {
            "base_rounds": [1, 2, 3, 4, 5],
            "base_train": [80, 55, 40, 32, 27],
            "base_validation": [85, 62, 49, 47, 50],
            "optimized_rounds": [1, 2, 3, 4, 5],
            "optimized_train": [70, 43, 29, 21, 17],
            "optimized_validation": [74, 48, 34, 30, 31],
            "best_round": 4,
        },
        "tuning": {
            "learning_rates": [0.03, 0.07, 0.12],
            "depths": [1, 2, 3],
            "cv_mean": [[8.6, 7.9, 8.1], [7.8, 6.9, 7.2], [7.7, 7.1, 7.8]],
            "selected_learning_rate": 0.07,
            "selected_depth": 2,
        },
        "interpretation": {
            "feature_names": ["Cement", "Age", "Water", "Slag"],
            "importances": [0.45, 0.29, 0.17, 0.09],
            "partial_dependence": [
                {
                    "feature_name": "Cement",
                    "grid": [100, 200, 300, 400, 500],
                    "mean": [20, 25, 32, 38, 41],
                    "low": [12, 17, 23, 29, 32],
                    "high": [29, 34, 42, 48, 51],
                },
                {
                    "feature_name": "Age",
                    "grid": [1, 7, 14, 28, 56],
                    "mean": [21, 25, 29, 35, 39],
                    "low": [13, 17, 21, 27, 31],
                    "high": [30, 34, 38, 44, 48],
                },
            ],
        },
        "diagnostics": {
            "true": true,
            "Base GBDT": {
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
            "methods": ["Mean", "Linear", "Base GBDT", DISPLAY_BEST],
            "rmse": [12.0, 8.5, 6.8, 5.4],
            "mae": [9.7, 6.6, 5.1, 4.0],
            "r2": [0.0, 0.50, 0.68, 0.80],
            "cv_rmse": {
                "Mean": [11.5, 12.1, 12.6],
                "Linear": [8.0, 8.5, 9.0],
                "Base GBDT": [6.4, 6.8, 7.2],
                DISPLAY_BEST: [5.1, 5.4, 5.7],
            },
            "fit_seconds": [0.0001, 0.01, 0.8, 0.3],
        },
    }


class GBDTVisualizationCalculationTests(unittest.TestCase):
    def test_linear_baseline_converges_on_a_standardized_plane(self):
        X = [[float(index), float(index % 3)] for index in range(30)]
        y = [2.0 + 3.0 * row[0] - 4.0 * row[1] for row in X]

        model = _linear_baseline(seed=7).fit(X, y)

        self.assertLess(model.n_iter, model.max_iter)
        self.assertLess(regression_metrics(y, model.predict(X))["rmse"], 1e-3)

    def test_stage_staircase_uses_actual_tree_split_thresholds(self):
        from Models.gbdt_regression import GBDTRegressorScratch

        X = [[0.0], [1.0], [4.0], [5.0]]
        y = [0.0, 0.0, 10.0, 10.0]
        model = GBDTRegressorScratch(
            n_estimators=2,
            learning_rate=0.5,
            max_depth=1,
        ).fit(X, y)

        staircase = stage_staircase(model, round_number=2, low=0.0, high=5.0)
        expected = sorted({tree.root.threshold for tree in model.estimators_})

        self.assertEqual(staircase["edges"][1:-1], expected)
        self.assertEqual(len(staircase["predicted"]), len(staircase["edges"]) - 1)

    def test_mean_regressor_learns_the_training_target_mean(self):
        model = _MeanRegressor().fit([[1.0], [2.0], [3.0]], [2.0, 5.0, 8.0])

        self.assertEqual(model.mean_, 5.0)
        self.assertEqual(model.predict([[10.0], [20.0]]), [5.0, 5.0])

    def test_stopping_annotation_distinguishes_a_limit_from_an_interior_choice(self):
        self.assertEqual(stopping_annotation(60, 60), "Run limit = 60")
        self.assertEqual(stopping_annotation(44, 60), "Selected round = 44")

    def test_heatmap_text_contrasts_with_reversed_blue_scale(self):
        self.assertEqual(heatmap_text_color(5.0, 5.0, 12.0), "white")
        self.assertEqual(heatmap_text_color(12.0, 5.0, 12.0), "#003049")

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

    def test_staged_predictions_start_at_mean_and_end_at_predict(self):
        from Models.gbdt_regression import GBDTRegressorScratch

        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 0.0, 10.0, 10.0]
        model = GBDTRegressorScratch(
            n_estimators=2,
            learning_rate=0.5,
            max_depth=1,
        ).fit(X, y)

        stages = staged_predictions(model, X)

        self.assertEqual(stages[0], [5.0] * 4)
        self.assertEqual(stages[-1], model.predict(X))

    def test_partial_dependence_averages_over_observed_rows(self):
        class AdditiveModel:
            @staticmethod
            def predict(rows):
                return [sum(row) for row in rows]

        result = partial_dependence(
            AdditiveModel(),
            [[1.0, 10.0], [3.0, 20.0]],
            feature_index=0,
            grid=[0.0, 2.0],
        )

        self.assertEqual(result["mean"], [15.0, 17.0])
        self.assertEqual(result["low"], [10.0, 12.0])
        self.assertEqual(result["high"], [20.0, 22.0])


class GBDTVisualizationContractTests(unittest.TestCase):
    def test_mechanism_figure_is_explicitly_marked_as_a_teaching_view(self):
        source = (PROJECT_ROOT / "visualization" / "gbdt_regression_figures.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("One-feature teaching view", source)
        self.assertIn("not a benchmark", source)
        self.assertIn("baseline=None", source)

    def test_best_model_label_states_that_the_search_was_bounded(self):
        self.assertEqual(DISPLAY_BEST, "Best-grid GBDT")

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

    def test_renderer_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            note = output_dir / "notes.txt"
            note.write_text("keep", encoding="utf-8")

            render_six_figures(sample_report(), output_dir)

            self.assertEqual(note.read_text(encoding="utf-8"), "keep")

    def test_command_line_help_runs_directly(self):
        script = PROJECT_ROOT / "visualization" / "gbdt_regression_figures.py"
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
