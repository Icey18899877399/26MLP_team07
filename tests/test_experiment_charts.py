"""Real execution checks: charts must account for the evaluated observations."""

import json
import unittest

from ml_core import ExperimentConfig, list_models, run_experiment


class ExperimentChartTests(unittest.TestCase):
    def test_every_runner_returns_finite_charts_from_its_evaluation(self):
        small = {
            "logistic_regression": {"max_iter": 5},
            "random_forest": {"n_estimators": 2, "max_depth": 3},
            "cart_decision_tree": {"max_depth": 3},
            "linear_regression": {"max_iter": 3},
            "mlp_regression": {"max_iter": 2},
            "gbdt_regression": {"n_estimators": 2},
            "isolation_forest": {"n_estimators": 2},
            "one_class_svm": {"max_iter": 2},
            "kmeans": {"n_init": 1, "max_iter": 5},
        }
        for model in list_models():
            with self.subTest(model=model.id):
                result = run_experiment(ExperimentConfig(
                    model=model.id, dataset=model.compatible_datasets[0],
                    params=small.get(model.id.split('.')[0], {}),
                ))
                charts = result.metadata.get("visualizations", [])
                self.assertGreaterEqual(len(charts), 2)
                json.dumps(result.to_dict(), allow_nan=False)
                by_id = {chart["id"]: chart for chart in charts}
                for chart in charts:
                    self.assertTrue(chart["description"])
                    self.assertTrue(chart["option"]["series"])
                if result.task == "classification":
                    data = by_id["confusion_matrix"]["option"]["series"][0]["data"]
                    self.assertEqual(sum(row[2] for row in data), result.metadata["test_sample_count"])
                    self.assertEqual([row[2] for row in data], [result.metrics[key] for key in ("tn", "fp", "fn", "tp")])
                    roc = by_id["roc_curve"]["option"]["series"][0]["data"]
                    self.assertEqual(roc[0], [0.0, 0.0])
                    self.assertEqual(roc[-1], [1.0, 1.0])
                elif result.task == "regression":
                    pairs = by_id["prediction_scatter"]["option"]["series"][0]["data"]
                    residuals = by_id["residual_scatter"]["option"]["series"][0]["data"]
                    self.assertEqual(len(pairs), result.metadata["test_sample_count"])
                    self.assertAlmostEqual(sum((a-b)**2 for a,b in pairs)/len(pairs), result.metrics["mse"], places=6)
                    self.assertEqual(residuals, [[pred, actual-pred] for actual, pred in pairs])
                    self.assertIn("training_loss", by_id)
                elif result.task == "clustering":
                    series = by_id["cluster_projection"]["option"]["series"]
                    self.assertEqual(sum(len(item["data"]) for item in series), 210)
                    xs = [row[0] for item in series for row in item["data"]]
                    self.assertAlmostEqual(sum(xs), 0.0, places=8)
                    self.assertAlmostEqual(sum(x*x for x in xs)/210, 1.0, places=8)
                    sizes = by_id["cluster_sizes"]["option"]["series"][0]["data"]
                    self.assertEqual(sum(sizes), 210)
                else:
                    self.assertIn("in_sample", result.metadata["evaluation_protocol"])
                    series = by_id["anomaly_score_distribution"]["option"]["series"]
                    self.assertEqual(sum(sum(s["data"]) for s in series), result.metadata["sample_count"])
                    self.assertEqual(sum(series[1]["data"]), result.metrics["true_anomalies"])
                    cells = by_id["anomaly_confusion_matrix"]["option"]["series"][0]["data"]
                    self.assertEqual(sum(row[2] for row in cells), result.metadata["sample_count"])
                    self.assertEqual(cells[1][2] + cells[3][2], result.metrics["detected_anomalies"])

    def test_roc_groups_equal_scores_in_one_threshold(self):
        from ml_core.adapters.charts import roc_points
        self.assertEqual(roc_points([0, 1, 0, 1], [0.2, 0.8, 0.8, 0.8]),
                         [[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]])

    def test_roc_positive_class_follows_model_probability_column_order(self):
        # Real KNN fits first-seen classes [1, 0] on seed 42 WDBC. Its high
        # accuracy must not be displayed as an inverted ROC for the negative class.
        result = run_experiment(ExperimentConfig(model="knn.optimized", dataset="wdbc"))
        chart = next(c for c in result.metadata["visualizations"] if c["id"] == "roc_curve")
        points = chart["option"]["series"][0]["data"]
        auc = sum((b[0]-a[0])*(b[1]+a[1])/2 for a, b in zip(points, points[1:]))
        self.assertGreater(auc, 0.95)

    def test_divergent_training_is_an_explicit_execution_error(self):
        from ml_core import ExperimentExecutionError
        with self.assertRaises(ExperimentExecutionError):
            run_experiment(ExperimentConfig(model="linear_regression.optimized",
                dataset="concrete", params={"learning_rate": 1e308, "max_iter": 3}))

    def test_large_regression_caps_training_but_preserves_entire_test_set(self):
        result = run_experiment(ExperimentConfig(model="linear_regression.optimized",
            dataset="california_housing", params={"max_iter": 1}))
        self.assertLessEqual(result.metadata["train_sample_count"], 2000)
        self.assertEqual(result.metadata["test_sample_count"], 4128)
        self.assertEqual(result.metadata["training_available_count"], 16512)
        pairs = result.metadata["visualizations"][0]["option"]["series"][0]["data"]
        self.assertEqual(len(pairs), 1000)


if __name__ == "__main__":
    unittest.main()
