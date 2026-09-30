import json
import unittest


class LogisticExperimentTests(unittest.TestCase):
    def test_logistic_experiment_is_deterministic_and_json_safe(self):
        from ml_core import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            model="logistic_regression.optimized",
            dataset="wdbc",
            params={"max_iter": 100},
            test_size=0.2,
            random_state=7,
        )

        first = run_experiment(config)
        second = run_experiment(config)

        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.task, "classification")
        self.assertEqual(
            set(first.metrics),
            {"accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"},
        )
        self.assertGreaterEqual(first.metrics["accuracy"], 0.0)
        self.assertLessEqual(first.metrics["accuracy"], 1.0)
        confusion_total = sum(
            int(first.metrics[key]) for key in ("tn", "fp", "fn", "tp")
        )
        self.assertEqual(confusion_total, first.metadata["test_sample_count"])
        self.assertEqual(first.metadata["random_state"], 7)
        self.assertEqual(first.metadata["test_size"], 0.2)
        json.dumps(first.to_dict())

    def test_logistic_parameter_validation_rejects_invalid_values(self):
        from ml_core import ExperimentConfig, InvalidParameterError, run_experiment

        invalid_parameters = [
            {"learning_rate": 0.0},
            {"learning_rate": True},
            {"max_iter": 0},
            {"max_iter": 2.5},
            {"threshold": -0.1},
            {"threshold": 1.1},
            {"l2": -1.0},
            {"tol": -1.0},
            {"standardize": 1},
            {"class_weight": "minority"},
        ]

        for params in invalid_parameters:
            with self.subTest(params=params):
                with self.assertRaises(InvalidParameterError):
                    run_experiment(
                        ExperimentConfig(
                            model="logistic_regression.optimized",
                            dataset="wdbc",
                            params=params,
                            test_size=0.2,
                        )
                    )


class KMeansExperimentTests(unittest.TestCase):
    def test_adjusted_rand_index_matches_hand_checked_partitions(self):
        from ml_core.adapters.clustering import adjusted_rand_index

        self.assertEqual(
            adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]),
            1.0,
        )
        self.assertAlmostEqual(
            adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]),
            -0.5,
        )

    def test_kmeans_experiment_is_deterministic_and_scores_reference_labels(self):
        from ml_core import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            model="kmeans.optimized",
            dataset="seeds",
            params={"n_init": 3, "max_iter": 100},
            random_state=11,
        )

        first = run_experiment(config)
        second = run_experiment(config)

        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.task, "clustering")
        self.assertEqual(
            set(first.metrics),
            {"inertia", "adjusted_rand_index", "n_clusters"},
        )
        self.assertGreaterEqual(first.metrics["inertia"], 0.0)
        self.assertGreaterEqual(first.metrics["adjusted_rand_index"], -1.0)
        self.assertLessEqual(first.metrics["adjusted_rand_index"], 1.0)
        self.assertEqual(first.metadata["sample_count"], 210)
        self.assertEqual(first.metadata["feature_count"], 7)
        self.assertEqual(first.metadata["random_state"], 11)
        self.assertEqual(sum(first.metadata["cluster_sizes"].values()), 210)
        json.dumps(first.to_dict())

    def test_kmeans_parameter_validation_rejects_invalid_values(self):
        from ml_core import ExperimentConfig, InvalidParameterError, run_experiment

        invalid_parameters = [
            {"n_clusters": 0},
            {"n_clusters": True},
            {"n_clusters": 211},
            {"init": "first"},
            {"n_init": 0},
            {"max_iter": 0},
            {"tol": -1.0},
            {"standardize": 1},
        ]

        for params in invalid_parameters:
            with self.subTest(params=params):
                with self.assertRaises(InvalidParameterError):
                    run_experiment(
                        ExperimentConfig(
                            model="kmeans.optimized",
                            dataset="seeds",
                            params=params,
                        )
                    )


class NewClassificationExperimentTests(unittest.TestCase):
    """新注册分类模型的轻量 smoke（削减参数控制耗时）。"""

    def test_new_classification_runners_produce_standard_metrics(self):
        from ml_core import ExperimentConfig, run_experiment

        cases = [
            ("knn.optimized", {}),
            ("gaussian_naive_bayes.optimized", {}),
            ("cart_decision_tree.optimized", {"max_depth": 4}),
            ("random_forest.optimized", {"n_estimators": 10, "max_depth": 4}),
        ]
        for model_id, params in cases:
            with self.subTest(model=model_id):
                result = run_experiment(
                    ExperimentConfig(model=model_id, dataset="wdbc", params=params)
                )
                self.assertEqual(result.task, "classification")
                self.assertEqual(
                    set(result.metrics),
                    {"accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"},
                )
                confusion_total = sum(
                    int(result.metrics[key]) for key in ("tn", "fp", "fn", "tp")
                )
                self.assertEqual(
                    confusion_total, result.metadata["test_sample_count"]
                )
                json.dumps(result.to_dict())


class RegressionExperimentTests(unittest.TestCase):
    def test_regression_runners_produce_standard_metrics(self):
        from ml_core import ExperimentConfig, run_experiment

        cases = [
            ("linear_regression.optimized", {"max_iter": 200}),
            ("gbdt_regression.optimized", {"n_estimators": 5}),
            ("mlp_regression.optimized", {"max_iter": 50}),
        ]
        for model_id, params in cases:
            with self.subTest(model=model_id):
                result = run_experiment(
                    ExperimentConfig(
                        model=model_id,
                        dataset="concrete",
                        params=params,
                        test_size=0.2,
                    )
                )
                self.assertEqual(result.task, "regression")
                self.assertEqual(set(result.metrics), {"mse", "rmse", "mae", "r2"})
                for value in result.metrics.values():
                    self.assertTrue(float(value) == float(value))  # 非 NaN
                json.dumps(result.to_dict())


class AnomalyAndDbscanExperimentTests(unittest.TestCase):
    def test_anomaly_runners_produce_standard_metrics(self):
        from ml_core import ExperimentConfig, run_experiment

        result = run_experiment(
            ExperimentConfig(
                model="isolation_forest.optimized",
                dataset="6_cardio",
                params={"n_estimators": 20},
            )
        )
        self.assertEqual(result.task, "anomaly_detection")
        self.assertEqual(
            set(result.metrics),
            {
                "true_anomalies",
                "detected_anomalies",
                "anomaly_recall",
                "anomaly_precision",
                "anomaly_f1",
            },
        )
        self.assertEqual(result.metrics["true_anomalies"], 176)
        for key in ("anomaly_recall", "anomaly_precision", "anomaly_f1"):
            self.assertGreaterEqual(result.metrics[key], 0.0)
            self.assertLessEqual(result.metrics[key], 1.0)
        json.dumps(result.to_dict())

    def test_one_class_svm_trains_on_normal_subsample(self):
        from ml_core import ExperimentConfig, run_experiment

        result = run_experiment(
            ExperimentConfig(
                model="one_class_svm.optimized",
                dataset="6_cardio",
                params={"max_iter": 100},
            )
        )
        self.assertEqual(result.task, "anomaly_detection")
        self.assertLessEqual(result.metadata["train_sample_count"], 300)
        self.assertEqual(result.metadata["train_anomaly_count"], 0)
        self.assertEqual(
            set(result.metrics),
            {
                "true_anomalies",
                "detected_anomalies",
                "anomaly_recall",
                "anomaly_precision",
                "anomaly_f1",
            },
        )
        json.dumps(result.to_dict())

    def test_dbscan_produces_cluster_report(self):
        from ml_core import ExperimentConfig, run_experiment

        result = run_experiment(
            ExperimentConfig(
                model="dbscan.optimized",
                dataset="seeds",
                params={"eps": 0.6, "min_samples": 4},
            )
        )
        self.assertEqual(result.task, "clustering")
        self.assertEqual(
            set(result.metrics),
            {"n_clusters", "noise_points", "adjusted_rand_index"},
        )
        self.assertGreaterEqual(result.metrics["adjusted_rand_index"], -1.0)
        self.assertLessEqual(result.metrics["adjusted_rand_index"], 1.0)
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
