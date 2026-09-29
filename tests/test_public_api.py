import importlib
import json
import unittest


class PublicTypeContractTests(unittest.TestCase):
    def test_top_level_exports_construct_the_public_contract(self):
        ml_core = importlib.import_module("ml_core")

        config = ml_core.ExperimentConfig(model="model", dataset="dataset")

        self.assertEqual(config.model, "model")
        self.assertEqual(config.dataset, "dataset")
        self.assertEqual(config.random_state, 42)
        self.assertEqual(dict(config.params), {})

    def test_result_converts_nested_dataclasses_to_json(self):
        ml_core = importlib.import_module("ml_core")
        result = ml_core.ExperimentResult(
            run_id="run-1",
            model="model",
            dataset="dataset",
            task="classification",
            effective_params={},
            metrics={"accuracy": 1.0},
            artifacts=(
                ml_core.ArtifactInfo(
                    name="summary",
                    media_type="application/json",
                    uri="artifacts/summary.json",
                ),
            ),
            metadata={"samples": 2},
        )

        payload = json.loads(json.dumps(result.to_dict()))

        self.assertEqual(payload["metrics"], {"accuracy": 1.0})
        self.assertEqual(payload["artifacts"][0]["name"], "summary")


class DiscoveryAndValidationTests(unittest.TestCase):
    def test_discovery_only_advertises_runnable_items_in_stable_order(self):
        ml_core = importlib.import_module("ml_core")

        self.assertEqual(
            [item.id for item in ml_core.list_models()],
            [
                "cart_decision_tree.optimized",
                "dbscan.optimized",
                "gaussian_naive_bayes.optimized",
                "gbdt_regression.optimized",
                "isolation_forest.optimized",
                "kmeans.optimized",
                "knn.optimized",
                "linear_regression.optimized",
                "logistic_regression.optimized",
                "mlp_regression.optimized",
                "one_class_svm.optimized",
                "random_forest.optimized",
            ],
        )
        self.assertEqual(
            [item.id for item in ml_core.list_datasets()],
            ["6_cardio", "23_mammography", "california_housing", "concrete", "seeds", "wdbc"],
        )

    def test_unknown_model_is_distinct_from_unknown_dataset(self):
        ml_core = importlib.import_module("ml_core")

        with self.assertRaises(ml_core.UnknownModelError):
            ml_core.run_experiment(
                ml_core.ExperimentConfig(model="missing", dataset="wdbc")
            )
        with self.assertRaises(ml_core.UnknownDatasetError):
            ml_core.run_experiment(
                ml_core.ExperimentConfig(
                    model="logistic_regression.optimized",
                    dataset="missing",
                )
            )

    def test_incompatible_model_and_dataset_is_rejected_before_execution(self):
        ml_core = importlib.import_module("ml_core")

        with self.assertRaises(ml_core.IncompatibleDatasetError):
            ml_core.run_experiment(
                ml_core.ExperimentConfig(
                    model="logistic_regression.optimized",
                    dataset="seeds",
                )
            )

    def test_common_configuration_rejects_ambiguous_values(self):
        ml_core = importlib.import_module("ml_core")

        invalid_configs = [
            ml_core.ExperimentConfig(model=1, dataset="wdbc"),
            ml_core.ExperimentConfig(
                model="logistic_regression.optimized",
                dataset="wdbc",
                params=[],
            ),
            ml_core.ExperimentConfig(
                model="logistic_regression.optimized",
                dataset="wdbc",
                random_state=True,
            ),
            ml_core.ExperimentConfig(
                model="logistic_regression.optimized",
                dataset="wdbc",
                test_size=1.0,
            ),
            ml_core.ExperimentConfig(
                model="kmeans.optimized",
                dataset="seeds",
                test_size=0.2,
            ),
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ml_core.InvalidConfigError):
                    ml_core.run_experiment(config)

    def test_unknown_model_parameter_is_rejected(self):
        ml_core = importlib.import_module("ml_core")

        with self.assertRaises(ml_core.InvalidParameterError):
            ml_core.run_experiment(
                ml_core.ExperimentConfig(
                    model="logistic_regression.optimized",
                    dataset="wdbc",
                    params={"not_a_parameter": 1},
                )
            )

    def test_non_string_parameter_keys_use_the_public_validation_error(self):
        ml_core = importlib.import_module("ml_core")
        invalid_parameter_maps = [
            {1: 2},
            {"l2": 0.1, 1: 2},
            {"unknown": 0, 1: 2},
        ]

        for params in invalid_parameter_maps:
            with self.subTest(params=params):
                with self.assertRaises(ml_core.InvalidParameterError):
                    ml_core.run_experiment(
                        ml_core.ExperimentConfig(
                            model="logistic_regression.optimized",
                            dataset="wdbc",
                            params=params,
                        )
                    )

    def test_every_advertised_model_runs_with_its_default_dataset(self):
        ml_core = importlib.import_module("ml_core")
        dataset_ids = {item.id for item in ml_core.list_datasets()}

        for model in ml_core.list_models():
            with self.subTest(model=model.id):
                self.assertTrue(set(model.compatible_datasets) <= dataset_ids)
                result = ml_core.run_experiment(
                    ml_core.ExperimentConfig(
                        model=model.id,
                        dataset=model.compatible_datasets[0],
                    )
                )
                json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
