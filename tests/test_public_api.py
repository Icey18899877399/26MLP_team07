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
            ["kmeans.optimized", "logistic_regression.optimized"],
        )
        self.assertEqual(
            [item.id for item in ml_core.list_datasets()],
            ["seeds", "wdbc"],
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


if __name__ == "__main__":
    unittest.main()
