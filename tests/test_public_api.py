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


if __name__ == "__main__":
    unittest.main()
