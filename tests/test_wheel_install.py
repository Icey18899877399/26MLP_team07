"""Smoke test that can be executed with a Python environment containing only the wheel."""

import json
import os
import tempfile
import unittest
from pathlib import Path


class InstalledWheelSmokeTests(unittest.TestCase):
    def test_public_api_and_package_data_work_outside_the_checkout(self):
        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                import ml_core
                from ml_core import (
                    ExperimentConfig,
                    list_datasets,
                    list_models,
                    run_experiment,
                )

                self.assertEqual(len(list_models()), 2)
                self.assertEqual(len(list_datasets()), 2)
                if os.environ.get("ML_CORE_EXPECT_WHEEL") == "1":
                    checkout = Path(__file__).resolve().parents[1]
                    imported_from = Path(ml_core.__file__).resolve()
                    self.assertNotIn(checkout, imported_from.parents)
                    self.assertIn("site-packages", imported_from.parts)
                result = run_experiment(
                    ExperimentConfig(
                        model="kmeans.optimized",
                        dataset="seeds",
                        params={"n_init": 2},
                        random_state=42,
                    )
                )
                json.dumps(result.to_dict())
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
