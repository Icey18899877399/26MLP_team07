import os
import tempfile
import unittest


class PackagedDatasetTests(unittest.TestCase):
    def test_wdbc_loads_outside_the_repository_working_directory(self):
        from ml_core.datasets import load_dataset

        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                dataset = load_dataset("wdbc")
            finally:
                os.chdir(original_directory)

        self.assertEqual(len(dataset.features), 569)
        self.assertEqual(len(dataset.features[0]), 30)
        self.assertEqual(set(dataset.targets), {0, 1})

    def test_seeds_exposes_features_and_reference_labels(self):
        from ml_core.datasets import load_dataset

        dataset = load_dataset("seeds")

        self.assertEqual(len(dataset.features), 210)
        self.assertEqual(len(dataset.features[0]), 7)
        self.assertEqual(set(dataset.targets), {0, 1, 2})

    def test_dataset_catalog_matches_loaded_shapes(self):
        from ml_core.datasets import dataset_catalog, load_dataset

        for info in dataset_catalog():
            loaded = load_dataset(info.id)
            self.assertEqual(len(loaded.features), info.sample_count)
            self.assertEqual(len(loaded.features[0]), info.feature_count)
            self.assertEqual(bool(loaded.targets), info.has_target)


if __name__ == "__main__":
    unittest.main()
