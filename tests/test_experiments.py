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


if __name__ == "__main__":
    unittest.main()
