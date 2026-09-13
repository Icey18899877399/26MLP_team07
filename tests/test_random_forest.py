import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.random_forest import RandomForestClassifierScratch


class RandomForestClassifierScratchTests(unittest.TestCase):
    def test_fractional_max_features_keeps_at_least_one_candidate(self):
        X = [[float(value), 1.0] for value in range(20)]
        y = [0] * 10 + [1] * 10
        model = RandomForestClassifierScratch(
            n_estimators=15,
            max_depth=2,
            max_features=0.5,
            random_state=41,
        ).fit(X, y)

        self.assertEqual(model.predict([[2.0, 1.0], [17.0, 1.0]]), [0, 1])

    def test_zero_estimators_is_rejected_before_training(self):
        with self.assertRaises(ValueError):
            RandomForestClassifierScratch(n_estimators=0).fit([[0.0]], [0])

    def test_bootstrap_forest_learns_a_separable_binary_problem(self):
        X = [[float(value), 1.0] for value in range(20)]
        y = [0] * 10 + [1] * 10

        model = RandomForestClassifierScratch(
            n_estimators=15,
            max_depth=3,
            max_features=None,
            random_state=7,
        ).fit(X, y)

        self.assertEqual(model.predict([[2.0, 1.0], [17.0, 1.0]]), [0, 1])
        self.assertEqual(len(model.estimators_), 15)

    def test_bootstrap_samples_have_training_size_and_repeated_rows(self):
        model = RandomForestClassifierScratch(
            n_estimators=7,
            max_depth=1,
            random_state=3,
        ).fit([[float(i)] for i in range(8)], [0, 0, 0, 0, 1, 1, 1, 1])

        self.assertTrue(all(len(indices) == 8 for indices in model.bootstrap_indices_))
        self.assertTrue(any(len(set(indices)) < 8 for indices in model.bootstrap_indices_))

    def test_hard_vote_probabilities_follow_class_order_and_sum_to_one(self):
        model = RandomForestClassifierScratch(
            n_estimators=9,
            max_depth=2,
            max_features=None,
            random_state=11,
        ).fit([[0.0], [0.5], [3.0], [3.5]], ["low", "low", "high", "high"])

        probabilities = model.predict_proba([[0.2], [3.2]])

        self.assertEqual(model.classes, ["low", "high"])
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in probabilities))
        self.assertEqual(model.predict([[0.2], [3.2]]), ["low", "high"])

    def test_fixed_seed_reproduces_bootstraps_and_predictions(self):
        X = [[float(i), float(i % 3)] for i in range(18)]
        y = [0 if i < 9 else 1 for i in range(18)]
        first = RandomForestClassifierScratch(n_estimators=11, random_state=19).fit(X, y)
        second = RandomForestClassifierScratch(n_estimators=11, random_state=19).fit(X, y)

        self.assertEqual(first.bootstrap_indices_, second.bootstrap_indices_)
        self.assertEqual(first.predict(X), second.predict(X))

    def test_feature_importance_identifies_the_informative_feature(self):
        X = [[float(i), 5.0, 2.0] for i in range(30)]
        y = [0] * 15 + [1] * 15
        model = RandomForestClassifierScratch(
            n_estimators=13,
            max_depth=2,
            max_features=None,
            random_state=5,
        ).fit(X, y)

        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertEqual(max(range(3), key=model.feature_importances_.__getitem__), 0)


if __name__ == "__main__":
    unittest.main()
