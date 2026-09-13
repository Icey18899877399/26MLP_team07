import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Models.random_forest_optimized import OptimizedRandomForestClassifierScratch


class OptimizedRandomForestClassifierScratchTests(unittest.TestCase):
    def test_invalid_ensemble_parameters_are_rejected(self):
        X = [[0.0], [1.0]]
        y = [0, 1]
        invalid_models = [
            OptimizedRandomForestClassifierScratch(n_estimators=0),
            OptimizedRandomForestClassifierScratch(voting="maybe"),
            OptimizedRandomForestClassifierScratch(max_samples=1.2),
            OptimizedRandomForestClassifierScratch(n_jobs=0),
            OptimizedRandomForestClassifierScratch(bootstrap=False, oob_score=True),
        ]
        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(X, y)

    def test_invalid_tree_optimization_parameters_are_rejected(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0, 0, 1, 1]
        invalid_models = [
            OptimizedRandomForestClassifierScratch(
                bootstrap=False,
                oob_score=False,
                max_samples=0.5,
            ),
            OptimizedRandomForestClassifierScratch(class_weight="unknown"),
            OptimizedRandomForestClassifierScratch(class_weight={0: 1.0, 1: 0.0}),
            OptimizedRandomForestClassifierScratch(class_weight={0: 1.0}),
            OptimizedRandomForestClassifierScratch(ccp_alpha=-0.01),
            OptimizedRandomForestClassifierScratch(min_impurity_decrease=-0.01),
        ]
        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(X, y)

    def test_predict_proba_accepts_a_single_pass_iterator(self):
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=3,
            bootstrap=False,
            oob_score=False,
            random_state=43,
        ).fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])

        probabilities = model.predict_proba(iter([[0.5], [2.5]]))

        self.assertEqual(len(probabilities), 2)
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in probabilities))

    def test_oob_coverage_reports_the_actual_denominator(self):
        X = [[float(i)] for i in range(12)]
        y = [0] * 6 + [1] * 6
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=1,
            max_depth=2,
            random_state=47,
        ).fit(X, y)

        covered = sum(row is not None for row in model.oob_decision_function_)
        self.assertAlmostEqual(model.oob_coverage_, covered / len(X))
        self.assertGreater(model.oob_coverage_, 0.0)
        self.assertLess(model.oob_coverage_, 1.0)

    def test_oob_probabilities_use_only_trees_that_excluded_each_sample(self):
        X = [[float(i), float(i % 3)] for i in range(18)]
        y = [0] * 9 + [1] * 9
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=9,
            max_depth=3,
            max_features=None,
            random_state=53,
        ).fit(X, y)

        for sample_index, actual in enumerate(model.oob_decision_function_):
            expected_sum = [0.0, 0.0]
            count = 0
            for tree, sampled in zip(model.estimators_, model.bootstrap_indices_):
                if sample_index in sampled:
                    continue
                tree_row = tree.predict_proba([X[sample_index]])[0]
                by_label = dict(zip(tree.classes, tree_row))
                expected_sum[0] += by_label.get(0, 0.0)
                expected_sum[1] += by_label.get(1, 0.0)
                count += 1
            expected = [value / count for value in expected_sum] if count else None
            if expected is None:
                self.assertIsNone(actual)
            else:
                for observed, hand_checked in zip(actual, expected):
                    self.assertAlmostEqual(observed, hand_checked)

    def test_oob_estimates_are_created_from_unseen_trees(self):
        X = [[float(i), float(i % 4)] for i in range(60)]
        y = [0] * 30 + [1] * 30

        model = OptimizedRandomForestClassifierScratch(
            n_estimators=31,
            max_depth=3,
            max_features=None,
            oob_score=True,
            random_state=13,
        ).fit(X, y)

        available = [row for row in model.oob_decision_function_ if row is not None]
        self.assertEqual(len(available), len(X))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in available))
        self.assertGreaterEqual(model.oob_score_, 0.0)
        self.assertLessEqual(model.oob_score_, 1.0)

    def test_parallel_and_serial_training_are_reproducible(self):
        X = [[float(i), float(i % 5), float(i % 2)] for i in range(40)]
        y = [0 if i < 20 else 1 for i in range(40)]
        serial = OptimizedRandomForestClassifierScratch(
            n_estimators=17,
            max_depth=3,
            n_jobs=1,
            random_state=23,
        ).fit(X, y)
        parallel = OptimizedRandomForestClassifierScratch(
            n_estimators=17,
            max_depth=3,
            n_jobs=2,
            random_state=23,
        ).fit(X, y)

        self.assertEqual(serial.bootstrap_indices_, parallel.bootstrap_indices_)
        self.assertEqual(serial.predict(X), parallel.predict(X))
        self.assertAlmostEqual(serial.oob_score_, parallel.oob_score_)

    def test_soft_vote_retains_leaf_uncertainty(self):
        X = [[0.0], [1.0], [2.0]]
        y = [0, 0, 1]
        soft = OptimizedRandomForestClassifierScratch(
            n_estimators=1,
            max_depth=0,
            bootstrap=False,
            oob_score=False,
            voting="soft",
            random_state=2,
        ).fit(X, y)
        hard = OptimizedRandomForestClassifierScratch(
            n_estimators=1,
            max_depth=0,
            bootstrap=False,
            oob_score=False,
            voting="hard",
            random_state=2,
        ).fit(X, y)

        self.assertEqual(soft.predict_proba([[10.0]])[0], [2 / 3, 1 / 3])
        self.assertEqual(hard.predict_proba([[10.0]])[0], [1.0, 0.0])

    def test_tree_optimization_parameters_are_forwarded(self):
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=3,
            max_depth=4,
            min_samples_leaf=2,
            min_impurity_decrease=0.01,
            max_features=1,
            class_weight="balanced",
            ccp_alpha=0.02,
            random_state=29,
        ).fit(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            [0, 0, 0, 1, 1, 1],
        )

        tree = model.estimators_[0]
        self.assertEqual(tree.max_depth, 4)
        self.assertEqual(tree.min_samples_leaf, 2)
        self.assertEqual(tree.min_impurity_decrease, 0.01)
        self.assertEqual(tree.max_features, 1)
        self.assertEqual(tree.class_weight, "balanced")
        self.assertEqual(tree.ccp_alpha, 0.02)

    def test_max_samples_controls_each_bootstrap_size(self):
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=5,
            max_samples=0.5,
            random_state=31,
        ).fit([[float(i)] for i in range(20)], [0] * 10 + [1] * 10)

        self.assertTrue(all(len(indices) == 10 for indices in model.bootstrap_indices_))

    def test_optimized_importances_are_normalized(self):
        X = [[float(i), float(i % 2)] for i in range(30)]
        y = [0] * 15 + [1] * 15
        model = OptimizedRandomForestClassifierScratch(
            n_estimators=15,
            max_depth=3,
            max_features=None,
            random_state=37,
        ).fit(X, y)

        self.assertAlmostEqual(sum(model.feature_importances_), 1.0)
        self.assertGreater(model.feature_importances_[0], model.feature_importances_[1])


if __name__ == "__main__":
    unittest.main()
