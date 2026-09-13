"""手写随机森林分类器优化版。"""

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

from .cart_decision_tree_optimized import OptimizedCARTClassifierScratch


# 进程启动时只传一次训练集，避免每棵树重复序列化完整数据。
_WORKER_FEATURES = None
_WORKER_TARGETS = None
_WORKER_PARAMETERS = None


def _initialize_worker(features, targets, parameters):
    global _WORKER_FEATURES, _WORKER_TARGETS, _WORKER_PARAMETERS
    _WORKER_FEATURES = features
    _WORKER_TARGETS = targets
    _WORKER_PARAMETERS = parameters


def _build_tree(features, targets, parameters, task):
    indices, tree_seed = task
    tree = OptimizedCARTClassifierScratch(random_state=tree_seed, **parameters)
    tree.fit(
        [features[index] for index in indices],
        [targets[index] for index in indices],
    )
    return tree


def _fit_tree_in_worker(task):
    return _build_tree(
        _WORKER_FEATURES,
        _WORKER_TARGETS,
        _WORKER_PARAMETERS,
        task,
    )


class OptimizedRandomForestClassifierScratch:
    """支持 OOB、软投票、剪枝与并行接口的随机森林。"""

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        max_features="sqrt",
        class_weight=None,
        ccp_alpha=0.0,
        bootstrap=True,
        max_samples=None,
        voting="soft",
        oob_score=True,
        n_jobs=1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.voting = voting
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classes = []
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.feature_importances_ = []
        self.oob_decision_function_ = []
        self.oob_score_ = None
        self.oob_coverage_ = 0.0
        self._features = []
        self._targets = []
        self._tree_max_features = max_features

    def _bootstrap_size(self, sample_count):
        if self.max_samples is None:
            return sample_count
        if isinstance(self.max_samples, float):
            return max(1, math.ceil(sample_count * self.max_samples))
        return min(sample_count, int(self.max_samples))

    def _validate_parameters(self, sample_count, feature_count):
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if self.voting not in ("soft", "hard"):
            raise ValueError("voting must be 'soft' or 'hard'")
        if self.n_jobs != -1 and (not isinstance(self.n_jobs, int) or self.n_jobs < 1):
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        if self.max_samples is not None and not self.bootstrap:
            raise ValueError("max_samples is only used when bootstrap=True")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if self.min_samples_split < 2 or self.min_samples_leaf < 1:
            raise ValueError("sample limits must be positive and min_samples_split at least 2")
        if not math.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease must be finite and non-negative")
        if not math.isfinite(self.ccp_alpha) or self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be finite and non-negative")

        if self.class_weight not in (None, "balanced"):
            if not isinstance(self.class_weight, dict):
                raise ValueError("class_weight must be None, 'balanced', or a dictionary")
            for label in self.classes:
                if label not in self.class_weight:
                    raise ValueError("class_weight must contain every observed class")
                weight = self.class_weight[label]
                if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError("class weights must be finite and positive")

        if self.max_samples is not None:
            if isinstance(self.max_samples, float):
                if not 0.0 < self.max_samples <= 1.0:
                    raise ValueError("fractional max_samples must be in (0, 1]")
            elif not isinstance(self.max_samples, int) or not 1 <= self.max_samples <= sample_count:
                raise ValueError("integer max_samples must be between 1 and the sample count")

        if self.max_features in (None, "sqrt", "log2"):
            self._tree_max_features = self.max_features
        elif isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError("fractional max_features must be in (0, 1]")
            self._tree_max_features = max(1, math.ceil(feature_count * self.max_features))
        elif isinstance(self.max_features, int) and 1 <= self.max_features <= feature_count:
            self._tree_max_features = self.max_features
        else:
            raise ValueError("max_features must select between 1 and all available features")

    def _tree_parameters(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self._tree_max_features,
            "class_weight": self.class_weight,
            "ccp_alpha": self.ccp_alpha,
        }

    def _make_tasks(self, sample_count):
        generator = random.Random(self.random_state)
        tasks = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = [
                    generator.randrange(sample_count)
                    for _ in range(self._bootstrap_size(sample_count))
                ]
            else:
                indices = list(range(sample_count))
            tasks.append((indices, generator.randrange(2**32)))
        return tasks

    def fit(self, X, y):
        """训练森林，并使用未进入相应 Bootstrap 样本的树计算 OOB。"""
        self._features = [list(row) for row in X]
        self._targets = list(y)
        self.classes = []
        for label in self._targets:
            if label not in self.classes:
                self.classes.append(label)
        self._validate_parameters(len(self._features), len(self._features[0]))

        tasks = self._make_tasks(len(self._features))
        self.bootstrap_indices_ = [indices for indices, _ in tasks]
        parameters = self._tree_parameters()
        if self.n_jobs == 1:
            self.estimators_ = [
                _build_tree(self._features, self._targets, parameters, task)
                for task in tasks
            ]
        else:
            requested = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
            workers = min(self.n_estimators, requested or 1)
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(self._features, self._targets, parameters),
            ) as executor:
                self.estimators_ = list(executor.map(_fit_tree_in_worker, tasks))

        self._collect_feature_importances(len(self._features[0]))
        self._collect_oob_estimates()
        return self

    def _aligned_probabilities(self, tree, X):
        tree_probabilities = tree.predict_proba(X)
        positions = {label: index for index, label in enumerate(tree.classes)}
        return [
            [row[positions[label]] if label in positions else 0.0 for label in self.classes]
            for row in tree_probabilities
        ]

    def _tree_votes(self, tree, X):
        if self.voting == "soft":
            return self._aligned_probabilities(tree, X)
        predictions = tree.predict(X)
        return [
            [1.0 if prediction == label else 0.0 for label in self.classes]
            for prediction in predictions
        ]

    def predict_proba(self, X):
        samples = [list(row) for row in X]
        totals = [[0.0] * len(self.classes) for _ in samples]
        for tree in self.estimators_:
            for row_total, row_vote in zip(totals, self._tree_votes(tree, samples)):
                for index, value in enumerate(row_vote):
                    row_total[index] += value
        return [
            [value / len(self.estimators_) for value in row]
            for row in totals
        ]

    def predict(self, X):
        return [
            self.classes[max(range(len(self.classes)), key=row.__getitem__)]
            for row in self.predict_proba(X)
        ]

    def _collect_feature_importances(self, feature_count):
        values = [0.0] * feature_count
        for tree in self.estimators_:
            for index, importance in enumerate(tree.feature_importances_):
                values[index] += importance
        total = sum(values)
        self.feature_importances_ = (
            [value / total for value in values]
            if total > 0.0
            else values
        )

    def _collect_oob_estimates(self):
        self.oob_decision_function_ = [None] * len(self._targets)
        self.oob_score_ = None
        self.oob_coverage_ = 0.0
        if not self.bootstrap or not self.oob_score:
            return

        totals = [[0.0] * len(self.classes) for _ in self._targets]
        counts = [0] * len(self._targets)
        all_indices = set(range(len(self._targets)))
        for tree, sampled in zip(self.estimators_, self.bootstrap_indices_):
            # 一棵树只能评价未进入该树 Bootstrap 样本的训练记录。
            oob_indices = sorted(all_indices - set(sampled))
            if not oob_indices:
                continue
            samples = [self._features[index] for index in oob_indices]
            votes = self._tree_votes(tree, samples)
            for sample_index, vote in zip(oob_indices, votes):
                counts[sample_index] += 1
                for class_index, value in enumerate(vote):
                    totals[sample_index][class_index] += value

        correct = 0
        evaluated = 0
        for index, count in enumerate(counts):
            if count == 0:
                continue
            probabilities = [value / count for value in totals[index]]
            self.oob_decision_function_[index] = probabilities
            prediction = self.classes[
                max(range(len(self.classes)), key=probabilities.__getitem__)
            ]
            correct += prediction == self._targets[index]
            evaluated += 1
        if evaluated:
            self.oob_score_ = correct / evaluated
            self.oob_coverage_ = evaluated / len(self._targets)
