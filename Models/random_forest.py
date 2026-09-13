"""手写随机森林分类器基础版。"""

import math
import random

from .cart_decision_tree import CARTClassifierScratch


class _RandomFeatureCART(CARTClassifierScratch):
    """在每个节点随机抽取候选特征的基础 CART。"""

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
    ):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self._generator = random.Random(random_state)

    def _feature_count(self, total):
        if self.max_features is None:
            return total
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(total)))
        if self.max_features == "log2":
            return max(1, int(math.log2(total)))
        if isinstance(self.max_features, float):
            return max(1, math.ceil(total * self.max_features))
        return min(total, int(self.max_features))

    def _best_split(self, features, targets):
        parent_impurity = self._gini(targets)
        sample_count = len(targets)
        feature_count = len(features[0])
        # 每个节点重新抽取候选特征，这是随机森林降低树间相关性的关键。
        selected = self._generator.sample(
            range(feature_count),
            self._feature_count(feature_count),
        )
        best_gain = 0.0
        best_split = None

        for feature_index in selected:
            values = sorted(set(row[feature_index] for row in features))
            for left_value, right_value in zip(values, values[1:]):
                threshold = (left_value + right_value) / 2.0
                left_indices = [
                    index
                    for index, row in enumerate(features)
                    if row[feature_index] <= threshold
                ]
                right_indices = [
                    index
                    for index, row in enumerate(features)
                    if row[feature_index] > threshold
                ]
                left_targets = [targets[index] for index in left_indices]
                right_targets = [targets[index] for index in right_indices]
                child_impurity = (
                    len(left_targets) / sample_count * self._gini(left_targets)
                    + len(right_targets) / sample_count * self._gini(right_targets)
                )
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_split = (
                        feature_index,
                        threshold,
                        gain,
                        left_indices,
                        right_indices,
                    )
        return best_split


class RandomForestClassifierScratch:
    """使用 Bootstrap、随机特征和多数投票构建分类随机森林。"""

    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.classes = []
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.feature_importances_ = []

    def _validate_parameters(self, feature_count):
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.max_features in (None, "sqrt", "log2"):
            return
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError("fractional max_features must be in (0, 1]")
            return
        if not isinstance(self.max_features, int) or not 1 <= self.max_features <= feature_count:
            raise ValueError("integer max_features must be between 1 and the feature count")

    def fit(self, X, y):
        """在有放回抽样的数据上训练多棵随机特征 CART。"""
        features = [list(row) for row in X]
        targets = list(y)
        self._validate_parameters(len(features[0]))
        self.classes = []
        for label in targets:
            if label not in self.classes:
                self.classes.append(label)

        generator = random.Random(self.random_state)
        self.estimators_ = []
        self.bootstrap_indices_ = []
        for _ in range(self.n_estimators):
            # Bootstrap 保持样本量不变，但允许同一样本被重复抽中。
            indices = [generator.randrange(len(features)) for _ in range(len(features))]
            tree_seed = generator.randrange(2**32)
            tree = _RandomFeatureCART(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit([features[index] for index in indices], [targets[index] for index in indices])
            self.estimators_.append(tree)
            self.bootstrap_indices_.append(indices)

        self._collect_feature_importances(len(features[0]))
        return self

    def _collect_feature_importances(self, feature_count):
        importances = [0.0] * feature_count
        for tree in self.estimators_:
            for index, value in enumerate(tree.feature_importances_):
                importances[index] += value
        total = sum(importances)
        self.feature_importances_ = (
            [value / total for value in importances]
            if total > 0.0
            else importances
        )

    def predict_proba(self, X):
        """返回各类别在所有树硬投票中所占的比例。"""
        probabilities = []
        for sample in X:
            counts = [0] * len(self.classes)
            for tree in self.estimators_:
                prediction = tree.predict([sample])[0]
                counts[self.classes.index(prediction)] += 1
            probabilities.append([count / len(self.estimators_) for count in counts])
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return [
            self.classes[max(range(len(self.classes)), key=row.__getitem__)]
            for row in probabilities
        ]
