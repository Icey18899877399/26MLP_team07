"""不依赖机器学习库的优化 GBDT 回归。"""

import math
import random

from .gbdt_regression import _RegressionTreeNode


class _FastRegressionTreeScratch:
    """用排序扫描和累计统计量寻找回归树分裂点。"""

    def __init__(
        self,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        l2_regularization,
        max_features,
        generator,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.generator = generator
        self.root = None
        self.raw_feature_importances_ = []

    def _leaf_value(self, targets):
        return sum(targets) / (len(targets) + self.l2_regularization)

    def _candidate_features(self, feature_count):
        if self.max_features is None:
            count = feature_count
        elif isinstance(self.max_features, float):
            count = max(1, math.ceil(feature_count * self.max_features))
        else:
            count = max(1, min(int(self.max_features), feature_count))
        if count == feature_count:
            return list(range(feature_count))
        return self.generator.sample(range(feature_count), count)

    def _best_split(self, features, targets):
        sample_count = len(targets)
        total_sum = sum(targets)
        parent_score = total_sum * total_sum / (
            sample_count + self.l2_regularization
        )
        best_gain = self.min_impurity_decrease
        best_split = None

        for feature_index in self._candidate_features(len(features[0])):
            ordered = sorted(
                (row[feature_index], targets[index], index)
                for index, row in enumerate(features)
            )
            left_sum = 0.0
            for position in range(sample_count - 1):
                value, target, _ = ordered[position]
                left_sum += target
                left_count = position + 1
                right_count = sample_count - left_count
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                    or value == ordered[position + 1][0]
                ):
                    continue

                right_sum = total_sum - left_sum
                split_score = (
                    left_sum * left_sum
                    / (left_count + self.l2_regularization)
                    + right_sum * right_sum
                    / (right_count + self.l2_regularization)
                )
                gain = split_score - parent_score
                if gain > best_gain:
                    threshold = (value + ordered[position + 1][0]) / 2.0
                    best_gain = gain
                    best_split = (
                        feature_index,
                        threshold,
                        gain,
                        ordered,
                        left_count,
                    )
        if best_split is None:
            return None
        feature_index, threshold, gain, ordered, left_count = best_split
        ordered_indices = [item[2] for item in ordered]
        return (
            feature_index,
            threshold,
            gain,
            ordered_indices[:left_count],
            ordered_indices[left_count:],
        )

    def _build_tree(self, features, targets, depth):
        node = _RegressionTreeNode(
            value=self._leaf_value(targets),
            sample_count=len(targets),
        )
        depth_limit = self.max_depth is not None and depth >= self.max_depth
        if (
            depth_limit
            or len(targets) < self.min_samples_split
            or len(targets) < 2 * self.min_samples_leaf
        ):
            return node

        split = self._best_split(features, targets)
        if split is None:
            return node

        feature_index, threshold, gain, left_indices, right_indices = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.gain = gain
        node.left = self._build_tree(
            [features[index] for index in left_indices],
            [targets[index] for index in left_indices],
            depth + 1,
        )
        node.right = self._build_tree(
            [features[index] for index in right_indices],
            [targets[index] for index in right_indices],
            depth + 1,
        )
        return node

    def _collect_importances(self, node):
        if node.is_leaf:
            return
        self.raw_feature_importances_[node.feature_index] += node.gain
        self._collect_importances(node.left)
        self._collect_importances(node.right)

    def fit(self, X, y):
        self.raw_feature_importances_ = [0.0] * len(X[0])
        self.root = self._build_tree([list(row) for row in X], list(y), depth=0)
        self._collect_importances(self.root)
        return self

    @staticmethod
    def _predict_one(sample, node):
        while not node.is_leaf:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return [self._predict_one(sample, self.root) for sample in X]


class OptimizedGBDTRegressorScratch:
    """加入高效分裂、随机采样、L2 正则化和早停的 GBDT 回归。"""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        l2_regularization=0.0,
        subsample=1.0,
        max_features=None,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.l2_regularization = l2_regularization
        self.subsample = subsample
        self.max_features = max_features
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

        self.init_prediction_ = 0.0
        self.estimators_ = []
        self.sample_indices_ = []
        self.training_indices_ = []
        self.validation_indices_ = []
        self.train_loss_ = []
        self.validation_loss_ = []
        self.n_estimators_ = 0
        self.feature_importances_ = []

    @staticmethod
    def _subset(values, indices):
        return [values[index] for index in indices]

    @staticmethod
    def _mse(targets, predictions):
        return sum(
            (target - prediction) ** 2
            for target, prediction in zip(targets, predictions)
        ) / len(targets)

    def _training_split(self, sample_count, generator):
        indices = list(range(sample_count))
        if self.n_iter_no_change is None:
            return indices, []
        generator.shuffle(indices)
        validation_count = min(
            sample_count - 1,
            max(1, round(sample_count * self.validation_fraction)),
        )
        return indices[validation_count:], indices[:validation_count]

    def _new_tree(self, generator):
        return _FastRegressionTreeScratch(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            generator=generator,
        )

    def fit(self, X, y):
        generator = random.Random(self.random_state)
        train_indices, validation_indices = self._training_split(len(y), generator)
        self.training_indices_ = list(train_indices)
        self.validation_indices_ = list(validation_indices)
        train_X = self._subset(X, train_indices)
        train_y = self._subset(y, train_indices)
        validation_X = self._subset(X, validation_indices)
        validation_y = self._subset(y, validation_indices)

        self.init_prediction_ = sum(train_y) / len(train_y)
        train_predictions = [self.init_prediction_] * len(train_y)
        validation_predictions = [self.init_prediction_] * len(validation_y)
        self.estimators_ = []
        self.sample_indices_ = []
        self.train_loss_ = []
        self.validation_loss_ = []

        best_validation_loss = float("inf")
        best_round = 0
        rounds_without_improvement = 0

        for _ in range(self.n_estimators):
            residuals = [
                target - prediction
                for target, prediction in zip(train_y, train_predictions)
            ]
            sample_count = max(1, round(len(train_y) * self.subsample))
            local_indices = (
                list(range(len(train_y)))
                if sample_count == len(train_y)
                else generator.sample(range(len(train_y)), sample_count)
            )
            tree = self._new_tree(generator).fit(
                self._subset(train_X, local_indices),
                self._subset(residuals, local_indices),
            )
            train_updates = tree.predict(train_X)
            train_predictions = [
                prediction + self.learning_rate * update
                for prediction, update in zip(train_predictions, train_updates)
            ]

            self.estimators_.append(tree)
            self.sample_indices_.append(
                [train_indices[index] for index in local_indices]
            )
            self.train_loss_.append(self._mse(train_y, train_predictions))

            if validation_indices:
                validation_updates = tree.predict(validation_X)
                validation_predictions = [
                    prediction + self.learning_rate * update
                    for prediction, update in zip(
                        validation_predictions,
                        validation_updates,
                    )
                ]
                validation_loss = self._mse(validation_y, validation_predictions)
                self.validation_loss_.append(validation_loss)
                if validation_loss < best_validation_loss - self.tol:
                    best_validation_loss = validation_loss
                    best_round = len(self.estimators_)
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.n_iter_no_change:
                        self.estimators_ = self.estimators_[:best_round]
                        self.sample_indices_ = self.sample_indices_[:best_round]
                        self.train_loss_ = self.train_loss_[:best_round]
                        self.validation_loss_ = self.validation_loss_[:best_round]
                        break

        self.n_estimators_ = len(self.estimators_)
        raw_importances = [0.0] * len(X[0])
        for tree in self.estimators_:
            raw_importances = [
                total + gain
                for total, gain in zip(
                    raw_importances,
                    tree.raw_feature_importances_,
                )
            ]
        total_gain = sum(raw_importances)
        self.feature_importances_ = (
            [gain / total_gain for gain in raw_importances]
            if total_gain > 0.0
            else raw_importances
        )
        return self

    def staged_predict(self, X):
        predictions = [self.init_prediction_] * len(X)
        for tree in self.estimators_:
            updates = tree.predict(X)
            predictions = [
                prediction + self.learning_rate * update
                for prediction, update in zip(predictions, updates)
            ]
            yield list(predictions)

    def predict(self, X):
        predictions = [self.init_prediction_] * len(X)
        for stage in self.staged_predict(X):
            predictions = stage
        return predictions
