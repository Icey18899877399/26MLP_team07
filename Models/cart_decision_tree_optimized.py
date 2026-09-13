"""带快速划分、类别权重和剪枝的手写 CART 分类树。"""

import math
import random


class _TreeNode:
    def __init__(
        self,
        prediction,
        probabilities,
        impurity,
        sample_count,
        weighted_count,
    ):
        self.prediction = prediction
        self.probabilities = probabilities
        self.impurity = impurity
        self.sample_count = sample_count
        self.weighted_count = weighted_count
        self.feature_index = None
        self.threshold = None
        self.gain = 0.0
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return self.left is None


class OptimizedCARTClassifierScratch:
    """通过累计计数扫描、预剪枝和代价复杂度剪枝优化 CART。"""

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        max_features=None,
        class_weight=None,
        ccp_alpha=0.0,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        self.classes = []
        self.root = None
        self.tree_depth_ = 0
        self.n_leaves_ = 0
        self.feature_importances_ = []
        self._generator = None

    @staticmethod
    def _gini_from_counts(counts, total_weight):
        if total_weight == 0.0:
            return 0.0
        return 1.0 - sum((value / total_weight) ** 2 for value in counts)

    def _weighted_counts(self, targets, weights):
        return [
            sum(weight for target, weight in zip(targets, weights) if target == label)
            for label in self.classes
        ]

    def _sample_feature_indices(self, feature_count):
        if self.max_features is None:
            return list(range(feature_count))
        if self.max_features == "sqrt":
            selected_count = max(1, int(math.sqrt(feature_count)))
        elif self.max_features == "log2":
            selected_count = max(1, int(math.log2(feature_count)))
        else:
            selected_count = min(feature_count, int(self.max_features))
        return sorted(self._generator.sample(range(feature_count), selected_count))

    def _best_split(self, features, targets, weights, parent_impurity):
        sample_count = len(targets)
        total_weight = sum(weights)
        best_gain = 0.0
        best_split = None

        for feature_index in self._sample_feature_indices(len(features[0])):
            ordered = sorted(
                range(sample_count),
                key=lambda index: features[index][feature_index],
            )
            total_counts = self._weighted_counts(targets, weights)
            left_counts = [0.0] * len(self.classes)
            left_weight = 0.0

            for position in range(sample_count - 1):
                index = ordered[position]
                class_index = self.classes.index(targets[index])
                left_counts[class_index] += weights[index]
                left_weight += weights[index]
                left_size = position + 1
                right_size = sample_count - left_size
                current_value = features[index][feature_index]
                next_value = features[ordered[position + 1]][feature_index]

                if (
                    current_value == next_value
                    or left_size < self.min_samples_leaf
                    or right_size < self.min_samples_leaf
                ):
                    continue

                right_weight = total_weight - left_weight
                right_counts = [
                    total - left
                    for total, left in zip(total_counts, left_counts)
                ]
                child_impurity = (
                    left_weight / total_weight
                    * self._gini_from_counts(left_counts, left_weight)
                    + right_weight / total_weight
                    * self._gini_from_counts(right_counts, right_weight)
                )
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    threshold = (current_value + next_value) / 2.0
                    best_gain = gain
                    best_split = (feature_index, threshold, gain)

        if best_split is None:
            return None
        feature_index, threshold, gain = best_split
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
        return feature_index, threshold, gain, left_indices, right_indices

    def _build_tree(self, features, targets, weights, depth):
        weighted_counts = self._weighted_counts(targets, weights)
        weighted_count = sum(weighted_counts)
        probabilities = [value / weighted_count for value in weighted_counts]
        best_class_index = max(
            range(len(self.classes)),
            key=lambda index: probabilities[index],
        )
        node = _TreeNode(
            prediction=self.classes[best_class_index],
            probabilities=probabilities,
            impurity=self._gini_from_counts(weighted_counts, weighted_count),
            sample_count=len(targets),
            weighted_count=weighted_count,
        )
        depth_limit_reached = self.max_depth is not None and depth >= self.max_depth
        if (
            node.impurity == 0.0
            or depth_limit_reached
            or len(targets) < self.min_samples_split
            or len(targets) < 2 * self.min_samples_leaf
        ):
            return node

        split = self._best_split(features, targets, weights, node.impurity)
        if split is None or split[2] < self.min_impurity_decrease:
            return node

        feature_index, threshold, gain, left_indices, right_indices = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.gain = gain
        node.left = self._build_tree(
            [features[index] for index in left_indices],
            [targets[index] for index in left_indices],
            [weights[index] for index in left_indices],
            depth + 1,
        )
        node.right = self._build_tree(
            [features[index] for index in right_indices],
            [targets[index] for index in right_indices],
            [weights[index] for index in right_indices],
            depth + 1,
        )
        return node

    def _prune(self, node, total_weight):
        if node.is_leaf:
            return node.weighted_count / total_weight * node.impurity, 1

        left_risk, left_leaves = self._prune(node.left, total_weight)
        right_risk, right_leaves = self._prune(node.right, total_weight)
        subtree_risk = left_risk + right_risk
        leaf_count = left_leaves + right_leaves
        collapsed_risk = node.weighted_count / total_weight * node.impurity

        if (
            collapsed_risk + self.ccp_alpha
            <= subtree_risk + self.ccp_alpha * leaf_count
        ):
            node.feature_index = None
            node.threshold = None
            node.gain = 0.0
            node.left = None
            node.right = None
            return collapsed_risk, 1
        return subtree_risk, leaf_count

    def _resolve_sample_weights(self, targets):
        if self.class_weight == "balanced":
            counts = {label: targets.count(label) for label in self.classes}
            class_weights = {
                label: len(targets) / (len(self.classes) * counts[label])
                for label in self.classes
            }
        elif isinstance(self.class_weight, dict):
            class_weights = self.class_weight
        else:
            class_weights = {label: 1.0 for label in self.classes}
        return [class_weights[label] for label in targets]

    def _collect_metadata(self):
        raw_importances = [0.0] * len(self.feature_importances_)

        def visit(node, depth):
            if node.is_leaf:
                return depth, 1
            raw_importances[node.feature_index] += node.gain * node.weighted_count
            left_depth, left_leaves = visit(node.left, depth + 1)
            right_depth, right_leaves = visit(node.right, depth + 1)
            return max(left_depth, right_depth), left_leaves + right_leaves

        self.tree_depth_, self.n_leaves_ = visit(self.root, 0)
        total = sum(raw_importances)
        self.feature_importances_ = (
            [value / total for value in raw_importances]
            if total > 0.0
            else raw_importances
        )

    def fit(self, X, y):
        self.classes = []
        for label in y:
            if label not in self.classes:
                self.classes.append(label)
        self._generator = random.Random(self.random_state)
        features = [list(row) for row in X]
        targets = list(y)
        weights = self._resolve_sample_weights(targets)
        self.feature_importances_ = [0.0] * len(features[0])
        self.root = self._build_tree(features, targets, weights, depth=0)
        if self.ccp_alpha > 0.0:
            self._prune(self.root, sum(weights))
        self._collect_metadata()
        return self

    @staticmethod
    def _leaf_for_sample(sample, node):
        while not node.is_leaf:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict_proba(self, X):
        return [self._leaf_for_sample(sample, self.root).probabilities for sample in X]

    def predict(self, X):
        return [self._leaf_for_sample(sample, self.root).prediction for sample in X]
