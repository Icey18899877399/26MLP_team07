"""手写 CART 分类决策树。"""


class _TreeNode:
    def __init__(self, prediction, probabilities, impurity, sample_count):
        self.prediction = prediction
        self.probabilities = probabilities
        self.impurity = impurity
        self.sample_count = sample_count
        self.feature_index = None
        self.threshold = None
        self.gain = 0.0
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return self.left is None


class CARTClassifierScratch:
    """使用基尼不纯度和穷举阈值构建二叉分类树。"""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.classes = []
        self.root = None
        self.tree_depth_ = 0
        self.n_leaves_ = 0
        self.feature_importances_ = []

    def _probabilities(self, targets):
        return [targets.count(label) / len(targets) for label in self.classes]

    @staticmethod
    def _gini(targets):
        return 1.0 - sum(
            (targets.count(label) / len(targets)) ** 2
            for label in set(targets)
        )

    def _best_split(self, features, targets):
        parent_impurity = self._gini(targets)
        best_gain = 0.0
        best_split = None
        sample_count = len(targets)

        for feature_index in range(len(features[0])):
            values = sorted(set(row[feature_index] for row in features))
            thresholds = [
                (left + right) / 2.0
                for left, right in zip(values, values[1:])
            ]
            for threshold in thresholds:
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

    def _build_tree(self, features, targets, depth):
        probabilities = self._probabilities(targets)
        best_class_index = max(
            range(len(self.classes)),
            key=lambda index: probabilities[index],
        )
        node = _TreeNode(
            prediction=self.classes[best_class_index],
            probabilities=probabilities,
            impurity=self._gini(targets),
            sample_count=len(targets),
        )
        depth_limit_reached = self.max_depth is not None and depth >= self.max_depth
        if (
            node.impurity == 0.0
            or depth_limit_reached
            or len(targets) < self.min_samples_split
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

    def _collect_metadata(self):
        raw_importances = [0.0] * len(self.feature_importances_)

        def visit(node, depth):
            if node.is_leaf:
                return depth, 1
            raw_importances[node.feature_index] += node.gain * node.sample_count
            left_depth, left_leaves = visit(node.left, depth + 1)
            right_depth, right_leaves = visit(node.right, depth + 1)
            return max(left_depth, right_depth), left_leaves + right_leaves

        self.tree_depth_, self.n_leaves_ = visit(self.root, 0)
        total = sum(raw_importances)
        if total > 0.0:
            self.feature_importances_ = [value / total for value in raw_importances]

    def fit(self, X, y):
        self.classes = []
        for label in y:
            if label not in self.classes:
                self.classes.append(label)
        self.feature_importances_ = [0.0] * len(X[0])
        self.root = self._build_tree([list(row) for row in X], list(y), depth=0)
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
