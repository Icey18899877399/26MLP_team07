"""不依赖机器学习库的基础 GBDT 回归。"""


class _RegressionTreeNode:
    def __init__(self, value, sample_count):
        self.value = value
        self.sample_count = sample_count
        self.feature_index = None
        self.threshold = None
        self.gain = 0.0
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return self.left is None


class _RegressionTreeScratch:
    """以平方误差下降量选择分裂点的手写 CART 回归树。"""

    def __init__(self, max_depth=2, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.raw_feature_importances_ = []

    @staticmethod
    def _squared_error(targets):
        mean = sum(targets) / len(targets)
        return sum((target - mean) ** 2 for target in targets)

    def _best_split(self, features, targets):
        parent_error = self._squared_error(targets)
        best_gain = 0.0
        best_split = None

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
                gain = (
                    parent_error
                    - self._squared_error(left_targets)
                    - self._squared_error(right_targets)
                )
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
        node = _RegressionTreeNode(
            value=sum(targets) / len(targets),
            sample_count=len(targets),
        )
        depth_limit = self.max_depth is not None and depth >= self.max_depth
        if depth_limit or len(targets) < self.min_samples_split:
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


class GBDTRegressorScratch:
    """使用回归树逐轮拟合平方损失的负梯度（残差）。"""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        min_samples_split=2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.init_prediction_ = 0.0
        self.estimators_ = []
        self.train_loss_ = []
        self.feature_importances_ = []

    def fit(self, X, y):
        self.init_prediction_ = sum(y) / len(y)
        predictions = [self.init_prediction_] * len(y)
        self.estimators_ = []
        self.train_loss_ = []
        raw_importances = [0.0] * len(X[0])

        for _ in range(self.n_estimators):
            residuals = [
                target - prediction
                for target, prediction in zip(y, predictions)
            ]
            tree = _RegressionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X, residuals)
            updates = tree.predict(X)
            predictions = [
                prediction + self.learning_rate * update
                for prediction, update in zip(predictions, updates)
            ]
            self.estimators_.append(tree)
            self.train_loss_.append(
                sum(
                    (target - prediction) ** 2
                    for target, prediction in zip(y, predictions)
                )
                / len(y)
            )
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

    def predict(self, X):
        predictions = [self.init_prediction_] * len(X)
        for tree in self.estimators_:
            updates = tree.predict(X)
            predictions = [
                prediction + self.learning_rate * update
                for prediction, update in zip(predictions, updates)
            ]
        return predictions
