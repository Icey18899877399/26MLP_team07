"""不依赖机器学习库的基础孤立森林。"""

import math
import random


def average_path_length(sample_count):
    """返回包含 sample_count 个样本的二叉搜索树平均失败路径长度。"""
    if sample_count <= 1:
        return 0.0
    if sample_count == 2:
        return 1.0
    harmonic = sum(1.0 / value for value in range(1, sample_count))
    return 2.0 * harmonic - 2.0 * (sample_count - 1) / sample_count


def contamination_threshold(scores, contamination):
    """选择分数阈值；边界同分时不任意拆分同分样本。"""
    ordered = sorted(scores, reverse=True)
    anomaly_count = max(1, math.ceil(contamination * len(ordered)))
    upper = ordered[anomaly_count - 1]
    if anomaly_count == len(ordered):
        return upper
    lower = ordered[anomaly_count]
    return (upper + lower) / 2.0 if upper > lower else upper


class _IsolationNode:
    """孤立树节点；叶节点只保存到达该处的样本数。"""

    def __init__(self, size, feature=None, split=None, left=None, right=None):
        self.size = size
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None


class IsolationForestScratch:
    """使用随机轴向切分实现经典 Isolation Forest。

    score_samples 返回异常分数，数值越大表示样本越异常；predict 使用
    1 表示正常样本，-1 表示异常样本。
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        contamination=0.1,
        max_depth=None,
        random_state=None,
    ):
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if not (
            isinstance(max_samples, int) and max_samples >= 1
            or isinstance(max_samples, float) and 0.0 < max_samples <= 1.0
        ):
            raise ValueError("max_samples must be a positive integer or a fraction in (0, 1]")
        if not isinstance(contamination, (int, float)) or not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if max_depth is not None and (
            not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_depth = max_depth
        self.random_state = random_state

        self.estimators_ = []
        self.scores_ = []
        self.labels_ = []
        self.threshold_ = None
        self.max_samples_ = None
        self.max_depth_ = None
        self._n_features_ = None
        self._is_fitted = False

    @staticmethod
    def _validate_matrix(X, expected_features=None):
        samples = [list(row) for row in X]
        if not samples or not samples[0]:
            raise ValueError("X must be a non-empty matrix")
        feature_count = len(samples[0])
        if expected_features is not None and feature_count != expected_features:
            raise ValueError("X has a different feature count from the fitted data")
        for row in samples:
            if len(row) != feature_count:
                raise ValueError("X must be rectangular")
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in row
            ):
                raise ValueError("X must contain only finite numeric values")
        return samples

    def _resolved_sample_count(self, total):
        if isinstance(self.max_samples, float):
            return min(total, max(1, math.ceil(self.max_samples * total)))
        return min(total, self.max_samples)

    def _build_tree(self, X, indices, depth, generator):
        if len(indices) <= 1 or depth >= self.max_depth_:
            return _IsolationNode(len(indices))

        varying_features = []
        ranges = {}
        for feature in range(self._n_features_):
            values = [X[index][feature] for index in indices]
            minimum, maximum = min(values), max(values)
            if minimum < maximum:
                varying_features.append(feature)
                ranges[feature] = (minimum, maximum)
        if not varying_features:
            return _IsolationNode(len(indices))

        feature = generator.choice(varying_features)
        minimum, maximum = ranges[feature]
        split = minimum + generator.random() * (maximum - minimum)
        left_indices = [index for index in indices if X[index][feature] < split]
        right_indices = [index for index in indices if X[index][feature] >= split]
        if not left_indices or not right_indices:
            return _IsolationNode(len(indices))

        return _IsolationNode(
            size=len(indices),
            feature=feature,
            split=split,
            left=self._build_tree(X, left_indices, depth + 1, generator),
            right=self._build_tree(X, right_indices, depth + 1, generator),
        )

    @staticmethod
    def _path_length(sample, node, depth=0):
        if node.is_leaf:
            return depth + average_path_length(node.size)
        child = node.left if sample[node.feature] < node.split else node.right
        return IsolationForestScratch._path_length(sample, child, depth + 1)

    def fit(self, X):
        samples = self._validate_matrix(X)
        self._n_features_ = len(samples[0])
        self.max_samples_ = self._resolved_sample_count(len(samples))
        self.max_depth_ = (
            self.max_depth
            if self.max_depth is not None
            else math.ceil(math.log2(self.max_samples_))
            if self.max_samples_ > 1
            else 0
        )

        generator = random.Random(self.random_state)
        self.estimators_ = []
        all_indices = list(range(len(samples)))
        for _ in range(self.n_estimators):
            sample_indices = generator.sample(all_indices, self.max_samples_)
            self.estimators_.append(
                self._build_tree(samples, sample_indices, 0, generator)
            )

        self._is_fitted = True
        self.scores_ = self.score_samples(samples)
        self.threshold_ = contamination_threshold(
            self.scores_,
            self.contamination,
        )
        self.labels_ = [
            -1 if score > self.threshold_ else 1 for score in self.scores_
        ]
        return self

    def score_samples(self, X):
        """计算异常分数；较短的平均路径对应较高的异常分数。"""
        if not self._is_fitted:
            raise ValueError("fit must be called before score_samples")
        samples = self._validate_matrix(X, self._n_features_)
        normalizer = average_path_length(self.max_samples_)
        if normalizer == 0.0:
            return [0.5 for _ in samples]

        scores = []
        for sample in samples:
            mean_path = sum(
                self._path_length(sample, tree) for tree in self.estimators_
            ) / len(self.estimators_)
            scores.append(2.0 ** (-mean_path / normalizer))
        return scores

    def decision_function(self, X):
        """返回 threshold - anomaly_score，正值表示正常。"""
        return [self.threshold_ - score for score in self.score_samples(X)]

    def predict(self, X):
        return [
            -1 if score > self.threshold_ else 1
            for score in self.score_samples(X)
        ]

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
