"""不依赖机器学习库的扩展孤立森林。"""

import math
import random

from .isolation_forest import average_path_length, contamination_threshold


class _ExtendedIsolationNode:
    """随机超平面孤立树节点。"""

    def __init__(self, size, normal=None, offset=None, left=None, right=None):
        self.size = size
        self.normal = normal
        self.offset = offset
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None


class OptimizedIsolationForestScratch:
    """使用 EIF 随机超平面和稀疏特征子采样实现优化孤立森林。

    n_split_candidates 表示节点生成有效随机超平面的最大尝试次数，首个
    有效切分会被直接采用，不使用均衡性规则改变 Isolation Forest 的随机性。
    score_samples 越大表示越异常；predict 使用 1/-1 表示正常/异常。
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        contamination=0.1,
        max_depth=None,
        max_features=1.0,
        n_split_candidates=3,
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
        if not (
            isinstance(max_features, int) and max_features >= 1
            or isinstance(max_features, float) and 0.0 < max_features <= 1.0
        ):
            raise ValueError("max_features must be a positive integer or a fraction in (0, 1]")
        if not isinstance(n_split_candidates, int) or n_split_candidates < 1:
            raise ValueError("n_split_candidates must be a positive integer")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_split_candidates = n_split_candidates
        self.random_state = random_state

        self.estimators_ = []
        self.estimators_features_ = []
        self.scores_ = []
        self.labels_ = []
        self.threshold_ = None
        self.max_samples_ = None
        self.max_features_ = None
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

    @staticmethod
    def _resolve_count(value, total):
        if isinstance(value, float):
            return min(total, max(1, math.ceil(value * total)))
        return min(total, value)

    @staticmethod
    def _projection(row, normal):
        return sum(row[feature] * weight for feature, weight in normal)

    def _candidate_split(self, X, indices, features, generator):
        normal = [
            (feature, generator.gauss(0.0, 1.0))
            for feature in features
        ]
        intercept_point = []
        for feature in features:
            values = [X[index][feature] for index in indices]
            minimum, maximum = min(values), max(values)
            intercept_point.append(
                minimum + generator.random() * (maximum - minimum)
            )
        offset = sum(
            value * weight
            for value, (_, weight) in zip(intercept_point, normal)
        )
        projections = [self._projection(X[index], normal) for index in indices]
        left = [
            index
            for index, projection in zip(indices, projections)
            if projection < offset
        ]
        right = [
            index
            for index, projection in zip(indices, projections)
            if projection >= offset
        ]
        if not left or not right:
            return None
        return normal, offset, left, right

    def _build_tree(self, X, indices, features, depth, generator):
        if len(indices) <= 1 or depth >= self.max_depth_:
            return _ExtendedIsolationNode(len(indices))

        candidate = None
        for _ in range(self.n_split_candidates):
            candidate = self._candidate_split(X, indices, features, generator)
            if candidate is not None:
                break
        if candidate is None:
            return _ExtendedIsolationNode(len(indices))

        normal, offset, left_indices, right_indices = candidate
        return _ExtendedIsolationNode(
            size=len(indices),
            normal=normal,
            offset=offset,
            left=self._build_tree(
                X, left_indices, features, depth + 1, generator
            ),
            right=self._build_tree(
                X, right_indices, features, depth + 1, generator
            ),
        )

    def _path_length(self, sample, node, depth=0):
        if node.is_leaf:
            return depth + average_path_length(node.size)
        child = (
            node.left
            if self._projection(sample, node.normal) < node.offset
            else node.right
        )
        return self._path_length(sample, child, depth + 1)

    def fit(self, X):
        samples = self._validate_matrix(X)
        self._n_features_ = len(samples[0])
        self.max_samples_ = self._resolve_count(self.max_samples, len(samples))
        self.max_features_ = self._resolve_count(
            self.max_features,
            self._n_features_,
        )
        self.max_depth_ = (
            self.max_depth
            if self.max_depth is not None
            else math.ceil(math.log2(self.max_samples_))
            if self.max_samples_ > 1
            else 0
        )

        generator = random.Random(self.random_state)
        sample_pool = list(range(len(samples)))
        feature_pool = list(range(self._n_features_))
        self.estimators_ = []
        self.estimators_features_ = []
        for _ in range(self.n_estimators):
            indices = generator.sample(sample_pool, self.max_samples_)
            features = sorted(generator.sample(feature_pool, self.max_features_))
            self.estimators_features_.append(features)
            self.estimators_.append(
                self._build_tree(samples, indices, features, 0, generator)
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
        """计算异常分数；较大值表示更可能是异常样本。"""
        if not self._is_fitted:
            raise ValueError("fit must be called before score_samples")
        samples = self._validate_matrix(X, self._n_features_)
        normalizer = average_path_length(self.max_samples_)
        if normalizer == 0.0:
            return [0.5 for _ in samples]
        return [
            2.0
            ** (
                -sum(self._path_length(sample, tree) for tree in self.estimators_)
                / len(self.estimators_)
                / normalizer
            )
            for sample in samples
        ]

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
