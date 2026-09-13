"""不依赖机器学习库的基础 DBSCAN 聚类。"""

from collections import deque
import math


class DBSCANScratch:
    """使用暴力半径搜索实现经典 DBSCAN。"""

    NOISE = -1
    _UNVISITED = -2

    def __init__(self, eps=0.5, min_samples=5):
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive number")
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        self.eps = eps
        self.min_samples = min_samples

        self.labels_ = []
        self.core_sample_indices_ = []
        self.components_ = []
        self.n_clusters_ = 0
        self._core_labels_ = []
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
    def _squared_distance(first, second):
        return sum((left - right) ** 2 for left, right in zip(first, second))

    def _radius_neighbors(self, X, sample_index):
        radius_squared = self.eps ** 2
        sample = X[sample_index]
        return [
            index
            for index, candidate in enumerate(X)
            if self._squared_distance(sample, candidate) <= radius_squared
        ]

    def _expand_cluster(self, X, labels, start, neighbors, cluster, core_indices):
        labels[start] = cluster
        core_indices.add(start)
        queue = deque(index for index in neighbors if index != start)
        queued = set(neighbors)

        while queue:
            current = queue.popleft()
            if labels[current] == self.NOISE:
                labels[current] = cluster
                continue
            if labels[current] != self._UNVISITED:
                continue

            labels[current] = cluster
            current_neighbors = self._radius_neighbors(X, current)
            if len(current_neighbors) < self.min_samples:
                continue

            core_indices.add(current)
            for neighbor in current_neighbors:
                if neighbor not in queued:
                    queue.append(neighbor)
                    queued.add(neighbor)

    def fit(self, X):
        samples = self._validate_matrix(X)
        self._n_features_ = len(samples[0])
        labels = [self._UNVISITED] * len(samples)
        core_indices = set()
        cluster = 0

        for sample_index in range(len(samples)):
            if labels[sample_index] != self._UNVISITED:
                continue
            neighbors = self._radius_neighbors(samples, sample_index)
            if len(neighbors) < self.min_samples:
                labels[sample_index] = self.NOISE
                continue
            self._expand_cluster(
                samples,
                labels,
                sample_index,
                neighbors,
                cluster,
                core_indices,
            )
            cluster += 1

        self.labels_ = labels
        self.core_sample_indices_ = sorted(core_indices)
        self.components_ = [list(samples[index]) for index in self.core_sample_indices_]
        self._core_labels_ = [labels[index] for index in self.core_sample_indices_]
        self.n_clusters_ = cluster
        self._is_fitted = True
        return self

    def predict(self, X):
        """将新样本分配给 eps 范围内最近的核心样本，否则标记为噪声。"""
        if not self._is_fitted:
            raise ValueError("fit must be called before predict")
        samples = self._validate_matrix(X, self._n_features_)
        radius_squared = self.eps ** 2
        predictions = []
        for sample in samples:
            best_distance = None
            best_label = self.NOISE
            for core, label in zip(self.components_, self._core_labels_):
                distance = self._squared_distance(sample, core)
                if distance <= radius_squared and (
                    best_distance is None or distance < best_distance
                ):
                    best_distance = distance
                    best_label = label
            predictions.append(best_label)
        return predictions

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
