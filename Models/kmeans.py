"""不依赖机器学习库的基础 K-Means 聚类。"""

import math
import random


class KMeansScratch:
    """使用随机初始化和 Lloyd 迭代的经典 K-Means。"""

    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.cluster_centers_ = []
        self.labels_ = []
        self.inertia_ = 0.0
        self.inertia_history_ = []
        self.n_iter_ = 0
        self._n_features_ = None

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

    def _nearest_center(self, sample, centers):
        return min(
            range(len(centers)),
            key=lambda index: self._squared_distance(sample, centers[index]),
        )

    def _assign(self, X, centers):
        return [self._nearest_center(sample, centers) for sample in X]

    def _recompute_centers(self, X, labels, old_centers):
        feature_count = len(X[0])
        centers = []
        for cluster in range(self.n_clusters):
            members = [sample for sample, label in zip(X, labels) if label == cluster]
            if not members:
                centers.append(list(old_centers[cluster]))
                continue
            centers.append(
                [
                    sum(sample[feature] for sample in members) / len(members)
                    for feature in range(feature_count)
                ]
            )
        return centers

    def _inertia(self, X, labels, centers):
        return sum(
            self._squared_distance(sample, centers[label])
            for sample, label in zip(X, labels)
        )

    def fit(self, X):
        samples = self._validate_matrix(X)
        if self.n_clusters > len(samples):
            raise ValueError("n_clusters cannot exceed the number of samples")
        self._n_features_ = len(samples[0])
        generator = random.Random(self.random_state)
        initial_indices = generator.sample(range(len(samples)), self.n_clusters)
        centers = [list(samples[index]) for index in initial_indices]

        self.inertia_history_ = []
        previous_labels = None
        for iteration in range(1, self.max_iter + 1):
            labels = self._assign(samples, centers)
            centers = self._recompute_centers(samples, labels, centers)
            labels = self._assign(samples, centers)
            inertia = self._inertia(samples, labels, centers)
            self.inertia_history_.append(inertia)
            if labels == previous_labels:
                break
            previous_labels = labels

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration
        return self

    def predict(self, X):
        if not self.cluster_centers_:
            raise ValueError("fit must be called before predict")
        samples = self._validate_matrix(X, self._n_features_)
        return self._assign(samples, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
