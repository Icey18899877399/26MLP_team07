"""不依赖机器学习库的优化 K-Means 聚类。"""

import math
import random


class OptimizedKMeansScratch:
    """结合 K-Means++、多次初始化、标准化和容差早停的 K-Means。

    开启标准化时，inertia_、inertia_history_ 和 run_inertias_ 均为
    标准化特征空间内的簇内平方和；cluster_centers_ 使用原始特征尺度。
    """

    def __init__(
        self,
        n_clusters=3,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        standardize=True,
        random_state=42,
    ):
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if init not in ("random", "k-means++"):
            raise ValueError("init must be 'random' or 'k-means++'")
        if not isinstance(n_init, int) or n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be a finite non-negative number")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.random_state = random_state

        self.means_ = []
        self.scales_ = []
        self.cluster_centers_ = []
        self.labels_ = []
        self.inertia_ = 0.0
        self.inertia_history_ = []
        self.run_inertias_ = []
        self.n_iter_ = 0
        self._centers_transformed_ = []
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

    def _fit_standardizer(self, X):
        sample_count = len(X)
        feature_count = len(X[0])
        self.means_ = [
            sum(row[feature] for row in X) / sample_count
            for feature in range(feature_count)
        ]
        self.scales_ = []
        for feature, mean in enumerate(self.means_):
            variance = sum(
                (row[feature] - mean) ** 2 for row in X
            ) / sample_count
            scale = math.sqrt(variance)
            self.scales_.append(scale if scale > 0.0 else 1.0)

    def _transform(self, X):
        if not self.standardize:
            return [list(row) for row in X]
        return [
            [
                (value - self.means_[feature]) / self.scales_[feature]
                for feature, value in enumerate(row)
            ]
            for row in X
        ]

    def _nearest_center(self, sample, centers):
        return min(
            range(len(centers)),
            key=lambda index: self._squared_distance(sample, centers[index]),
        )

    def _assign(self, X, centers):
        return [self._nearest_center(sample, centers) for sample in X]

    def _initialize_random(self, X, generator):
        indices = generator.sample(range(len(X)), self.n_clusters)
        return [list(X[index]) for index in indices]

    def _initialize_kmeans_plus_plus(self, X, generator):
        selected = [generator.randrange(len(X))]
        while len(selected) < self.n_clusters:
            distances = [
                min(
                    self._squared_distance(sample, X[index])
                    for index in selected
                )
                for sample in X
            ]
            total = sum(distances)
            if total == 0.0:
                next_index = next(
                    (index for index in range(len(X)) if index not in selected),
                    selected[0],
                )
            else:
                threshold = generator.random() * total
                cumulative = 0.0
                next_index = len(X) - 1
                for index, distance in enumerate(distances):
                    cumulative += distance
                    if threshold < cumulative:
                        next_index = index
                        break
            selected.append(next_index)
        return [list(X[index]) for index in selected]

    def _initialize_centers(self, X, generator):
        if self.init == "random":
            return self._initialize_random(X, generator)
        return self._initialize_kmeans_plus_plus(X, generator)

    def _recompute_centers(self, X, labels):
        feature_count = len(X[0])
        centers = [None] * self.n_clusters
        for cluster in range(self.n_clusters):
            members = [sample for sample, label in zip(X, labels) if label == cluster]
            if members:
                centers[cluster] = [
                    sum(sample[feature] for sample in members) / len(members)
                    for feature in range(feature_count)
                ]

        used_indices = set()
        occupied_centers = [center for center in centers if center is not None]
        for cluster, center in enumerate(centers):
            if center is not None:
                continue
            candidate = max(
                (index for index in range(len(X)) if index not in used_indices),
                key=lambda index: (
                    min(
                        self._squared_distance(X[index], occupied)
                        for occupied in occupied_centers
                    ),
                    -index,
                ),
            )
            centers[cluster] = list(X[candidate])
            occupied_centers.append(centers[cluster])
            used_indices.add(candidate)
        return centers

    def _inertia(self, X, labels, centers):
        return sum(
            self._squared_distance(sample, centers[label])
            for sample, label in zip(X, labels)
        )

    def _single_run(self, X, generator):
        centers = self._initialize_centers(X, generator)
        history = []
        for iteration in range(1, self.max_iter + 1):
            labels = self._assign(X, centers)
            updated = self._recompute_centers(X, labels)
            shift = max(
                math.sqrt(self._squared_distance(old, new))
                for old, new in zip(centers, updated)
            )
            centers = updated
            labels = self._assign(X, centers)
            history.append(self._inertia(X, labels, centers))
            if shift <= self.tol:
                break
        return centers, labels, history[-1], history, iteration

    def _centers_in_original_scale(self):
        return [
            [
                value * self.scales_[feature] + self.means_[feature]
                for feature, value in enumerate(center)
            ]
            for center in self._centers_transformed_
        ]

    def fit(self, X):
        samples = self._validate_matrix(X)
        if self.n_clusters > len(samples):
            raise ValueError("n_clusters cannot exceed the number of samples")
        self._n_features_ = len(samples[0])
        if self.standardize:
            self._fit_standardizer(samples)
        else:
            self.means_ = [0.0] * len(samples[0])
            self.scales_ = [1.0] * len(samples[0])
        transformed = self._transform(samples)

        master = random.Random(self.random_state)
        best = None
        self.run_inertias_ = []
        for _ in range(self.n_init):
            generator = random.Random(master.randrange(2**63))
            result = self._single_run(transformed, generator)
            self.run_inertias_.append(result[2])
            if best is None or result[2] < best[2]:
                best = result

        centers, labels, inertia, history, iteration = best
        self._centers_transformed_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.inertia_history_ = history
        self.n_iter_ = iteration
        self.cluster_centers_ = self._centers_in_original_scale()
        return self

    def predict(self, X):
        if not self._centers_transformed_:
            raise ValueError("fit must be called before predict")
        samples = self._validate_matrix(X, self._n_features_)
        transformed = self._transform(samples)
        return self._assign(transformed, self._centers_transformed_)

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
