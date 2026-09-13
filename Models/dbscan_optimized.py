"""不依赖机器学习库的优化 DBSCAN 聚类。"""

from collections import deque
import math


class _KDNode:
    def __init__(self, indices=None, point_index=None, axis=None, left=None, right=None):
        self.indices = indices
        self.point_index = point_index
        self.axis = axis
        self.left = left
        self.right = right


class _KDTree:
    """用于精确半径查询的轻量 KD-tree。"""

    def __init__(self, X, leaf_size=20):
        self.X = X
        self.leaf_size = leaf_size
        self._root = self._build(list(range(len(X))))

    def _build(self, indices):
        if len(indices) <= self.leaf_size:
            return _KDNode(indices=sorted(indices))

        feature_count = len(self.X[0])
        axis = max(
            range(feature_count),
            key=lambda feature: max(self.X[index][feature] for index in indices)
            - min(self.X[index][feature] for index in indices),
        )
        ordered = sorted(indices, key=lambda index: (self.X[index][axis], index))
        middle = len(ordered) // 2
        return _KDNode(
            point_index=ordered[middle],
            axis=axis,
            left=self._build(ordered[:middle]),
            right=self._build(ordered[middle + 1 :]),
        )

    @staticmethod
    def _squared_distance(first, second):
        return sum((left - right) ** 2 for left, right in zip(first, second))

    def query_radius(self, point, radius_squared):
        matches = []

        def visit(node):
            if node.indices is not None:
                for index in node.indices:
                    if self._squared_distance(point, self.X[index]) <= radius_squared:
                        matches.append(index)
                return

            sample = self.X[node.point_index]
            if self._squared_distance(point, sample) <= radius_squared:
                matches.append(node.point_index)

            difference = point[node.axis] - sample[node.axis]
            near, far = (node.left, node.right) if difference <= 0.0 else (node.right, node.left)
            visit(near)
            if difference ** 2 <= radius_squared:
                visit(far)

        visit(self._root)
        return sorted(matches)


class OptimizedDBSCANScratch:
    """结合标准化和精确 KD-tree 邻域搜索的 DBSCAN。

    开启标准化时，eps 和最近邻距离位于标准化空间，components_ 保留原始尺度。
    """

    NOISE = -1
    _UNVISITED = -2

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        algorithm="kd_tree",
        leaf_size=20,
        standardize=True,
    ):
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive number")
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if algorithm not in ("kd_tree", "brute"):
            raise ValueError("algorithm must be 'kd_tree' or 'brute'")
        if not isinstance(leaf_size, int) or leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.standardize = standardize

        self.means_ = []
        self.scales_ = []
        self.labels_ = []
        self.core_sample_indices_ = []
        self.components_ = []
        self.n_clusters_ = 0
        self._core_transformed_ = []
        self._core_labels_ = []
        self._core_tree = None
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

    def _fit_standardizer(self, X):
        sample_count = len(X)
        feature_count = len(X[0])
        self.means_ = [
            sum(row[feature] for row in X) / sample_count
            for feature in range(feature_count)
        ]
        self.scales_ = []
        for feature, mean in enumerate(self.means_):
            variance = sum((row[feature] - mean) ** 2 for row in X) / sample_count
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

    def _make_neighbor_query(self, X):
        radius_squared = self.eps ** 2
        if self.algorithm == "kd_tree":
            tree = _KDTree(X, self.leaf_size)
            return lambda index: tree.query_radius(X[index], radius_squared)
        return lambda index: [
            candidate
            for candidate, row in enumerate(X)
            if self._squared_distance(X[index], row) <= radius_squared
        ]

    def _expand_cluster(
        self,
        labels,
        start,
        neighbors,
        cluster,
        core_indices,
        radius_neighbors,
    ):
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
            current_neighbors = radius_neighbors(current)
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
        if self.standardize:
            self._fit_standardizer(samples)
        else:
            self.means_ = [0.0] * self._n_features_
            self.scales_ = [1.0] * self._n_features_
        transformed = self._transform(samples)
        radius_neighbors = self._make_neighbor_query(transformed)

        labels = [self._UNVISITED] * len(samples)
        core_indices = set()
        cluster = 0
        for sample_index in range(len(samples)):
            if labels[sample_index] != self._UNVISITED:
                continue
            neighbors = radius_neighbors(sample_index)
            if len(neighbors) < self.min_samples:
                labels[sample_index] = self.NOISE
                continue
            self._expand_cluster(
                labels,
                sample_index,
                neighbors,
                cluster,
                core_indices,
                radius_neighbors,
            )
            cluster += 1

        self.labels_ = labels
        self.core_sample_indices_ = sorted(core_indices)
        self.components_ = [list(samples[index]) for index in self.core_sample_indices_]
        self._core_transformed_ = [
            list(transformed[index]) for index in self.core_sample_indices_
        ]
        self._core_labels_ = [labels[index] for index in self.core_sample_indices_]
        self._core_tree = (
            _KDTree(self._core_transformed_, self.leaf_size)
            if self.algorithm == "kd_tree" and self._core_transformed_
            else None
        )
        self.n_clusters_ = cluster
        self._is_fitted = True
        return self

    def predict(self, X):
        """将新样本分配给 eps 范围内最近的核心样本，否则标记为噪声。"""
        if not self._is_fitted:
            raise ValueError("fit must be called before predict")
        samples = self._validate_matrix(X, self._n_features_)
        transformed = self._transform(samples)
        radius_squared = self.eps ** 2
        predictions = []

        for sample in transformed:
            if self._core_tree is not None:
                candidate_indices = self._core_tree.query_radius(sample, radius_squared)
            else:
                candidate_indices = [
                    index
                    for index, core in enumerate(self._core_transformed_)
                    if self._squared_distance(sample, core) <= radius_squared
                ]
            if not candidate_indices:
                predictions.append(self.NOISE)
                continue
            nearest = min(
                candidate_indices,
                key=lambda index: (
                    self._squared_distance(sample, self._core_transformed_[index]),
                    index,
                ),
            )
            predictions.append(self._core_labels_[nearest])
        return predictions

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
