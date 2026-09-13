"""优化K近邻分类器"""

import heapq
import math


class OptimizedKNNScratch:
    """增加标准化、Minkowski距离、加权投票和堆式近邻搜索"""

    def __init__(
        self,
        n_neighbors=5,
        p=2,
        weights="distance",
        standardize=True,
    ):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.standardize = standardize

        self.X_train = []
        self.y_train = []
        self.classes = []
        self.means = []
        self.scales = []

    def _fit_standardizer(self, X):
        # 均值和标准差只由训练集计算，预测时复用，避免数据泄漏。
        sample_count = len(X)
        feature_count = len(X[0])
        self.means = [
            sum(row[index] for row in X) / sample_count
            for index in range(feature_count)
        ]
        self.scales = []
        for index, mean in enumerate(self.means):
            variance = sum((row[index] - mean) ** 2 for row in X) / sample_count
            scale = math.sqrt(variance)
            self.scales.append(scale if scale > 0.0 else 1.0)

    def _transform(self, X):
        if not self.standardize:
            return [list(row) for row in X]
        return [
            [
                (value - self.means[index]) / self.scales[index]
                for index, value in enumerate(row)
            ]
            for row in X
        ]

    def fit(self, X, y):
        if self.standardize:
            self._fit_standardizer(X)
        else:
            self.means = [0.0] * len(X[0])
            self.scales = [1.0] * len(X[0])

        self.X_train = self._transform(X)
        self.y_train = list(y)
        self.classes = []
        for label in y:
            if label not in self.classes:
                self.classes.append(label)
        return self

    def _minkowski_distance(self, first, second):
        if self.p == 1:
            return sum(abs(left - right) for left, right in zip(first, second))
        if self.p == 2:
            return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))
        return sum(abs(left - right) ** self.p for left, right in zip(first, second)) ** (
            1.0 / self.p
        )

    def _nearest_neighbors(self, sample):
        # nsmallest维护大小约为K的堆，避免对全部训练样本完整排序
        candidates = (
            (self._minkowski_distance(sample, train_sample), index, label)
            for index, (train_sample, label) in enumerate(zip(self.X_train, self.y_train))
        )
        return heapq.nsmallest(
            self.n_neighbors,
            candidates,
            key=lambda item: (item[0], item[1]),
        )

    def _class_scores(self, sample):
        neighbors = self._nearest_neighbors(sample)
        scores = {label: 0.0 for label in self.classes}

        if self.weights == "distance":
            # 查询点与训练点重合时，只让零距离邻居参与投票。
            exact_matches = [neighbor for neighbor in neighbors if neighbor[0] == 0.0]
            if exact_matches:
                for _, _, label in exact_matches:
                    scores[label] += 1.0
                return scores

            for distance, _, label in neighbors:
                scores[label] += 1.0 / distance
            return scores

        for _, _, label in neighbors:
            scores[label] += 1.0
        return scores

    def predict_proba(self, X):
        transformed_X = self._transform(X)
        probabilities = []
        for sample in transformed_X:
            scores = self._class_scores(sample)
            total_score = sum(scores.values())
            probabilities.append([scores[label] / total_score for label in self.classes])
        return probabilities

    def predict(self, X):
        predictions = []
        for probabilities in self.predict_proba(X):
            best_index = max(range(len(self.classes)), key=lambda index: probabilities[index])
            predictions.append(self.classes[best_index])
        return predictions

