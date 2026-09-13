
import math


class KNNScratch:
    """使用欧氏距离和多数投票完成KNN 分类"""

    def __init__(self, n_neighbors=3):
        # n_neighbors超参数。
        self.n_neighbors = n_neighbors
        self.X_train = []
        self.y_train = []
        self.classes = []

    def fit(self, X, y):
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)
        self.classes = []
        for label in y:
            if label not in self.classes:
                self.classes.append(label)
        return self

    @staticmethod
    def _euclidean_distance(first, second):
        return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))

    def _nearest_labels(self, sample):
        # 基础实现计算全部距离并完整排序，便于直接理解算法原理。
        distances = [
            (self._euclidean_distance(sample, train_sample), label)
            for train_sample, label in zip(self.X_train, self.y_train)
        ]
        distances.sort(key=lambda item: item[0])
        return [label for _, label in distances[: self.n_neighbors]]

    def _class_probabilities(self, sample):
        labels = self._nearest_labels(sample)
        return [labels.count(label) / len(labels) for label in self.classes]

    def predict_proba(self, X):
        """按self.classes的顺序返回各类别在K 邻居中的占比"""
        return [self._class_probabilities(sample) for sample in X]

    def predict(self, X):
        predictions = []
        for probabilities in self.predict_proba(X):
            # 平票时保留训练集中先出现的类别，使结果可重复
            best_index = max(range(len(self.classes)), key=lambda index: probabilities[index])
            predictions.append(self.classes[best_index])
        return predictions

