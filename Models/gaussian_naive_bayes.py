"""手写高斯朴素贝叶斯分类器。"""

import math


class GaussianNaiveBayesScratch:
    """使用高斯密度连乘完成多分类。"""

    def __init__(self, variance_floor=1e-9):
        self.variance_floor = variance_floor
        self.classes = []
        self.class_priors = []
        self.means = []
        self.variances = []

    def fit(self, X, y):
        self.classes = []
        for label in y:
            if label not in self.classes:
                self.classes.append(label)

        sample_count = len(y)
        feature_count = len(X[0])
        self.class_priors = []
        self.means = []
        self.variances = []

        for label in self.classes:
            class_samples = [row for row, target in zip(X, y) if target == label]
            class_count = len(class_samples)
            class_means = [
                sum(row[index] for row in class_samples) / class_count
                for index in range(feature_count)
            ]
            class_variances = [
                max(
                    sum(
                        (row[index] - class_means[index]) ** 2
                        for row in class_samples
                    )
                    / class_count,
                    self.variance_floor,
                )
                for index in range(feature_count)
            ]

            self.class_priors.append(class_count / sample_count)
            self.means.append(class_means)
            self.variances.append(class_variances)
        return self

    @staticmethod
    def _gaussian_density(value, mean, variance):
        coefficient = 1.0 / math.sqrt(2.0 * math.pi * variance)
        exponent = math.exp(-((value - mean) ** 2) / (2.0 * variance))
        return coefficient * exponent

    def _class_likelihoods(self, sample):
        likelihoods = []
        for prior, means, variances in zip(
            self.class_priors,
            self.means,
            self.variances,
        ):
            likelihood = prior
            for value, mean, variance in zip(sample, means, variances):
                likelihood *= self._gaussian_density(value, mean, variance)
            likelihoods.append(likelihood)
        return likelihoods

    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            likelihoods = self._class_likelihoods(sample)
            total = sum(likelihoods)
            probabilities.append([value / total for value in likelihoods])
        return probabilities

    def predict(self, X):
        predictions = []
        for probabilities in self.predict_proba(X):
            best_index = max(range(len(self.classes)), key=lambda i: probabilities[i])
            predictions.append(self.classes[best_index])
        return predictions
