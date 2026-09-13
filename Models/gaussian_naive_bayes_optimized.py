"""数值稳定并支持增量训练的高斯朴素贝叶斯分类器。"""

import math


class OptimizedGaussianNaiveBayesScratch:
    """使用对数似然、方差平滑和在线统计完成多分类。"""

    def __init__(self, var_smoothing=1e-9, class_prior=None):
        self.var_smoothing = var_smoothing
        self.class_prior = class_prior
        self._reset()

    def _reset(self):
        self.classes = []
        self.class_counts = []
        self.class_priors = []
        self.means = []
        self.variances = []
        self._class_m2 = []
        self._global_count = 0
        self._global_mean = []
        self._global_m2 = []
        self.epsilon = 0.0

    def _initialize_statistics(self, feature_count, classes):
        self.classes = list(classes)
        class_count = len(self.classes)
        self.class_counts = [0] * class_count
        self.means = [[0.0] * feature_count for _ in self.classes]
        self.variances = [[0.0] * feature_count for _ in self.classes]
        self._class_m2 = [[0.0] * feature_count for _ in self.classes]
        self._global_mean = [0.0] * feature_count
        self._global_m2 = [0.0] * feature_count

    def _update_global_statistics(self, sample):
        self._global_count += 1
        for index, value in enumerate(sample):
            delta = value - self._global_mean[index]
            self._global_mean[index] += delta / self._global_count
            delta_after_update = value - self._global_mean[index]
            self._global_m2[index] += delta * delta_after_update

    def _update_class_statistics(self, sample, class_index):
        self.class_counts[class_index] += 1
        count = self.class_counts[class_index]
        for index, value in enumerate(sample):
            delta = value - self.means[class_index][index]
            self.means[class_index][index] += delta / count
            delta_after_update = value - self.means[class_index][index]
            self._class_m2[class_index][index] += delta * delta_after_update

    def _refresh_statistics(self):
        global_variances = [
            value / self._global_count for value in self._global_m2
        ]
        self.epsilon = max(
            1e-12,
            self.var_smoothing * max(global_variances),
        )

        self.variances = []
        for count, class_m2 in zip(self.class_counts, self._class_m2):
            if count == 0:
                self.variances.append([self.epsilon] * len(class_m2))
            else:
                self.variances.append(
                    [value / count + self.epsilon for value in class_m2]
                )

        if self.class_prior is None:
            self.class_priors = [
                count / self._global_count for count in self.class_counts
            ]
        else:
            self.class_priors = [self.class_prior[label] for label in self.classes]

    def fit(self, X, y):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y, classes=None):
        if not self.classes:
            learned_classes = classes
            if learned_classes is None:
                learned_classes = []
                for label in y:
                    if label not in learned_classes:
                        learned_classes.append(label)
            self._initialize_statistics(len(X[0]), learned_classes)

        for sample, label in zip(X, y):
            class_index = self.classes.index(label)
            self._update_global_statistics(sample)
            self._update_class_statistics(sample, class_index)

        self._refresh_statistics()
        return self

    def _joint_log_likelihoods(self, sample):
        scores = []
        for prior, means, variances in zip(
            self.class_priors,
            self.means,
            self.variances,
        ):
            if prior == 0.0:
                scores.append(float("-inf"))
                continue

            score = math.log(prior)
            for value, mean, variance in zip(sample, means, variances):
                score -= 0.5 * (
                    math.log(2.0 * math.pi * variance)
                    + ((value - mean) ** 2) / variance
                )
            scores.append(score)
        return scores

    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            log_scores = self._joint_log_likelihoods(sample)
            maximum = max(log_scores)
            shifted_scores = [math.exp(score - maximum) for score in log_scores]
            total = sum(shifted_scores)
            probabilities.append([score / total for score in shifted_scores])
        return probabilities

    def predict(self, X):
        predictions = []
        for probabilities in self.predict_proba(X):
            best_index = max(range(len(self.classes)), key=lambda i: probabilities[i])
            predictions.append(self.classes[best_index])
        return predictions
