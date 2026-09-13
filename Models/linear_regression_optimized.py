"""不依赖机器学习库的优化线性回归。"""

import math
import random


class OptimizedLinearRegressionScratch:
    """结合标准化、Adam、小批量、L2正则化和早停的线性回归。

    开启标准化时，weights、bias和loss_history记录的是标准化坐标中的训练参数；
    predict会自动将预测值还原到原始目标量纲。
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1_000,
        l2=0.0,
        tol=1e-10,
        batch_size=32,
        standardize=True,
        standardize_target=True,
        random_state=42,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.tol = tol
        self.batch_size = batch_size
        self.standardize = standardize
        self.standardize_target = standardize_target
        self.random_state = random_state
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.weights = []
        self.bias = 0.0
        self.means = []
        self.scales = []
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.loss_history = []
        self.n_iter = 0

    def _fit_standardizer(self, X):
        sample_count = len(X)
        feature_count = len(X[0])
        self.means = [
            sum(row[j] for row in X) / sample_count
            for j in range(feature_count)
        ]
        self.scales = []
        for j, mean in enumerate(self.means):
            variance = sum((row[j] - mean) ** 2 for row in X) / sample_count
            scale = math.sqrt(variance)
            self.scales.append(scale if scale > 0.0 else 1.0)

    def _transform(self, X):
        if not self.standardize:
            return [list(row) for row in X]
        return [
            [(row[j] - self.means[j]) / self.scales[j] for j in range(len(row))]
            for row in X
        ]

    def _fit_target_standardizer(self, y):
        self.target_mean = sum(y) / len(y)
        variance = sum((target - self.target_mean) ** 2 for target in y) / len(y)
        scale = math.sqrt(variance)
        self.target_scale = scale if scale > 0.0 else 1.0

    def _transform_targets(self, y):
        if not self.standardize_target:
            return list(y)
        return [(target - self.target_mean) / self.target_scale for target in y]

    @staticmethod
    def _linear_output(X, weights, bias):
        return [
            sum(weight * value for weight, value in zip(weights, row)) + bias
            for row in X
        ]

    def _loss(self, X, y):
        predictions = self._linear_output(X, self.weights, self.bias)
        mse = sum((prediction - target) ** 2 for prediction, target in zip(predictions, y)) / len(y)
        return mse + self.l2 * sum(weight * weight for weight in self.weights)

    def fit(self, X, y):
        sample_count = len(X)
        feature_count = len(X[0])
        if self.standardize:
            self._fit_standardizer(X)
        else:
            self.means = [0.0] * feature_count
            self.scales = [1.0] * feature_count
        transformed_X = self._transform(X)
        if self.standardize_target:
            self._fit_target_standardizer(y)
        else:
            self.target_mean = 0.0
            self.target_scale = 1.0
        transformed_y = self._transform_targets(y)

        self.weights = [0.0] * feature_count
        self.bias = 0.0
        self.loss_history = []
        self.n_iter = 0

        first_moment = [0.0] * feature_count
        second_moment = [0.0] * feature_count
        bias_first_moment = 0.0
        bias_second_moment = 0.0
        update_count = 0
        previous_loss = None
        generator = random.Random(self.random_state)
        batch_size = sample_count if self.batch_size is None else min(self.batch_size, sample_count)

        for iteration in range(1, self.max_iter + 1):
            indices = list(range(sample_count))
            if batch_size < sample_count:
                generator.shuffle(indices)

            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_X = [transformed_X[index] for index in batch_indices]
                batch_y = [transformed_y[index] for index in batch_indices]
                predictions = self._linear_output(batch_X, self.weights, self.bias)
                errors = [prediction - target for prediction, target in zip(predictions, batch_y)]
                denominator = len(batch_indices)

                gradients = [
                    2.0 * sum(error * row[j] for error, row in zip(errors, batch_X)) / denominator
                    + 2.0 * self.l2 * self.weights[j]
                    for j in range(feature_count)
                ]
                bias_gradient = 2.0 * sum(errors) / denominator
                update_count += 1

                # Adam使用梯度的一阶、二阶矩估计调整每个参数的步长。
                for j, gradient in enumerate(gradients):
                    first_moment[j] = self.beta1 * first_moment[j] + (1.0 - self.beta1) * gradient
                    second_moment[j] = self.beta2 * second_moment[j] + (1.0 - self.beta2) * gradient * gradient
                    corrected_first = first_moment[j] / (1.0 - self.beta1 ** update_count)
                    corrected_second = second_moment[j] / (1.0 - self.beta2 ** update_count)
                    self.weights[j] -= self.learning_rate * corrected_first / (math.sqrt(corrected_second) + self.epsilon)

                bias_first_moment = self.beta1 * bias_first_moment + (1.0 - self.beta1) * bias_gradient
                bias_second_moment = self.beta2 * bias_second_moment + (1.0 - self.beta2) * bias_gradient * bias_gradient
                corrected_bias_first = bias_first_moment / (1.0 - self.beta1 ** update_count)
                corrected_bias_second = bias_second_moment / (1.0 - self.beta2 ** update_count)
                self.bias -= self.learning_rate * corrected_bias_first / (math.sqrt(corrected_bias_second) + self.epsilon)

            loss = self._loss(transformed_X, transformed_y)
            self.loss_history.append(loss)
            self.n_iter = iteration
            if previous_loss is not None and abs(previous_loss - loss) <= self.tol:
                break
            previous_loss = loss

        return self

    def predict(self, X):
        transformed_X = self._transform(X)
        predictions = self._linear_output(transformed_X, self.weights, self.bias)
        if not self.standardize_target:
            return predictions
        return [
            prediction * self.target_scale + self.target_mean
            for prediction in predictions
        ]
