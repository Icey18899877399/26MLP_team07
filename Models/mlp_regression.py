"""不依赖机器学习库的基础 MLP 回归。"""

import math
import random


class MLPRegressorScratch:
    """单隐藏层 MLP，使用 tanh、均方误差和全批量梯度下降。"""

    def __init__(
        self,
        hidden_size=8,
        learning_rate=0.01,
        max_iter=1_000,
        random_state=42,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self.input_weights_ = []
        self.hidden_bias_ = []
        self.output_weights_ = []
        self.output_bias_ = 0.0
        self.loss_history_ = []

    def _initialize(self, feature_count):
        generator = random.Random(self.random_state)
        input_scale = 1.0 / math.sqrt(feature_count)
        hidden_scale = 1.0 / math.sqrt(self.hidden_size)
        self.input_weights_ = [
            [generator.uniform(-input_scale, input_scale) for _ in range(self.hidden_size)]
            for _ in range(feature_count)
        ]
        self.hidden_bias_ = [0.0] * self.hidden_size
        self.output_weights_ = [
            generator.uniform(-hidden_scale, hidden_scale)
            for _ in range(self.hidden_size)
        ]
        self.output_bias_ = 0.0

    def _forward_one(self, row):
        hidden = [
            math.tanh(
                self.hidden_bias_[unit]
                + sum(
                    row[feature] * self.input_weights_[feature][unit]
                    for feature in range(len(row))
                )
            )
            for unit in range(self.hidden_size)
        ]
        output = self.output_bias_ + sum(
            hidden[unit] * self.output_weights_[unit]
            for unit in range(self.hidden_size)
        )
        return hidden, output

    def fit(self, X, y):
        feature_count = len(X[0])
        sample_count = len(X)
        self._initialize(feature_count)
        self.loss_history_ = []

        for _ in range(self.max_iter):
            input_gradient = [
                [0.0] * self.hidden_size
                for _ in range(feature_count)
            ]
            hidden_bias_gradient = [0.0] * self.hidden_size
            output_gradient = [0.0] * self.hidden_size
            output_bias_gradient = 0.0
            squared_error = 0.0

            for row, target in zip(X, y):
                hidden, prediction = self._forward_one(row)
                error = prediction - target
                squared_error += error * error
                # 输出层为线性单元；误差由输出层逐层传回隐藏层。
                output_delta = 2.0 * error / sample_count
                output_bias_gradient += output_delta

                for unit in range(self.hidden_size):
                    output_gradient[unit] += output_delta * hidden[unit]
                    hidden_delta = (
                        output_delta
                        * self.output_weights_[unit]
                        * (1.0 - hidden[unit] * hidden[unit])
                    )
                    hidden_bias_gradient[unit] += hidden_delta
                    for feature in range(feature_count):
                        input_gradient[feature][unit] += hidden_delta * row[feature]

            for feature in range(feature_count):
                for unit in range(self.hidden_size):
                    self.input_weights_[feature][unit] -= (
                        self.learning_rate * input_gradient[feature][unit]
                    )
            for unit in range(self.hidden_size):
                self.hidden_bias_[unit] -= self.learning_rate * hidden_bias_gradient[unit]
                self.output_weights_[unit] -= self.learning_rate * output_gradient[unit]
            self.output_bias_ -= self.learning_rate * output_bias_gradient
            self.loss_history_.append(squared_error / sample_count)

        return self

    def predict(self, X):
        return [self._forward_one(row)[1] for row in X]
