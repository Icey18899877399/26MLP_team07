"""不依赖机器学习库的优化 MLP 回归。"""

import math
import random


class OptimizedMLPRegressorScratch:
    """支持多隐藏层、Adam、标准化、L2 与验证早停的 MLP 回归器。"""

    def __init__(
        self,
        hidden_layer_sizes=(16, 8),
        activation="relu",
        learning_rate=0.001,
        max_iter=1_000,
        batch_size=32,
        l2=0.0,
        tol=1e-6,
        validation_fraction=0.2,
        n_iter_no_change=20,
        standardize=True,
        standardize_target=True,
        gradient_clip=5.0,
        random_state=42,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.standardize = standardize
        self.standardize_target = standardize_target
        self.gradient_clip = gradient_clip
        self.random_state = random_state
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.weights_ = []
        self.biases_ = []
        self.feature_means_ = []
        self.feature_scales_ = []
        self.target_mean_ = 0.0
        self.target_scale_ = 1.0
        self.training_indices_ = []
        self.validation_indices_ = []
        self.train_loss_ = []
        self.validation_loss_ = []
        self.n_iter_ = 0
        self.best_iteration_ = 0

    def _activate(self, value):
        if self.activation == "tanh":
            return math.tanh(value)
        return value if value > 0.0 else 0.0

    def _activation_derivative(self, activated):
        if self.activation == "tanh":
            return 1.0 - activated * activated
        return 1.0 if activated > 0.0 else 0.0

    @staticmethod
    def _mean_scale(values):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        return mean, scale if scale > 0.0 else 1.0

    def _fit_standardizers(self, X, y):
        feature_count = len(X[0])
        if self.standardize:
            statistics = [
                self._mean_scale([row[feature] for row in X])
                for feature in range(feature_count)
            ]
            self.feature_means_ = [item[0] for item in statistics]
            self.feature_scales_ = [item[1] for item in statistics]
        else:
            self.feature_means_ = [0.0] * feature_count
            self.feature_scales_ = [1.0] * feature_count

        if self.standardize_target:
            self.target_mean_, self.target_scale_ = self._mean_scale(y)
        else:
            self.target_mean_, self.target_scale_ = 0.0, 1.0

    def _transform_features(self, X):
        return [
            [
                (value - self.feature_means_[feature])
                / self.feature_scales_[feature]
                for feature, value in enumerate(row)
            ]
            for row in X
        ]

    def _transform_targets(self, y):
        return [
            (target - self.target_mean_) / self.target_scale_
            for target in y
        ]

    def _initialize(self, feature_count, generator):
        layer_sizes = (feature_count, *self.hidden_layer_sizes, 1)
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:]):
            deviation = math.sqrt(
                (2.0 if self.activation == "relu" else 1.0) / fan_in
            )
            self.weights_.append(
                [
                    [generator.gauss(0.0, deviation) for _ in range(fan_out)]
                    for _ in range(fan_in)
                ]
            )
            self.biases_.append([0.0] * fan_out)

    def _forward_one(self, row):
        activations = [row]
        current = row
        last_layer = len(self.weights_) - 1
        for layer, (weights, biases) in enumerate(zip(self.weights_, self.biases_)):
            values = [
                biases[output]
                + sum(
                    current[input_index] * weights[input_index][output]
                    for input_index in range(len(current))
                )
                for output in range(len(biases))
            ]
            current = (
                values
                if layer == last_layer
                else [self._activate(value) for value in values]
            )
            activations.append(current)
        return activations

    def _zero_gradients(self):
        weight_gradients = [
            [[0.0] * len(layer[0]) for _ in range(len(layer))]
            for layer in self.weights_
        ]
        bias_gradients = [[0.0] * len(layer) for layer in self.biases_]
        return weight_gradients, bias_gradients

    def _batch_gradients(self, X, y):
        weight_gradients, bias_gradients = self._zero_gradients()
        batch_size = len(X)

        for row, target in zip(X, y):
            activations = self._forward_one(row)
            deltas = [2.0 * (activations[-1][0] - target) / batch_size]

            # 从线性输出层开始，按层反向累计权重与偏置梯度。
            for layer in range(len(self.weights_) - 1, -1, -1):
                previous = activations[layer]
                for input_index in range(len(previous)):
                    for output_index in range(len(deltas)):
                        weight_gradients[layer][input_index][output_index] += (
                            previous[input_index] * deltas[output_index]
                        )
                for output_index, delta in enumerate(deltas):
                    bias_gradients[layer][output_index] += delta

                if layer > 0:
                    deltas = [
                        sum(
                            deltas[output]
                            * self.weights_[layer][hidden][output]
                            for output in range(len(deltas))
                        )
                        * self._activation_derivative(activations[layer][hidden])
                        for hidden in range(len(activations[layer]))
                    ]

        for layer, weights in enumerate(self.weights_):
            for input_index, row in enumerate(weights):
                for output_index, weight in enumerate(row):
                    weight_gradients[layer][input_index][output_index] += (
                        2.0 * self.l2 * weight
                    )
        return weight_gradients, bias_gradients

    def _clip(self, value):
        if self.gradient_clip is None:
            return value
        return max(-self.gradient_clip, min(self.gradient_clip, value))

    def _adam_step(self, gradients, moments, velocities, step):
        for layer in range(len(self.weights_)):
            weight_gradients, bias_gradients = gradients
            for input_index in range(len(self.weights_[layer])):
                for output_index in range(len(self.weights_[layer][input_index])):
                    gradient = self._clip(
                        weight_gradients[layer][input_index][output_index]
                    )
                    moments[0][layer][input_index][output_index] = (
                        self.beta1 * moments[0][layer][input_index][output_index]
                        + (1.0 - self.beta1) * gradient
                    )
                    velocities[0][layer][input_index][output_index] = (
                        self.beta2 * velocities[0][layer][input_index][output_index]
                        + (1.0 - self.beta2) * gradient * gradient
                    )
                    first = moments[0][layer][input_index][output_index] / (
                        1.0 - self.beta1 ** step
                    )
                    second = velocities[0][layer][input_index][output_index] / (
                        1.0 - self.beta2 ** step
                    )
                    self.weights_[layer][input_index][output_index] -= (
                        self.learning_rate
                        * first
                        / (math.sqrt(second) + self.epsilon)
                    )

            for output_index in range(len(self.biases_[layer])):
                gradient = self._clip(bias_gradients[layer][output_index])
                moments[1][layer][output_index] = (
                    self.beta1 * moments[1][layer][output_index]
                    + (1.0 - self.beta1) * gradient
                )
                velocities[1][layer][output_index] = (
                    self.beta2 * velocities[1][layer][output_index]
                    + (1.0 - self.beta2) * gradient * gradient
                )
                first = moments[1][layer][output_index] / (1.0 - self.beta1 ** step)
                second = velocities[1][layer][output_index] / (1.0 - self.beta2 ** step)
                self.biases_[layer][output_index] -= (
                    self.learning_rate
                    * first
                    / (math.sqrt(second) + self.epsilon)
                )

    @staticmethod
    def _copy_parameters(weights, biases):
        return (
            [[list(row) for row in layer] for layer in weights],
            [list(layer) for layer in biases],
        )

    def _loss(self, X, y):
        predictions = [self._forward_one(row)[-1][0] for row in X]
        mse = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, y)
        ) / len(y)
        penalty = self.l2 * sum(
            weight * weight
            for layer in self.weights_
            for row in layer
            for weight in row
        )
        return mse + penalty

    def fit(self, X, y):
        generator = random.Random(self.random_state)
        indices = list(range(len(X)))
        generator.shuffle(indices)
        if len(indices) < 2 or self.validation_fraction <= 0.0:
            validation_count = 0
        else:
            validation_count = max(
                1,
                min(len(indices) - 1, round(len(indices) * self.validation_fraction)),
            )
        self.validation_indices_ = sorted(indices[:validation_count])
        self.training_indices_ = sorted(indices[validation_count:])

        training_X = [list(X[index]) for index in self.training_indices_]
        training_y = [y[index] for index in self.training_indices_]
        validation_X = [list(X[index]) for index in self.validation_indices_]
        validation_y = [y[index] for index in self.validation_indices_]

        # 只用内部训练子集拟合标准化参数，避免验证信息泄漏。
        self._fit_standardizers(training_X, training_y)
        training_X = self._transform_features(training_X)
        training_y = self._transform_targets(training_y)
        validation_X = self._transform_features(validation_X)
        validation_y = self._transform_targets(validation_y)
        self._initialize(len(training_X[0]), generator)

        first_moments = self._zero_gradients()
        second_moments = self._zero_gradients()
        batch_size = (
            len(training_X)
            if self.batch_size is None
            else min(self.batch_size, len(training_X))
        )
        best_loss = math.inf
        patience_loss = math.inf
        best_parameters = self._copy_parameters(self.weights_, self.biases_)
        best_iteration = 0
        stale_epochs = 0
        update_count = 0
        self.train_loss_ = []
        self.validation_loss_ = []

        for iteration in range(1, self.max_iter + 1):
            epoch_indices = list(range(len(training_X)))
            if batch_size < len(training_X):
                generator.shuffle(epoch_indices)
            for start in range(0, len(epoch_indices), batch_size):
                batch_indices = epoch_indices[start:start + batch_size]
                batch_X = [training_X[index] for index in batch_indices]
                batch_y = [training_y[index] for index in batch_indices]
                gradients = self._batch_gradients(batch_X, batch_y)
                update_count += 1
                self._adam_step(
                    gradients,
                    first_moments,
                    second_moments,
                    update_count,
                )

            train_loss = self._loss(training_X, training_y)
            monitored_loss = (
                self._loss(validation_X, validation_y)
                if validation_X
                else train_loss
            )
            self.train_loss_.append(train_loss)
            self.validation_loss_.append(monitored_loss)

            # 最佳快照追踪严格最小值；tol 只决定早停耐心是否重置。
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_parameters = self._copy_parameters(self.weights_, self.biases_)
                best_iteration = iteration

            if monitored_loss < patience_loss - self.tol:
                patience_loss = monitored_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.n_iter_no_change:
                    break

        self.weights_, self.biases_ = best_parameters
        self.best_iteration_ = best_iteration
        self.n_iter_ = iteration
        return self

    def predict(self, X):
        transformed = self._transform_features(X)
        predictions = [self._forward_one(row)[-1][0] for row in transformed]
        return [
            value * self.target_scale_ + self.target_mean_
            for value in predictions
        ]
