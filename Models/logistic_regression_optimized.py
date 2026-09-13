"""优化二分类逻辑回归"""

import math


class OptimizedLogisticRegressionScratch:
    """增加标准化、L2、类别平衡、稳定损失和早停"""

    def __init__(
        self,
        learning_rate=0.1,
        max_iter=1_000,
        threshold=0.5,
        l2=0.0,
        tol=1e-8,
        standardize=True,
        class_weight=None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.l2 = l2
        self.tol = tol
        self.standardize = standardize
        self.class_weight = class_weight

        self.weights = []
        self.bias = 0.0
        self.means = []
        self.scales = []
        self.loss_history = []
        self.n_iter = 0

    @staticmethod
    def _sigmoid(value):
        # 稳定计算Sigmoid，防止绝对值较大的输入造成指数溢出
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    @staticmethod
    def _softplus(value):
        # softplus(z) - y*z 等价于二元交叉熵
        return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

    def _fit_standardizer(self, X):
        # 均值和标准差只从训练数据计算
        sample_count = len(X)
        feature_count = len(X[0])
        self.means = [sum(row[j] for row in X) / sample_count for j in range(feature_count)]
        self.scales = []

        for j, mean in enumerate(self.means):
            variance = sum((row[j] - mean) ** 2 for row in X) / sample_count
            scale = math.sqrt(variance)
            self.scales.append(scale if scale > 0 else 1.0)

    def _transform(self, X):
        if not self.standardize:
            return X
        return [
            [(row[j] - self.means[j]) / self.scales[j] for j in range(len(row))]
            for row in X
        ]

    def _sample_weights(self, y):
        if self.class_weight != "balanced":
            return [1.0] * len(y)

        negative_count = y.count(0)
        positive_count = y.count(1)
        class_weights = {
            0: len(y) / (2.0 * negative_count),
            1: len(y) / (2.0 * positive_count),
        }
        return [class_weights[target] for target in y]

    def fit(self, X, y):
        sample_count = len(X)
        feature_count = len(X[0])
        if self.standardize:
            self._fit_standardizer(X)
        else:
            self.means = [0.0] * feature_count
            self.scales = [1.0] * feature_count

        transformed_X = self._transform(X)
        sample_weights = self._sample_weights(y)
        total_weight = sum(sample_weights)

        self.weights = [0.0] * feature_count
        self.bias = 0.0
        self.loss_history = []
        previous_loss = None

        for iteration in range(1, self.max_iter + 1):
            # 前向计算线性输出及正类概率
            logits = [
                sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
                for row in transformed_X
            ]
            probabilities = [self._sigmoid(logit) for logit in logits]
            errors = [
                weight * (probability - target)
                for weight, probability, target in zip(sample_weights, probabilities, y)
            ]

            # L2只惩罚特征权重，不惩罚截距bias
            gradients = [
                sum(error * row[j] for error, row in zip(errors, transformed_X)) / total_weight
                + self.l2 * self.weights[j]
                for j in range(feature_count)
            ]
            bias_gradient = sum(errors) / total_weight

            self.weights = [
                weight - self.learning_rate * gradient
                for weight, gradient in zip(self.weights, gradients)
            ]
            self.bias -= self.learning_rate * bias_gradient

            updated_logits = [
                sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
                for row in transformed_X
            ]
            data_loss = sum(
                weight * (self._softplus(logit) - target * logit)
                for weight, logit, target in zip(sample_weights, updated_logits, y)
            ) / total_weight
            # 保存完整目标函数，供收敛曲线和早停判断使用
            loss = data_loss + 0.5 * self.l2 * sum(weight * weight for weight in self.weights)
            self.loss_history.append(loss)
            self.n_iter = iteration

            # 相邻两轮损失变化足够小时停止训练
            if previous_loss is not None and abs(previous_loss - loss) <= self.tol:
                break
            previous_loss = loss

        return self

    def _positive_probability(self, X):
        transformed_X = self._transform(X)
        return [
            self._sigmoid(sum(weight * value for weight, value in zip(self.weights, row)) + self.bias)
            for row in transformed_X
        ]

    def predict_proba(self, X):
        return [[1.0 - probability, probability] for probability in self._positive_probability(X)]

    def predict(self, X):
        return [
            int(probability >= self.threshold)
            for probability in self._positive_probability(X)
        ]
