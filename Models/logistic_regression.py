
import math


class LogisticRegressionScratch:
    """使用批量梯度下降训练原始逻辑回归"""

    def __init__(self, learning_rate=0.1, max_iter=1_000, threshold=0.5):
        # 这三个参数可直接交给后续的网格搜索或其他调参模块。
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.weights = []
        self.bias = 0.0

    @staticmethod
    def _sigmoid(value):
        # 分正负两种形式计算，避免value很小时exp(-value)溢出
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def fit(self, X, y):
        sample_count = len(X)
        feature_count = len(X[0])

        # 参数从零开始，bias对应模型中的截距项
        self.weights = [0.0] * feature_count
        self.bias = 0.0

        for _ in range(self.max_iter):
            # 逻辑回归模型：p(y=1|x) = sigmoid(w·x + b)
            probabilities = self._positive_probability(X)
            # 交叉熵损失对线性输出的导数为prediction - target
            errors = [probability - target for probability, target in zip(probabilities, y)]

            # 对全部样本求平均梯度，即批量梯度下降
            weight_gradients = [
                sum(error * row[j] for error, row in zip(errors, X)) / sample_count
                for j in range(feature_count)
            ]
            bias_gradient = sum(errors) / sample_count

            # 沿损失函数梯度的反方向更新权重和截距
            self.weights = [
                weight - self.learning_rate * gradient
                for weight, gradient in zip(self.weights, weight_gradients)
            ]
            self.bias -= self.learning_rate * bias_gradient

        return self

    def _positive_probability(self, X):
        return [
            self._sigmoid(sum(weight * value for weight, value in zip(self.weights, row)) + self.bias)
            for row in X
        ]

    def predict_proba(self, X):
        return [[1.0 - probability, probability] for probability in self._positive_probability(X)]

    def predict(self, X):
        return [
            int(probability >= self.threshold)
            for probability in self._positive_probability(X)
        ]
