"""不依赖机器学习库的基础线性回归。"""


class LinearRegressionScratch:
    """使用批量梯度下降最小化均方误差。"""

    def __init__(self, learning_rate=0.01, max_iter=1_000):
        # 参数保留为公开属性，便于后续网格搜索或前端调参。
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        sample_count = len(X)
        feature_count = len(X[0])
        self.weights = [0.0] * feature_count
        self.bias = 0.0

        for _ in range(self.max_iter):
            predictions = self.predict(X)
            errors = [prediction - target for prediction, target in zip(predictions, y)]

            # MSE 对每个权重和截距的梯度。
            weight_gradients = [
                2.0 * sum(error * row[j] for error, row in zip(errors, X)) / sample_count
                for j in range(feature_count)
            ]
            bias_gradient = 2.0 * sum(errors) / sample_count

            self.weights = [
                weight - self.learning_rate * gradient
                for weight, gradient in zip(self.weights, weight_gradients)
            ]
            self.bias -= self.learning_rate * bias_gradient

        return self

    def predict(self, X):
        return [
            sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
            for row in X
        ]
