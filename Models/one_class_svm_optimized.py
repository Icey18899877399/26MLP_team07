"""不依赖机器学习库和数值计算包的优化 One-Class SVM。"""

import math

from .one_class_svm import _dot, _dual_offset, _validate_matrix


class OptimizedOneClassSVMScratch:
    """核化 One-Class SVM，使用 Frank-Wolfe 与精确线搜索求解。

    相比基础版，本实现支持 RBF 非线性边界、训练集标准化、对称核矩阵缓存，
    并在预测时仅计算支持向量对应的核函数。
    """

    def __init__(
        self,
        nu=0.1,
        kernel="rbf",
        gamma="scale",
        standardize=True,
        max_iter=500,
        tol=1e-6,
    ):
        if (
            not isinstance(nu, (int, float))
            or not math.isfinite(nu)
            or not 0.0 < nu <= 1.0
        ):
            raise ValueError("nu must be in (0, 1]")
        if kernel not in ("linear", "rbf"):
            raise ValueError("kernel must be 'linear' or 'rbf'")
        if gamma != "scale" and (
            not isinstance(gamma, (int, float))
            or not math.isfinite(gamma)
            or gamma <= 0.0
        ):
            raise ValueError("gamma must be 'scale' or a positive number")
        if not isinstance(standardize, bool):
            raise ValueError("standardize must be a boolean")
        if kernel == "linear" and standardize:
            raise ValueError(
                "standardize must be False for the linear kernel because mean "
                "centering collapses the one-class separating direction"
            )
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if (
            not isinstance(tol, (int, float))
            or not math.isfinite(tol)
            or tol < 0.0
        ):
            raise ValueError("tol must be non-negative")

        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol

        self.alphas_ = []
        self.offset_ = None
        self.gamma_ = None
        self.means_ = []
        self.scales_ = []
        self.kernel_matrix_ = []
        self.support_indices_ = []
        self.support_vectors_ = []
        self.support_alphas_ = []
        self.labels_ = []
        self.objective_history_ = []
        self.n_iter_ = 0
        self._training_X_ = []
        self._n_features_ = None
        self._is_fitted = False

    def _fit_standardizer(self, samples):
        if not self.standardize:
            self.means_ = [0.0] * self._n_features_
            self.scales_ = [1.0] * self._n_features_
            return
        self.means_ = [
            sum(row[feature] for row in samples) / len(samples)
            for feature in range(self._n_features_)
        ]
        self.scales_ = []
        for feature, mean in enumerate(self.means_):
            variance = sum(
                (row[feature] - mean) ** 2 for row in samples
            ) / len(samples)
            self.scales_.append(math.sqrt(variance) if variance > 0.0 else 1.0)

    def _transform(self, samples):
        return [
            [
                (value - self.means_[feature]) / self.scales_[feature]
                for feature, value in enumerate(row)
            ]
            for row in samples
        ]

    def _resolve_gamma(self, samples):
        if self.gamma != "scale":
            return float(self.gamma)
        value_count = len(samples) * self._n_features_
        overall_mean = sum(value for row in samples for value in row) / value_count
        variance = sum(
            (value - overall_mean) ** 2 for row in samples for value in row
        ) / value_count
        return 1.0 / (self._n_features_ * variance) if variance > 0.0 else 1.0

    def _kernel(self, first, second):
        if self.kernel == "linear":
            return _dot(first, second)
        squared_distance = sum(
            (left - right) ** 2 for left, right in zip(first, second)
        )
        return math.exp(-self.gamma_ * squared_distance)

    def _kernel_matrix(self, samples):
        matrix = [[0.0] * len(samples) for _ in samples]
        for row in range(len(samples)):
            for column in range(row, len(samples)):
                value = self._kernel(samples[row], samples[column])
                matrix[row][column] = value
                matrix[column][row] = value
        return matrix

    @staticmethod
    def _matrix_vector_product(matrix, vector):
        return [_dot(row, vector) for row in matrix]

    @staticmethod
    def _linear_minimizer(gradient, upper_bound):
        """在线性目标下填充梯度最小的坐标，保持对偶约束可行。"""
        vertex = [0.0] * len(gradient)
        remaining = 1.0
        for index in sorted(
            range(len(gradient)), key=lambda item: (gradient[item], item)
        ):
            amount = min(upper_bound, remaining)
            vertex[index] = amount
            remaining -= amount
            if remaining <= 1e-14:
                break
        return vertex

    def fit(self, X):
        raw_samples = _validate_matrix(X)
        self._n_features_ = len(raw_samples[0])
        self._fit_standardizer(raw_samples)
        samples = self._transform(raw_samples)
        self.gamma_ = self._resolve_gamma(samples)
        self._training_X_ = [list(row) for row in samples]
        self.kernel_matrix_ = self._kernel_matrix(samples)

        sample_count = len(samples)
        upper_bound = 1.0 / (self.nu * sample_count)
        alphas = [1.0 / sample_count] * sample_count
        gradient = self._matrix_vector_product(self.kernel_matrix_, alphas)
        objective = 0.5 * _dot(alphas, gradient)
        self.objective_history_ = [objective]

        for iteration in range(1, self.max_iter + 1):
            vertex = self._linear_minimizer(gradient, upper_bound)
            direction = [target - current for target, current in zip(vertex, alphas)]
            gap = -_dot(gradient, direction)
            self.n_iter_ = iteration
            if gap <= self.tol:
                break

            kernel_direction = self._matrix_vector_product(
                self.kernel_matrix_, direction
            )
            curvature = _dot(direction, kernel_direction)
            step = min(1.0, gap / curvature) if curvature > 1e-15 else 1.0
            alphas = [
                alpha + step * delta for alpha, delta in zip(alphas, direction)
            ]
            gradient = [
                value + step * delta
                for value, delta in zip(gradient, kernel_direction)
            ]
            objective = 0.5 * _dot(alphas, gradient)
            self.objective_history_.append(objective)

        self.alphas_ = alphas
        training_scores = self._matrix_vector_product(
            self.kernel_matrix_, self.alphas_
        )
        self.offset_ = _dual_offset(
            training_scores,
            self.alphas_,
            upper_bound,
        )
        self.support_indices_ = [
            index for index, alpha in enumerate(self.alphas_) if alpha > 1e-8
        ]
        self.support_vectors_ = [
            list(samples[index]) for index in self.support_indices_
        ]
        self.support_alphas_ = [
            self.alphas_[index] for index in self.support_indices_
        ]
        self._is_fitted = True
        self.labels_ = self.predict(raw_samples)
        return self

    def score_samples(self, X):
        """返回核展开得到的未中心化正常性分数。"""
        if not self._is_fitted:
            raise ValueError("fit must be called before score_samples")
        raw_samples = _validate_matrix(X, self._n_features_)
        samples = self._transform(raw_samples)
        return [
            sum(
                alpha * self._kernel(support, sample)
                for alpha, support in zip(
                    self.support_alphas_, self.support_vectors_
                )
            )
            for sample in samples
        ]

    def decision_function(self, X):
        return [score - self.offset_ for score in self.score_samples(X)]

    def predict(self, X):
        return [
            1 if decision >= -1e-10 else -1
            for decision in self.decision_function(X)
        ]

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
