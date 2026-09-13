"""不依赖机器学习库和数值计算包的基础 One-Class SVM。"""

import math


def _validate_matrix(X, expected_features=None):
    """将输入转为列表矩阵，并检查数值、形状和特征数。"""
    samples = [list(row) for row in X]
    if not samples or not samples[0]:
        raise ValueError("X must be a non-empty matrix")
    feature_count = len(samples[0])
    if expected_features is not None and feature_count != expected_features:
        raise ValueError("X has a different feature count from the fitted data")
    for row in samples:
        if len(row) != feature_count:
            raise ValueError("X must be rectangular")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in row
        ):
            raise ValueError("X must contain only finite numeric values")
    return samples


def _dot(first, second):
    return sum(left * right for left, right in zip(first, second))


def _matrix_vector_product(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _dual_objective(kernel_matrix, alphas):
    return 0.5 * _dot(alphas, _matrix_vector_product(kernel_matrix, alphas))


def _project_capped_simplex(values, upper_bound):
    """投影到 sum(alpha)=1 且 0<=alpha<=upper_bound 的集合。"""
    low = min(values) - upper_bound
    high = max(values)
    for _ in range(80):
        threshold = (low + high) / 2.0
        total = sum(
            min(upper_bound, max(0.0, value - threshold))
            for value in values
        )
        if total > 1.0:
            low = threshold
        else:
            high = threshold

    projected = [
        min(upper_bound, max(0.0, value - (low + high) / 2.0))
        for value in values
    ]
    residual = 1.0 - sum(projected)
    if residual > 0.0:
        for index in range(len(projected)):
            addition = min(upper_bound - projected[index], residual)
            projected[index] += addition
            residual -= addition
            if residual <= 1e-14:
                break
    elif residual < 0.0:
        for index in range(len(projected)):
            reduction = min(projected[index], -residual)
            projected[index] -= reduction
            residual += reduction
            if residual >= -1e-14:
                break
    return projected


def _dual_offset(training_scores, alphas, upper_bound, tolerance=1e-8):
    """根据 KKT 条件由自由支持向量估计 rho。"""
    free_scores = [
        score
        for score, alpha in zip(training_scores, alphas)
        if tolerance < alpha < upper_bound - tolerance
    ]
    if free_scores:
        return sum(free_scores) / len(free_scores)

    capped_scores = [
        score
        for score, alpha in zip(training_scores, alphas)
        if alpha >= upper_bound - tolerance
    ]
    zero_scores = [
        score for score, alpha in zip(training_scores, alphas) if alpha <= tolerance
    ]
    if capped_scores and zero_scores:
        return (max(capped_scores) + min(zero_scores)) / 2.0
    if capped_scores:
        return max(capped_scores)

    support_scores = [
        score for score, alpha in zip(training_scores, alphas) if alpha > tolerance
    ]
    return sum(support_scores) / len(support_scores)


class OneClassSVMScratch:
    """使用线性核和投影梯度求解对偶问题的一类支持向量机。

    ``nu`` 是训练异常比例上界和支持向量比例下界的控制参数，并不强制模型
    输出固定比例的异常。``score_samples`` 返回未减偏移量的正常性分数，
    ``decision_function`` 大于等于零时预测为正常样本 ``1``。
    """

    def __init__(self, nu=0.1, learning_rate=1.0, max_iter=500, tol=1e-6):
        if (
            not isinstance(nu, (int, float))
            or not math.isfinite(nu)
            or not 0.0 < nu <= 1.0
        ):
            raise ValueError("nu must be in (0, 1]")
        if (
            not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be positive")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if (
            not isinstance(tol, (int, float))
            or not math.isfinite(tol)
            or tol < 0.0
        ):
            raise ValueError("tol must be non-negative")

        self.nu = nu
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.alphas_ = []
        self.coef_ = []
        self.offset_ = None
        self.support_indices_ = []
        self.support_vectors_ = []
        self.labels_ = []
        self.objective_history_ = []
        self.n_iter_ = 0
        self._n_features_ = None
        self._is_fitted = False

    def fit(self, X):
        samples = _validate_matrix(X)
        sample_count = len(samples)
        self._n_features_ = len(samples[0])
        upper_bound = 1.0 / (self.nu * sample_count)
        kernel_matrix = [
            [_dot(first, second) for second in samples] for first in samples
        ]

        alphas = [1.0 / sample_count] * sample_count
        objective = _dual_objective(kernel_matrix, alphas)
        self.objective_history_ = [objective]
        lipschitz_bound = max(
            sum(abs(value) for value in row) for row in kernel_matrix
        )
        base_step = self.learning_rate / max(lipschitz_bound, 1e-12)

        for iteration in range(1, self.max_iter + 1):
            gradient = _matrix_vector_product(kernel_matrix, alphas)
            step = base_step
            candidate = alphas
            candidate_objective = objective
            found_descent = False
            for _ in range(40):
                candidate = _project_capped_simplex(
                    [
                        alpha - step * gradient_value
                        for alpha, gradient_value in zip(alphas, gradient)
                    ],
                    upper_bound,
                )
                candidate_objective = _dual_objective(kernel_matrix, candidate)
                if candidate_objective <= objective + 1e-14:
                    found_descent = True
                    break
                step *= 0.5
            if not found_descent:
                candidate = list(alphas)
                candidate_objective = objective

            change = max(
                abs(new - old) for new, old in zip(candidate, alphas)
            )
            alphas = candidate
            objective = candidate_objective
            self.objective_history_.append(objective)
            self.n_iter_ = iteration
            if change <= self.tol:
                break

        self.alphas_ = alphas
        self.coef_ = [
            sum(alpha * row[feature] for alpha, row in zip(alphas, samples))
            for feature in range(self._n_features_)
        ]
        training_scores = [_dot(self.coef_, row) for row in samples]
        self.offset_ = _dual_offset(
            training_scores,
            alphas,
            upper_bound,
        )
        self.support_indices_ = [
            index for index, alpha in enumerate(alphas) if alpha > 1e-8
        ]
        self.support_vectors_ = [list(samples[index]) for index in self.support_indices_]
        self._is_fitted = True
        self.labels_ = self.predict(samples)
        return self

    def score_samples(self, X):
        """返回线性超平面给出的未中心化正常性分数。"""
        if not self._is_fitted:
            raise ValueError("fit must be called before score_samples")
        samples = _validate_matrix(X, self._n_features_)
        return [_dot(self.coef_, row) for row in samples]

    def decision_function(self, X):
        return [score - self.offset_ for score in self.score_samples(X)]

    def predict(self, X):
        return [
            1 if decision >= -1e-12 else -1
            for decision in self.decision_function(X)
        ]

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)
