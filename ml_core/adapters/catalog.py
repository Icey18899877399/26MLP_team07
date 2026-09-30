"""Curated, bounded web defaults for the repository's 24 scratch estimators."""

from importlib import import_module


# A base and an optimized implementation are deliberately advertised separately.
FAMILIES = {
    "logistic_regression": ("Logistic Regression", "classification", "wdbc", "LogisticRegressionScratch"),
    "knn": ("K-Nearest Neighbors", "classification", "wdbc", "KNNScratch"),
    "gaussian_naive_bayes": ("Gaussian Naive Bayes", "classification", "wdbc", "GaussianNaiveBayesScratch"),
    "cart_decision_tree": ("CART Decision Tree", "classification", "wdbc", "CARTClassifierScratch"),
    "random_forest": ("Random Forest", "classification", "wdbc", "RandomForestClassifierScratch"),
    "linear_regression": ("Linear Regression", "regression", "concrete", "LinearRegressionScratch"),
    "gbdt_regression": ("Gradient Boosted Trees", "regression", "concrete", "GBDTRegressorScratch"),
    "mlp_regression": ("MLP Regression", "regression", "concrete", "MLPRegressorScratch"),
    "kmeans": ("K-Means", "clustering", "seeds", "KMeansScratch"),
    "dbscan": ("DBSCAN", "clustering", "seeds", "DBSCANScratch"),
    "isolation_forest": ("Isolation Forest", "anomaly_detection", "cardio", "IsolationForestScratch"),
    "one_class_svm": ("One-Class SVM", "anomaly_detection", "cardio", "OneClassSVMScratch"),
}

BASE_DEFAULTS = {
    "logistic_regression": {"learning_rate": 0.1, "max_iter": 300, "threshold": 0.5, "standardize": True},
    "knn": {"n_neighbors": 5, "standardize": True},
    "gaussian_naive_bayes": {"variance_floor": 1e-9, "standardize": True},
    "cart_decision_tree": {"max_depth": 5, "min_samples_split": 2, "standardize": False},
    "random_forest": {"n_estimators": 12, "max_depth": 5, "min_samples_split": 2, "max_features": "sqrt", "standardize": False},
    "linear_regression": {"learning_rate": 0.01, "max_iter": 300, "standardize": True, "train_sample_limit": 240},
    "gbdt_regression": {"n_estimators": 25, "learning_rate": 0.1, "max_depth": 2, "min_samples_split": 2, "train_sample_limit": 240},
    "mlp_regression": {"hidden_size": 8, "learning_rate": 0.03, "max_iter": 100, "standardize": True, "train_sample_limit": 240},
    "kmeans": {"n_clusters": 3, "max_iter": 100, "standardize": True},
    "dbscan": {"eps": 1.0, "min_samples": 5, "standardize": True},
    "isolation_forest": {"n_estimators": 30, "max_samples": 128, "contamination": 0.1, "train_sample_limit": 400, "standardize": True},
    "one_class_svm": {"nu": 0.1, "learning_rate": 1.0, "max_iter": 100, "tol": 1e-6, "train_sample_limit": 72, "standardize": True},
}

OPTIMIZED_DEFAULTS = {
    "logistic_regression": {**BASE_DEFAULTS["logistic_regression"], "l2": 0.0, "tol": 1e-8, "class_weight": None},
    "knn": {**BASE_DEFAULTS["knn"], "p": 2, "weights": "distance"},
    "gaussian_naive_bayes": {"var_smoothing": 1e-9, "standardize": True},
    "cart_decision_tree": {**BASE_DEFAULTS["cart_decision_tree"], "min_samples_leaf": 2, "ccp_alpha": 0.0},
    "random_forest": {**BASE_DEFAULTS["random_forest"], "min_samples_leaf": 2, "voting": "soft", "oob_score": True},
    "linear_regression": {**BASE_DEFAULTS["linear_regression"], "batch_size": 32, "l2": 0.0, "tol": 1e-8},
    "gbdt_regression": {**BASE_DEFAULTS["gbdt_regression"], "min_samples_leaf": 2, "subsample": 0.8, "l2_regularization": 0.0},
    "mlp_regression": {"hidden_layer_sizes": [16, 8], "activation": "relu", "learning_rate": 0.01, "max_iter": 100, "batch_size": 32, "l2": 0.0, "standardize": True, "train_sample_limit": 240},
    "kmeans": {**BASE_DEFAULTS["kmeans"], "init": "k-means++", "n_init": 5, "tol": 1e-4},
    "dbscan": {**BASE_DEFAULTS["dbscan"], "algorithm": "kd_tree", "leaf_size": 20},
    "isolation_forest": {**BASE_DEFAULTS["isolation_forest"], "n_split_candidates": 3, "max_features": 1.0},
    "one_class_svm": {"nu": 0.1, "kernel": "rbf", "gamma": "scale", "max_iter": 100, "tol": 1e-6, "standardize": True, "train_sample_limit": 72},
}

DESCRIPTIONS = {
    "learning_rate": "Positive optimizer step size (maximum 1).",
    "max_iter": "Maximum training iterations (1–2000); curves show only iterations actually run.",
    "standardize": "Fit feature scaling on training data only (all rows for clustering). Base linear one-class SVM uses scale-only normalization.",
    "threshold": "Probability threshold for the positive class, between 0 and 1.",
    "n_neighbors": "Number of nearest training neighbors (1–100).",
    "p": "Minkowski distance exponent (1 or 2).",
    "weights": "Neighbor vote weighting: uniform or distance.",
    "variance_floor": "Strictly positive minimum Gaussian variance.",
    "var_smoothing": "Non-negative Gaussian variance smoothing.",
    "max_depth": "Maximum tree depth (1–12).",
    "min_samples_split": "Minimum samples required to split a tree node (at least 2).",
    "min_samples_leaf": "Minimum samples in a tree leaf.",
    "n_estimators": "Number of trees or boosting rounds (1–100).",
    "max_features": "Features per split: sqrt/log2 for random forest; fraction (0,1] for isolation forest.",
    "l2": "Non-negative L2 penalty.",
    "l2_regularization": "Non-negative leaf weight regularization.",
    "tol": "Non-negative convergence tolerance.",
    "class_weight": "null or balanced class weighting.",
    "ccp_alpha": "Non-negative cost-complexity pruning penalty.",
    "voting": "Forest voting method: soft or hard.",
    "oob_score": "Compute genuine out-of-bag diagnostics during fitting.",
    "batch_size": "Training batch size (1–256).",
    "train_sample_limit": "Deterministic cap applied after the hold-out split; evaluation uses all held-out rows. SVM selects only normal training rows.",
    "subsample": "Fraction of training rows per boosting round (0,1].",
    "hidden_size": "Number of base MLP hidden units (1–64).",
    "hidden_layer_sizes": "One to three hidden layer widths, each 1–64.",
    "activation": "Hidden activation: relu or tanh.",
    "n_clusters": "Number of clusters (cannot exceed dataset rows).",
    "init": "Centroid initialization: random or k-means++.",
    "n_init": "Number of independent K-Means initializations (1–20).",
    "eps": "Positive DBSCAN neighborhood radius in transformed feature space.",
    "min_samples": "Minimum neighborhood size, including the sample itself.",
    "algorithm": "Neighbor lookup: kd_tree or brute.",
    "leaf_size": "KD-tree leaf size.",
    "max_samples": "Maximum training rows sampled per isolation tree (2–400).",
    "contamination": "Training-score threshold quantile via expected anomaly fraction (0,0.5].",
    "n_split_candidates": "Candidate hyperplanes per isolation split (1–10).",
    "nu": "One-class SVM upper-bound fraction (0,1].",
    "kernel": "Kernel: rbf or linear.",
    "gamma": "RBF gamma: scale, auto, or a positive finite number.",
}


def estimator_class(family, variant):
    module = family + ("_optimized" if variant == "optimized" else "")
    name = ("Optimized" if variant == "optimized" else "") + FAMILIES[family][3]
    return getattr(import_module("Models." + module), name)
