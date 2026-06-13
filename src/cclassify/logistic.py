import numpy as np
from ._logistic import fit_binary_flat, predict_proba_binary_flat, predict_binary_flat


class LogisticRegressionBinary:
    def __init__(self, learning_rate=0.1, n_iter=1000, threshold=0.5):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be between 0 and 1.")

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.threshold = threshold

        self.w_ = None
        self.b_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("Binary logistic regression expects labels in {0, 1}.")

        n_samples, n_features = X.shape
        X_flat = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)
        y_contig = np.ascontiguousarray(y, dtype=np.int64)

        w, b = fit_binary_flat(
            X_flat,
            y_contig,
            n_samples,
            n_features,
            self.learning_rate,
            self.n_iter,
        )

        self.w_ = w
        self.b_ = b
        self.n_features_ = n_features
        return self

    def predict_proba(self, X):
        if self.w_ is None or self.b_ is None:
            raise ValueError("Call fit before predict_proba.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_:
            raise ValueError("Wrong number of features.")

        n_samples = X.shape[0]
        X_flat = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)

        p1 = predict_proba_binary_flat(
            X_flat,
            np.ascontiguousarray(self.w_, dtype=np.float64),
            self.b_,
            n_samples,
            self.n_features_,
        )

        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        if self.w_ is None or self.b_ is None:
            raise ValueError("Call fit before predict.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_:
            raise ValueError("Wrong number of features.")

        n_samples = X.shape[0]
        X_flat = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)

        return predict_binary_flat(
            X_flat,
            np.ascontiguousarray(self.w_, dtype=np.float64),
            self.b_,
            n_samples,
            self.n_features_,
            self.threshold,
        )