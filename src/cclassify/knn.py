import numpy as np
from ._knn import predict_knn_flat


class KNNClassifier:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = k
        self.X_train = None
        self.y_train = None
        self.n_train = None
        self.n_features = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.n_train = X.shape[0]
        self.n_features = X.shape[1]

        if self.k > self.n_train:
            raise ValueError("k cannot be larger than the number of training samples.")

        self.X_train = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)
        self.y_train = np.ascontiguousarray(y, dtype=np.int64)
        return self

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Call fit before predict().")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of features as training data.")

        n_test = X.shape[0]
        X_test_flat = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)

        return predict_knn_flat(
            self.X_train,
            self.y_train,
            X_test_flat,
            self.n_train,
            n_test,
            self.n_features,
            self.k,
        )