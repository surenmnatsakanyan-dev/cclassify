import numpy as np
from ._naive_bayes import fit_nb_flat, predict_nb_flat


class GaussianNB:
    def __init__(self):
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        classes = np.unique(y)

        if classes.size == 0:
            raise ValueError("y must not be empty.")

        expected = np.arange(classes.size)
        if not np.array_equal(classes, expected):
            raise ValueError("Class labels must be encoded as 0, 1, ..., n_classes-1.")

        n_samples, n_features = X.shape
        n_classes = classes.size

        X_flat = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)
        y_contig = np.ascontiguousarray(y, dtype=np.int64)

        class_priors, means, variances = fit_nb_flat(
            X_flat,
            y_contig,
            n_samples,
            n_features,
            n_classes,
        )

        self.class_priors_ = class_priors
        self.means_ = means.reshape(n_classes, n_features)
        self.variances_ = variances.reshape(n_classes, n_features)
        self.n_features_ = n_features
        self.n_classes_ = n_classes
        self.classes_ = classes

        return self

    def predict(self, X):
        if self.class_priors_ is None:
            raise ValueError("Call fit before predict.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_:
            raise ValueError("Wrong number of features.")

        n_samples = X.shape[0]
        X_flat = np.ascontiguousarray(X.reshape(-1), dtype=np.float64)

        out = predict_nb_flat(
            X_flat,
            np.ascontiguousarray(self.class_priors_, dtype=np.float64),
            np.ascontiguousarray(self.means_.reshape(-1), dtype=np.float64),
            np.ascontiguousarray(self.variances_.reshape(-1), dtype=np.float64),
            n_samples,
            self.n_features_,
            self.n_classes_,
        )

        return out