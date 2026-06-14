import numpy as np
from ._logistic import fit_binary_flat, predict_proba_binary_flat, predict_binary_flat


class LogisticRegressionBinary:
    """
    Binary logistic regression classifier.

    This classifier fits a linear decision boundary using batch gradient
    descent and the logistic sigmoid function.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Gradient descent step size.
    n_iter : int, default=1000
        Number of gradient descent iterations.
    threshold : float, default=0.5
        Probability threshold used to convert predicted probabilities
        into class labels.

    Notes
    -----
    The current implementation supports binary classification only,
    with labels encoded as 0 and 1.
    """

    def __init__(self, learning_rate=0.1, n_iter=1000, threshold=0.5):
        """
        Initialize the binary logistic regression classifier.

        Parameters
        ----------
        learning_rate : float, default=0.1
            Gradient descent step size.
        n_iter : int, default=1000
            Number of training iterations.
        threshold : float, default=0.5
            Classification threshold.

        Raises
        ------
        ValueError
            If learning_rate is not positive, if n_iter is not positive,
            or if threshold is not between 0 and 1.
        """        
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
        """
        Fit the binary logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary labels encoded as 0 or 1.

        Returns
        -------
        self : LogisticRegressionBinary
            Fitted classifier.

        Raises
        ------
        ValueError
            If X is not 2D, y is not 1D, if the number of samples does not
            match, or if labels are not in {0, 1}.
        """        
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
        """
        Predict class probabilities for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Predicted probabilities for classes 0 and 1, where the second
            column corresponds to the probability of class 1.

        Raises
        ------
        ValueError
            If fit has not been called, if X is not 2D, or if the number
            of features does not match the fitted model.
        """        
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
        """
        Predict binary class labels for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels in {0, 1}.

        Raises
        ------
        ValueError
            If fit has not been called, if X is not 2D, or if the number
            of features does not match the fitted model.
        """        
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