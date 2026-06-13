import numpy as np
from cclassify.logistic import LogisticRegressionBinary


def test_logistic_binary_simple():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 0, 0, 1])

    clf = LogisticRegressionBinary(learning_rate=0.5, n_iter=2000)
    clf.fit(X, y)

    pred = clf.predict(X)
    assert pred.tolist() == [0, 0, 0, 1]


def test_logistic_binary_train_test_split():
    X_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 0.0],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [0.1, 0.2],   # expect 0
        [0.2, 0.9],   # expect 0
        [1.2, 1.1],   # expect 1
        [1.8, 0.2],   # expect 1
    ])

    clf = LogisticRegressionBinary(learning_rate=0.3, n_iter=3000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    assert pred.tolist() == [0, 0, 1, 1]    