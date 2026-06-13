import numpy as np
from cclassify.naive_bayes import GaussianNB


def test_gaussian_nb_simple():
    X = np.array([
        [1.0, 1.1],
        [1.2, 0.9],
        [0.8, 1.0],
        [4.9, 5.1],
        [5.2, 4.8],
        [5.0, 5.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = GaussianNB().fit(X, y)
    pred = clf.predict(X)

    assert pred.tolist() == y.tolist()

def test_gaussian_nb_train_test_split():
    X_train = np.array([
        [1.0, 1.1],
        [1.2, 0.9],
        [0.8, 1.0],
        [4.9, 5.1],
        [5.2, 4.8],
        [5.0, 5.0],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [1.1, 1.0],   # expect 0
        [0.9, 1.2],   # expect 0
        [5.1, 4.9],   # expect 1
        [4.8, 5.2],   # expect 1
    ])

    clf = GaussianNB().fit(X_train, y_train)
    pred = clf.predict(X_test)

    assert pred.tolist() == [0, 0, 1, 1]    