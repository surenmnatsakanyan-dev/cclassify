import numpy as np
from cclassify.naive_bayes import GaussianNB
import pytest

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

def test_gaussian_nb_invalid_labels():
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 2])

    clf = GaussianNB()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_gaussian_nb_predict_before_fit():
    clf = GaussianNB()
    X_test = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError):
        clf.predict(X_test)


def test_gaussian_nb_wrong_feature_count():
    X_train = np.array([[1.0, 1.0], [2.0, 2.0]])
    y_train = np.array([0, 1])

    clf = GaussianNB().fit(X_train, y_train)

    X_test = np.array([[1.0, 1.0, 3.0]])
    with pytest.raises(ValueError):
        clf.predict(X_test)    