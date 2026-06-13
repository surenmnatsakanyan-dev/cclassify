import numpy as np
from cclassify.knn import KNNClassifier

def test_knn_simple():
    X_train = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [9.0, 10.0],
    ])
    y_train = np.array([0, 0, 0, 1, 1])

    X_test = np.array([
        [0.2, 0.1],
        [9.5, 9.8],
    ])

    clf = KNNClassifier(k=3).fit(X_train, y_train)
    pred = clf.predict(X_test)

    assert pred.tolist() == [0, 1]