import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "knn.h":
    void knn_predict(
        const double *X_train,
        const long *y_train,
        const double *X_test,
        long *out,
        size_t n_train,
        size_t n_test,
        size_t n_features,
        size_t k
    )

def predict_knn_flat(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X_train,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] y_train,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X_test,
    size_t n_train,
    size_t n_test,
    size_t n_features,
    size_t k
):
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out

    if X_train.shape[0] != n_train * n_features:
        raise ValueError("X_train length must be n_train * n_features.")
    if X_test.shape[0] != n_test * n_features:
        raise ValueError("X_test length must be n_test * n_features.")
    if y_train.shape[0] != n_train:
        raise ValueError("y_train length must be n_train.")
    if k == 0 or k > n_train:
        raise ValueError("k must satisfy 1 <= k <= n_train.")

    out = np.empty(n_test, dtype=np.int64)

    knn_predict(
        &X_train[0],
        <const long *> &y_train[0],
        &X_test[0],
        <long *> &out[0],
        n_train,
        n_test,
        n_features,
        k
    )

    return np.asarray(out)