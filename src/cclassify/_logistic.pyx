import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "logistic.h":
    double sigmoid(double z)

    void logistic_fit_binary(
        const double *X,
        const long *y,
        double *w,
        double *b,
        size_t n_samples,
        size_t n_features,
        double learning_rate,
        size_t n_iter
    )

    void logistic_predict_proba_binary(
        const double *X,
        const double *w,
        double b,
        double *out,
        size_t n_samples,
        size_t n_features
    )

    void logistic_predict_binary(
        const double *X,
        const double *w,
        double b,
        long *out,
        size_t n_samples,
        size_t n_features,
        double threshold
    )

def fit_binary_flat(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] y,
    size_t n_samples,
    size_t n_features,
    double learning_rate,
    size_t n_iter
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w
    cdef cnp.ndarray[cnp.float64_t, ndim=1] b_arr

    if X.shape[0] != n_samples * n_features:
        raise ValueError("X length must be n_samples * n_features.")
    if y.shape[0] != n_samples:
        raise ValueError("y length must be n_samples.")

    w = np.empty(n_features, dtype=np.float64)
    b_arr = np.empty(1, dtype=np.float64)

    logistic_fit_binary(
        &X[0],
        <const long *> &y[0],
        &w[0],
        &b_arr[0],
        n_samples,
        n_features,
        learning_rate,
        n_iter
    )

    return np.asarray(w), float(b_arr[0])

def predict_proba_binary_flat(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] w,
    double b,
    size_t n_samples,
    size_t n_features
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out

    if X.shape[0] != n_samples * n_features:
        raise ValueError("X length must be n_samples * n_features.")
    if w.shape[0] != n_features:
        raise ValueError("w length must be n_features.")

    out = np.empty(n_samples, dtype=np.float64)

    logistic_predict_proba_binary(
        &X[0],
        &w[0],
        b,
        &out[0],
        n_samples,
        n_features
    )

    return np.asarray(out)

def predict_binary_flat(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] w,
    double b,
    size_t n_samples,
    size_t n_features,
    double threshold
):
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out

    if X.shape[0] != n_samples * n_features:
        raise ValueError("X length must be n_samples * n_features.")
    if w.shape[0] != n_features:
        raise ValueError("w length must be n_features.")

    out = np.empty(n_samples, dtype=np.int64)

    logistic_predict_binary(
        &X[0],
        &w[0],
        b,
        <long *> &out[0],
        n_samples,
        n_features,
        threshold
    )

    return np.asarray(out)