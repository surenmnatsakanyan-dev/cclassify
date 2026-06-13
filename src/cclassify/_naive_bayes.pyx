import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "naive_bayes.h":
    void naive_bayes_fit(
        const double *X,
        const long *y,
        double *class_priors,
        double *means,
        double *variances,
        size_t n_samples,
        size_t n_features,
        size_t n_classes
    )

    void naive_bayes_predict(
        const double *X,
        const double *class_priors,
        const double *means,
        const double *variances,
        long *out,
        size_t n_samples,
        size_t n_features,
        size_t n_classes
    )

def fit_nb_flat(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] y,
    size_t n_samples,
    size_t n_features,
    size_t n_classes
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] class_priors
    cdef cnp.ndarray[cnp.float64_t, ndim=1] means
    cdef cnp.ndarray[cnp.float64_t, ndim=1] variances

    if X.shape[0] != n_samples * n_features:
        raise ValueError("X length must be n_samples * n_features.")
    if y.shape[0] != n_samples:
        raise ValueError("y length must be n_samples.")

    class_priors = np.empty(n_classes, dtype=np.float64)
    means = np.empty(n_classes * n_features, dtype=np.float64)
    variances = np.empty(n_classes * n_features, dtype=np.float64)

    naive_bayes_fit(
        &X[0],
        <const long *> &y[0],
        &class_priors[0],
        &means[0],
        &variances[0],
        n_samples,
        n_features,
        n_classes
    )

    return (
        np.asarray(class_priors),
        np.asarray(means),
        np.asarray(variances),
    )

def predict_nb_flat(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] X,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] class_priors,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] means,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] variances,
    size_t n_samples,
    size_t n_features,
    size_t n_classes
):
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out

    if X.shape[0] != n_samples * n_features:
        raise ValueError("X length must be n_samples * n_features.")
    if class_priors.shape[0] != n_classes:
        raise ValueError("class_priors length must be n_classes.")
    if means.shape[0] != n_classes * n_features:
        raise ValueError("means length must be n_classes * n_features.")
    if variances.shape[0] != n_classes * n_features:
        raise ValueError("variances length must be n_classes * n_features.")

    out = np.empty(n_samples, dtype=np.int64)

    naive_bayes_predict(
        &X[0],
        &class_priors[0],
        &means[0],
        &variances[0],
        <long *> &out[0],
        n_samples,
        n_features,
        n_classes
    )

    return np.asarray(out)