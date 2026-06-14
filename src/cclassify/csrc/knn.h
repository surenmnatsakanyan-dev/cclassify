#ifndef CCLASSIFY_KNN_H
#define CCLASSIFY_KNN_H

#include <stddef.h>

/*
 * Compute squared Euclidean distance between two feature vectors.
 *
 * a, b       : input vectors of length n_features
 * n_features : number of features
 *
 * returns the squared distance sum_j (a[j] - b[j])^2
 */
double squared_euclidean_distance(
    const double *a,
    const double *b,
    size_t n_features);

/*
 * Predict labels for X_test using k-nearest-neighbor classification.
 *
 * Parameters
 * ----------
 * X_train    : flattened training matrix of shape (n_train, n_features)
 * y_train    : training labels of length n_train
 * X_test     : flattened test matrix of shape (n_test, n_features)
 * out        : output labels of length n_test
 * n_train    : number of training samples
 * n_test     : number of test samples
 * n_features : number of features
 * k: neighborhood parameter
 */
void knn_predict(
    const double *X_train,
    const long *y_train,
    const double *X_test,
    long *out,
    size_t n_train,
    size_t n_test,
    size_t n_features,
    size_t k);

#endif