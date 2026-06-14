#ifndef CCLASSIFY_NAIVE_BAYES_H
#define CCLASSIFY_NAIVE_BAYES_H

#include <stddef.h>

/*
 * Fit a Gaussian Naive Bayes model.
 *
 * X            : flattened row-major matrix of shape (n_samples, n_features)
 * y            : class labels of length n_samples, encoded as 0..n_classes-1
 * class_priors : output array of length n_classes
 * means        : output array of shape (n_classes, n_features), flattened
 * variances    : output array of shape (n_classes, n_features), flattened
 * n_samples    : number of samples
 * n_features   : number of features
 * n_classes    : number of classes
 */
void naive_bayes_fit(
    const double *X,
    const long *y,
    double *class_priors,
    double *means,
    double *variances,
    size_t n_samples,
    size_t n_features,
    size_t n_classes);

/*
 * Predict class labels using a fitted Gaussian Naive Bayes model.
 *
 * X            : flattened row-major matrix of shape (n_samples, n_features)
 * class_priors : class prior probabilities of length n_classes
 * means        : class-wise means of shape (n_classes, n_features), flattened
 * variances    : class-wise variances of shape (n_classes, n_features), flattened
 * out          : output labels of length n_samples
 * n_samples    : number of samples
 * n_features   : number of features
 * n_classes    : number of classes
 *
 * Prediction is based on the maximum log-posterior score.
 */
void naive_bayes_predict(
    const double *X,
    const double *class_priors,
    const double *means,
    const double *variances,
    long *out,
    size_t n_samples,
    size_t n_features,
    size_t n_classes);

#endif