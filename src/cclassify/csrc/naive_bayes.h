#ifndef CCLASSIFY_NAIVE_BAYES_H
#define CCLASSIFY_NAIVE_BAYES_H

#include <stddef.h>

void naive_bayes_fit(
    const double *X,
    const long *y,
    double *class_priors,
    double *means,
    double *variances,
    size_t n_samples,
    size_t n_features,
    size_t n_classes);

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