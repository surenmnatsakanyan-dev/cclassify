#ifndef CCLASSIFY_LOGISTIC_H
#define CCLASSIFY_LOGISTIC_H

#include <stddef.h>

double sigmoid(double z);

void logistic_fit_binary(
    const double *X,
    const long *y,
    double *w,
    double *b,
    size_t n_samples,
    size_t n_features,
    double learning_rate,
    size_t n_iter);

void logistic_predict_proba_binary(
    const double *X,
    const double *w,
    double b,
    double *out,
    size_t n_samples,
    size_t n_features);

void logistic_predict_binary(
    const double *X,
    const double *w,
    double b,
    long *out,
    size_t n_samples,
    size_t n_features,
    double threshold);

#endif