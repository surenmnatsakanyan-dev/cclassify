#include "logistic.h"
#include <stdlib.h>
#include <math.h>

double sigmoid(double z)
{
    if (z >= 0.0)
    {
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    }
    else
    {
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

void logistic_fit_binary(
    const double *X,
    const long *y,
    double *w,
    double *b,
    size_t n_samples,
    size_t n_features,
    double learning_rate,
    size_t n_iter)
{
    double *grad_w = (double *)malloc(n_features * sizeof(double));
    if (grad_w == NULL)
    {
        return;
    }

    *b = 0.0;
    for (size_t j = 0; j < n_features; ++j)
    {
        w[j] = 0.0;
    }

    for (size_t iter = 0; iter < n_iter; ++iter)
    {
        for (size_t j = 0; j < n_features; ++j)
        {
            grad_w[j] = 0.0;
        }

        double grad_b = 0.0;

        for (size_t i = 0; i < n_samples; ++i)
        {
            const double *x_i = X + i * n_features;

            double z = *b;
            for (size_t j = 0; j < n_features; ++j)
            {
                z += w[j] * x_i[j];
            }

            double p = sigmoid(z);
            double err = p - (double)y[i];

            for (size_t j = 0; j < n_features; ++j)
            {
                grad_w[j] += err * x_i[j];
            }

            grad_b += err;
        }

        for (size_t j = 0; j < n_features; ++j)
        {
            grad_w[j] /= (double)n_samples;
            w[j] -= learning_rate * grad_w[j];
        }

        grad_b /= (double)n_samples;
        *b -= learning_rate * grad_b;
    }

    free(grad_w);
}

void logistic_predict_proba_binary(
    const double *X,
    const double *w,
    double b,
    double *out,
    size_t n_samples,
    size_t n_features)
{
    for (size_t i = 0; i < n_samples; ++i)
    {
        const double *x_i = X + i * n_features;

        double z = b;
        for (size_t j = 0; j < n_features; ++j)
        {
            z += w[j] * x_i[j];
        }

        out[i] = sigmoid(z);
    }
}

void logistic_predict_binary(
    const double *X,
    const double *w,
    double b,
    long *out,
    size_t n_samples,
    size_t n_features,
    double threshold)
{
    for (size_t i = 0; i < n_samples; ++i)
    {
        const double *x_i = X + i * n_features;

        double z = b;
        for (size_t j = 0; j < n_features; ++j)
        {
            z += w[j] * x_i[j];
        }

        double p = sigmoid(z);
        out[i] = (p >= threshold) ? 1L : 0L;
    }
}