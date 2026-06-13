#include "naive_bayes.h"
#include <stdlib.h>
#include <math.h>

#define EPS 1e-9

void naive_bayes_fit(
    const double *X,
    const long *y,
    double *class_priors,
    double *means,
    double *variances,
    size_t n_samples,
    size_t n_features,
    size_t n_classes)
{
    size_t *class_counts = (size_t *)calloc(n_classes, sizeof(size_t));
    if (class_counts == NULL)
    {
        return;
    }

    for (size_t c = 0; c < n_classes; ++c)
    {
        class_priors[c] = 0.0;
        for (size_t j = 0; j < n_features; ++j)
        {
            means[c * n_features + j] = 0.0;
            variances[c * n_features + j] = 0.0;
        }
    }

    /* Count class sizes and accumulate sums for means */
    for (size_t i = 0; i < n_samples; ++i)
    {
        size_t c = (size_t)y[i];
        class_counts[c]++;

        for (size_t j = 0; j < n_features; ++j)
        {
            means[c * n_features + j] += X[i * n_features + j];
        }
    }

    /* Convert sums to means and compute priors */
    for (size_t c = 0; c < n_classes; ++c)
    {
        if (class_counts[c] == 0)
        {
            continue;
        }

        class_priors[c] = (double)class_counts[c] / (double)n_samples;

        for (size_t j = 0; j < n_features; ++j)
        {
            means[c * n_features + j] /= (double)class_counts[c];
        }
    }

    /* Accumulate squared deviations for variances */
    for (size_t i = 0; i < n_samples; ++i)
    {
        size_t c = (size_t)y[i];

        for (size_t j = 0; j < n_features; ++j)
        {
            double diff = X[i * n_features + j] - means[c * n_features + j];
            variances[c * n_features + j] += diff * diff;
        }
    }

    /* Finalize variances */
    for (size_t c = 0; c < n_classes; ++c)
    {
        if (class_counts[c] == 0)
        {
            continue;
        }

        for (size_t j = 0; j < n_features; ++j)
        {
            variances[c * n_features + j] /= (double)class_counts[c];

            if (variances[c * n_features + j] < EPS)
            {
                variances[c * n_features + j] = EPS;
            }
        }
    }

    free(class_counts);
}

void naive_bayes_predict(
    const double *X,
    const double *class_priors,
    const double *means,
    const double *variances,
    long *out,
    size_t n_samples,
    size_t n_features,
    size_t n_classes)
{
    const double log2pi = log(2.0 * M_PI);

    for (size_t i = 0; i < n_samples; ++i)
    {
        size_t best_class = 0;
        double best_score = -INFINITY;

        for (size_t c = 0; c < n_classes; ++c)
        {
            double score = log(class_priors[c]);

            for (size_t j = 0; j < n_features; ++j)
            {
                double mu = means[c * n_features + j];
                double var = variances[c * n_features + j];
                double xij = X[i * n_features + j];
                double diff = xij - mu;

                score += -0.5 * (log2pi + log(var) + (diff * diff) / var);
            }

            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        out[i] = (long)best_class;
    }
}