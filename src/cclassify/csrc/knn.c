#include "knn.h"
#include <stdlib.h>
#include <math.h>

double squared_euclidean_distance(
    const double *a,
    const double *b,
    size_t n_features)
{
    double d = 0.0;

    for (size_t i = 0; i < n_features; ++i)
    {
        double diff = a[i] - b[i];
        d += diff * diff;
    }

    return d;
}

static size_t index_of_largest(const double *arr, size_t n)
{
    size_t idx = 0;
    for (size_t i = 1; i < n; ++i)
    {
        if (arr[i] > arr[idx])
        {
            idx = i;
        }
    }
    return idx;
}

static long majority_vote(const long *labels, size_t k)
{
    long best_label = labels[0];
    size_t best_count = 1;

    for (size_t i = 0; i < k; ++i)
    {
        long current_label = labels[i];
        size_t count = 0;

        for (size_t j = 0; j < k; ++j)
        {
            if (labels[j] == current_label)
            {
                ++count;
            }
        }

        if (count > best_count)
        {
            best_count = count;
            best_label = current_label;
        }
    }

    return best_label;
}

void knn_predict(
    const double *X_train,
    const long *y_train,
    const double *X_test,
    long *out,
    size_t n_train,
    size_t n_test,
    size_t n_features,
    size_t k)
{
    if (k == 0 || k > n_train)
    {
        return;
    }

    double *best_dist = (double *)malloc(k * sizeof(double));
    long *best_labels = (long *)malloc(k * sizeof(long));

    if (best_dist == NULL || best_labels == NULL)
    {
        free(best_dist);
        free(best_labels);
        return;
    }

    for (size_t i = 0; i < n_test; ++i)
    {
        const double *x_test_i = X_test + i * n_features;

        for (size_t t = 0; t < k; ++t)
        {
            best_dist[t] = INFINITY;
            best_labels[t] = 0;
        }

        for (size_t j = 0; j < n_train; ++j)
        {
            const double *x_train_j = X_train + j * n_features;

            double d = squared_euclidean_distance(
                x_test_i,
                x_train_j,
                n_features);

            size_t worst_idx = index_of_largest(best_dist, k);

            if (d < best_dist[worst_idx])
            {
                best_dist[worst_idx] = d;
                best_labels[worst_idx] = y_train[j];
            }
        }

        out[i] = majority_vote(best_labels, k);
    }

    free(best_dist);
    free(best_labels);
}