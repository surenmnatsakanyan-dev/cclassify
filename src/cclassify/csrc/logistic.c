#include "logistic.h"
#include <stdlib.h>
#include <math.h>

/*
 * Compute the logistic sigmoid function.
 *
 * z : input scalar
 *
 * returns 1 / (1 + exp(-z))
 */
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

/*
 * Fit a binary logistic regression model using batch gradient descent.
 *
 * X             : flattened row-major matrix of shape (n_samples, n_features)
 * y             : binary labels of length n_samples, encoded as 0 or 1
 * w             : output weight vector of length n_features
 * b             : output bias term
 * n_samples     : number of samples
 * n_features    : number of features
 * learning_rate : gradient descent step size
 * n_iter        : number of training iterations
 */
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

/*
 * Predict probabilities for the positive class in binary logistic regression.
 *
 * X          : flattened row-major matrix of shape (n_samples, n_features)
 * w          : fitted weight vector of length n_features
 * b          : fitted bias term
 * out        : output probabilities of length n_samples
 * n_samples  : number of samples
 * n_features : number of features
 */
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

/*
 * Predict binary class labels from a fitted logistic regression model.
 *
 * X          : flattened row-major matrix of shape (n_samples, n_features)
 * w          : fitted weight vector of length n_features
 * b          : fitted bias term
 * out        : output labels of length n_samples
 * n_samples  : number of samples
 * n_features : number of features
 * threshold  : probability threshold used for classification
 */
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