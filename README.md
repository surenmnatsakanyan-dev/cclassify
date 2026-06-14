# cclassify

`cclassify` is a Python package that wraps C implementations of classical machine learning classification algorithms through Cython bindings.

The project focuses on three algorithms:

- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Binary Logistic Regression

The goal of the project is to combine:
- a user-friendly Python API,
- nontrivial compiled C code,
- Cython-based bindings,
- unit tests,
- and documentation suitable for a public GitHub repository.

## Features

- C implementations of core classification algorithms
- Python wrappers with NumPy-friendly interfaces
- Cython bridge between Python and C
- Unit tests with `pytest`
- Designed as a small educational machine learning package

## Implemented algorithms

### 1. KNNClassifier
A K-nearest-neighbors classifier using Euclidean distance and majority voting.

### 2. GaussianNB
A Gaussian Naive Bayes classifier for continuous features.  
The implementation estimates:
- class priors,
- class-wise feature means,
- class-wise feature variances.

Prediction is performed in log-space for numerical stability.

### 3. LogisticRegressionBinary
A binary logistic regression classifier trained with batch gradient descent.

## Project structure

```text
cclassify/
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── CHANGELOG.md
├── src/
│   └── cclassify/
│       ├── __init__.py
│       ├── knn.py
│       ├── naive_bayes.py
│       ├── logistic.py
│       ├── _knn.pyx
│       ├── _naive_bayes.pyx
│       ├── _logistic.pyx
│       └── csrc/
│           ├── knn.c
│           ├── knn.h
│           ├── naive_bayes.c
│           ├── naive_bayes.h
│           ├── logistic.c
│           └── logistic.h
├── tests/
└── docs/