# cclassify

`cclassify` is a Python package that wraps C implementations of classical machine learning classification algorithms through Cython bindings.

## Project overview

The project implements three classical classifiers:

- **KNNClassifier**
- **GaussianNB**
- **LogisticRegressionBinary**

The main goal of the project is to combine:

- a user-friendly Python API,
- nontrivial compiled C code,
- Cython-based bindings,
- unit tests,
- and documentation suitable for a public GitHub repository and homepage.

## Motivation

This project was developed as a small educational machine learning package that demonstrates how low-level compiled C code can be integrated into a Python package.

Instead of relying only on pure Python implementations, `cclassify` moves the computational core into C and uses Cython as a bridge between Python and C. This makes the package a good example of combining performance-oriented systems programming with a high-level user-facing API.

## Implemented algorithms

### 1. KNNClassifier

A K-nearest-neighbors classifier using Euclidean distance and majority voting among the `k` nearest training samples.

### 2. GaussianNB

A Gaussian Naive Bayes classifier for continuous features.

The implementation estimates:

- class priors,
- class-wise feature means,
- class-wise feature variances.

Prediction is performed in log-space for numerical stability.

### 3. LogisticRegressionBinary

A binary logistic regression classifier trained with batch gradient descent.

## Package structure

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
```

## Installation
Clone the repository and install the package:

```bash
git clone https://github.com/surenmnatsakanyan-dev/cclassify.git
cd cclassify
python -m pip install .
```

## How to use

### KNN

```python
import numpy as np
from cclassify.knn import KNNClassifier

X_train = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [10.0, 10.0],
    [9.0, 10.0],
])
y_train = np.array([0, 0, 0, 1, 1])

X_test = np.array([
    [0.2, 0.1],
    [9.5, 9.8],
])

clf = KNNClassifier(k=3).fit(X_train, y_train)
print(clf.predict(X_test))
```

### Gaussian Naive Bayes

```python
import numpy as np
from cclassify.naive_bayes import GaussianNB

X_train = np.array([
    [1.0, 1.1],
    [1.2, 0.9],
    [0.8, 1.0],
    [4.9, 5.1],
    [5.2, 4.8],
    [5.0, 5.0],
])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [1.1, 1.0],
    [5.1, 4.9],
])

clf = GaussianNB().fit(X_train, y_train)
print(clf.predict(X_test))
```

### Binary Logistic Regression

```python
import numpy as np
from cclassify.logistic import LogisticRegressionBinary

X_train = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [2.0, 0.0],
])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [0.1, 0.2],
    [1.5, 1.2],
])

clf = LogisticRegressionBinary(learning_rate=0.3, n_iter=3000)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
```

## Testing

The package includes unit tests written with pytest.

```bash
pytest -v
```

## What I am proud of

- implementing three classifiers with C cores,
- exposing them through Cython,
- designing a clean Python package structure,
- and documenting/testing the package as a public project.

## Limitations

Current limitations include:

- GaussianNB expects class labels encoded as 0, 1, ..., n_classes-1,
- logistic regression is binary only,
- KNN currently uses a simple majority vote implementation.