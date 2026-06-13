from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cclassify._knn",
        sources=[
            "src/cclassify/_knn.pyx",
            "src/cclassify/csrc/knn.c",
        ],
        include_dirs=[
            np.get_include(),
            "src/cclassify/csrc",
        ],
    ),
    Extension(
        "cclassify._naive_bayes",
        sources=[
            "src/cclassify/_naive_bayes.pyx",
            "src/cclassify/csrc/naive_bayes.c",
        ],
        include_dirs=[
            np.get_include(),
            "src/cclassify/csrc",
        ],
    ),
    Extension(
        "cclassify._logistic",
        sources=[
            "src/cclassify/_logistic.pyx",
            "src/cclassify/csrc/logistic.c",
        ],
        include_dirs=[
            np.get_include(),
            "src/cclassify/csrc",
        ],
    ),
]

setup(
    package_dir={"": "src"},
    packages= find_packages(where="src"),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3},
    ),
)