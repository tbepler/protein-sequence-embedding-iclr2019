from distutils.core import setup

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages

setup(
    name="bepler",
    packages=find_packages(),
    ext_modules=cythonize(["bepler/metrics.pyx", "bepler/alignment.pyx"]),
    include_dirs=[np.get_include()],
)
