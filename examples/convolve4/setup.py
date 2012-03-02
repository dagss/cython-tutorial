# Run with:
#
#    python setup.py build_ext -i
#
# to compile Cython extensions in-place (useful during development)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

extensions = [
    Extension("cy_convolve", ["cy_convolve.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
    ]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)

# Note setuptools messes things up, as usual, and a hack is needed to
# have it work together with Cython, see, e.g.,
# https://github.com/pydata/pandas/fake_pyrex
