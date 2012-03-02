# Run with:
#
#    python setup.py build_ext -i
#
# to compile Cython extensions in-place (useful during development)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

extensions = [
    Extension("integrate", ["integrate.pyx"]),
    Extension("funcs", ["funcs.pyx"], libraries=['m'])
    ]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)

# Note setuptools messes things up, as usual, and a hack is needed to
# have it work together with Cython, see, e.g.,
# https://github.com/pydata/pandas/fake_pyrex
