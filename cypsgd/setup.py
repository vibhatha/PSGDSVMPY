from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('cypsgd/PSGDMinibatchAdam.pyx'), include_dirs=[numpy.get_include()])




# setup(
#     ext_modules=cythonize("cypsgd/PSGDMinibatchAdam.pyx"),
#     include_dirs=[numpy.get_include()]
# )

setup(
    ext_modules=cythonize("cypsgd/sgd.pyx"),
    include_dirs=[numpy.get_include()]
)
