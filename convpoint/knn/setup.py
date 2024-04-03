# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from setuptools import setup, find_packages,Extension
# from distutils.core import setup, Extension


ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",
       extra_compile_args = [ "-std=c++11", "-fopenmp"],
       extra_link_args=["-std=c++11","-fopenmp"],
  )]

setup(
    name = "KNN NanoFLANN",
    ext_modules = ext_modules,
    # include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    cmdclass = {'build_ext': build_ext},
)
