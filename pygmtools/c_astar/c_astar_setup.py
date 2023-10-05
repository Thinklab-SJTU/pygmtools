from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from glob import glob

setup(
    name='c_astar function',
    ext_modules=cythonize(
        Extension(
            'c_astar',
            glob('*.pyx'),
            include_dirs=[np.get_include(),"."],
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-std=c++11"],
        ),
        language_level = "3",
    ),
    zip_safe=False,
)
