from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["client.py"]))
setup(ext_modules=cythonize(["common.py"]))
setup(ext_modules=cythonize(["dma.py"]))
setup(ext_modules=cythonize(["labels.py"]))
setup(ext_modules=cythonize(["process.py"]))
setup(ext_modules=cythonize(["render.py"]))
setup(ext_modules=cythonize(["triton.py"]))