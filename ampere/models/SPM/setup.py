from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os

# delete old files
try:
    os.remove("SPM_fd.c")
    os.remove("SPM_fd_sei.c")
    os.remove("SPM_par.c")
except FileNotFoundError:
    pass

ida_dir = "../ida"
# setup(
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[Extension("P2D_fd", ["P2D_fd.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
# 				ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
# 							include_dirs=[numpy.get_include(), ida_dir])]
# )


#
extension = Extension(
    name="SPM_fd",
    sources=["SPM_fd.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
    				ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
    # sources=["P2D_fd.pyx"],
    library_dirs=['ida'],
    include_dirs=[numpy.get_include(), ida_dir])

setup(
    name='SPM_fd',
    # cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([extension])
)

extension = Extension(
    name="SPM_fd_sei",
    sources = ["SPM_fd_sei.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
    				ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
    # sources=["P2D_fd.pyx"],
    library_dirs=['ida'],
    include_dirs=[numpy.get_include(), ida_dir])

setup(
    name='SPM_fd_sei',
    # cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([extension])
)

extension = Extension(
    name="SPM_par",
    sources = ["SPM_par.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
    				ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
    # sources=["P2D_fd.pyx"],
    library_dirs=['ida'],
    include_dirs=[numpy.get_include(), ida_dir])

setup(
    name='SPM_par',
    # cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([extension])
)

# try:
#     os.remove("SPM_par.c")
#     os.remove("SPM_fd_sei.c")
#     os.remove("SPM_fd.c")
# except FileNotFoundError:
#     pass
#
# ida_dir = "../../../ida/"
# setup(
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[Extension("SPM_par", ["SPM_par.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
# 				ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
# 							include_dirs=[numpy.get_include(), ida_dir]),
# 				Extension("SPM_fd_sei", ["SPM_fd_sei.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
# 							ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
# 										include_dirs=[numpy.get_include(), ida_dir]),
# 				Extension("SPM_fd", ["SPM_fd.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
# 							ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
# 										include_dirs=[numpy.get_include(), ida_dir])]
# )
