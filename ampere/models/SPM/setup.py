from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
ida_dir = "../../../ida/"
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("SPM_par", ["SPM_par.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
				ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
							include_dirs=[numpy.get_include(), ida_dir]),
				Extension("SPM_fd_sei", ["SPM_fd_sei.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
							ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
										include_dirs=[numpy.get_include(), ida_dir]),
				Extension("SPM_fd", ["SPM_fd.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
							ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
										include_dirs=[numpy.get_include(), ida_dir])]
)
