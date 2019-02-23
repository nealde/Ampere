import setuptools
import numpy

import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# import _version

# USE_CYTHON = False
#
# if USE_CYTHON:
#     try:
#         from Cython.Distutils import build_ext
#     except ImportError:
#         print("failed")
#         if USE_CYTHON=='auto':
#             USE_CYTHON=False
#         else:
#             raise

cmdclass = { }
ext_modules = [ ]

with open("README.md", "r") as fh:
    long_description = fh.read()
ida_dir = "ida"
ida_requirements1 = [ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
			ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"]
# ida_requirements2 = [ida_dir+"ida.h",ida_dir+"ida_band.h",ida_dir+"ida_dense.h",ida_dir+"/ida_direct.h",ida_dir+"/ida_ic.h",ida_dir+"/ida_io.h",ida_dir+"/sundials_types.h", ida_dir+"/ida_impl.h",
#             			ida_dir+"/nvector_serial.h",ida_dir+"/sundials_band.h",ida_dir+"/sundials_dense.h",ida_dir+"/sundials_direct.h",ida_dir+"/sundials_math.h",ida_dir+"/sundials_nvector.h"]
# set up the cython compilation
# if USE_CYTHON:
# print(ida_requirements2)
ext_modules += [
    Extension("ampere.models.P2D.P2D_fd", ["ampere/models/P2D/P2D_fd.c", *ida_requirements1],
    include_dirs = [numpy.get_include()]),

    Extension("ampere.models.SPM.SPM_fd", ["ampere/models/SPM/SPM_fd.c", *ida_requirements1],
    include_dirs = [numpy.get_include()]),

    Extension("ampere.models.SPM.SPM_fd_sei", ["ampere/models/SPM/SPM_fd_sei.c", *ida_requirements1],
    include_dirs = [numpy.get_include()]),

    Extension("ampere.models.SPM.SPM_par", ["ampere/models/SPM/SPM_par.c", *ida_requirements1],
    include_dirs = [numpy.get_include()]),
    # libraries = [*ida_requirements2],
    # library_dirs = ["./ida"])#,  "ampere/models/P2D/p2d_fd_source.c", *ida_requirements2]),
# include_dirs=[numpy.get_include(), ida_dir+"/ida.h",ida_dir+"/ida_band.h",ida_dir+"/ida_dense.h",ida_dir+"/ida_direct.h",ida_dir+"/ida_ic.h",ida_dir+"/ida_io.h",
#             ida_dir+"/nvector_serial.h",ida_dir+"/sundials_band.h",ida_dir+"/sundials_dense.h",ida_dir+"/sundials_direct.h",ida_dir+"/sundials_math.h",ida_dir+"/sundials_nvector.h"]),
    # Extension("ampere.models.P2D.P2D_fd", ["ampere/models/P2D/P2D_fd.pyx", "ampere/models/P2D/p2d_fd_source.c", *ida_requirements],
    # include_dirs=[numpy.get_include(), ida_dir+"/ida.h",ida_dir+"/ida_band.h",ida_dir+"/ida_dense.h",ida_dir+"/ida_direct.h",ida_dir+"/ida_ic.h",ida_dir+"/ida_io.h",
    # 			ida_dir+"/nvector_serial.h",ida_dir+"/sundials_band.h",ida_dir+"/sundials_dense.h",ida_dir+"/sundials_direct.h",ida_dir+"/sundials_math.h",ida_dir+"/sundials_nvector.h"]),
# Extension("ampere.models.P2D.P2D_fd", ["ampere/models/P2D/P2D_fd.pyx", ida_dir+"/ida.c",ida_dir+"/ida_band.c",ida_dir+"/ida_dense.c",ida_dir+"/ida_direct.c",ida_dir+"/ida_ic.c",ida_dir+"/ida_io.c",
# 			ida_dir+"/nvector_serial.c",ida_dir+"/sundials_band.c",ida_dir+"/sundials_dense.c",ida_dir+"/sundials_direct.c",ida_dir+"/sundials_math.c",ida_dir+"/sundials_nvector.c"],
# 						include_dirs=[numpy.get_include(), ida_dir]),
    # Extension("ampere.models.SPM.SPM_par", [ "ampere/models/SPM/SPM_par.pyx" , "ampere/models/SPM/SPM_par_source.c", *ida_requirements1],
    # include_dirs=[numpy.get_include(), ida_dir]),
    # Extension("ampere.models.SPM.SPM_fd", [ "ampere/models/SPM/SPM_fd.pyx" , "ampere/models/SPM/SPM_fd_source.c",*ida_requirements1],
    # include_dirs=[numpy.get_include(), ida_dir]),
    # Extension("ampere.models.SPM.SPM_fd_sei", [ "ampere/models/SPM/SPM_fd_sei.pyx" , "ampere/models/SPM/SPM_fd_sei_source.c",*ida_requirements1],
    # include_dirs=[numpy.get_include(), ida_dir])
    # Extension("ampere.models.SPM.SPM_fd", [ "ampere/models/SPM/SPM_fd.pyx" ]),
    # Extension("ampere.models.SPM.SPM_fd_sei", [ "ampere/models/SPM/SPM_fd_sei.c" ]),
]
cmdclass.update({ 'build_ext': build_ext })
# else:
#     ext_modules += [
#         # Extension("ampere.models.SPM.SPM_par", [ "ampere/models/SPM/SPM_par.c" ]),
#         # Extension("ampere.models.SPM.SPM_fd", [ "ampere/models/SPM/SPM_fd.c" ]),
#         # Extension("ampere.models.SPM.SPM_fd_sei", [ "ampere/models/SPM/SPM_fd_sei.c" ]),
#         Extension("ampere.models.P2D.P2D_fd", [ "ampere/models/P2D/P2D_fd.c" ]),
#     ]
print(setuptools.find_packages())
setup(
    name="ampere",
    version="0.4.0",
    author="Neal Dawson-Elli",
    author_email="nealde@uw.edu",
    description="A Python package for working with battery discharge data and physics-based battery models",

    cmdclass = cmdclass,
    ext_modules = ext_modules,

    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nealde/Ampere",
    packages=[*setuptools.find_packages()],
    install_requires=['cython','matplotlib', 'numpy', 'scipy'],
    classifiers=(
        "Programming Language :: Python :: 3",
        'Programming Language :: Cython',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Mathematics',
    ),
    keywords = "battery numerical simulation modeling",
)
