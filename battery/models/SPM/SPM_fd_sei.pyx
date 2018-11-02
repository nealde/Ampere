import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


cdef extern from "SPM_fd_sei.source":
    int main(double* input, double* output, int n)

def model(np.ndarray[double, ndim=1, mode="c"] input not None, np.ndarray[double, ndim=2, mode="c"] output not None):
    cdef int n

    n = input.shape[0]

    success = main(&input[0], &output[0,0], n)

    return success
