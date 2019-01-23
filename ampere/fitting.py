import numpy as np
# from scipy.optimize import leastsq
# from scipy.optimize import minimize


def rmse(a, b):
    """
    A function which calculates the root mean squared error
    between two vectors.

    Notes
    ---------
    .. math::

        RMSE = \\sqrt{\\frac{1}{n}(a-b)^2}
    """
    assert isinstance(a, np.ndarray), \
        'This function expects a numpy array as input'
    assert isinstance(b, np.ndarray), \
        'This function expects a numpy array as input'
    assert len(a) == len(b), \
        'a and b should be the same length, but got %i, %i' % (len(a), len(b))
    return(np.sqrt(np.mean(np.square(a-b))))



#     try:
#         # replace only the parameters to be estimated
#         x = np.copy(initial)
#         x[inds] = x0
#         # pass all values to the model
#         error = rmse(model(x, t_exp), v_exp)
#         if verbose:
#             print(list(x0), error)
#         return error
#     except:
#         # if the solver fails, let the optimizer know with a large error value
# #        print('failed')
#         error = 500
#         return error
