import numpy as np


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
    return np.sqrt(np.mean(np.square(a-b)))

