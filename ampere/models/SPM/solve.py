import numpy as np
from typing import Tuple, List
from .SPM_par import model as SPM_par
from .SPM_fd_sei import model as SPM_fd_sei
from .SPM_fd import model as SPM_fd


def spm_parabolic(p: np.ndarray, initial_state=(), tf=0, internal: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # initial_state is ignored if it is not defined
    model_inputs = np.concatenate([p, [0], [tf], initial_state])
    result_dimensionality = len(initial_state) + 1
    expected_max_result_length = 10000

    # pre-allocate the array for the C library to fill
    model_results = np.zeros((expected_max_result_length, result_dimensionality))
    SPM_par(model_inputs, model_results)

    # throw out the extra values
    count = np.nonzero(model_results[:, 1])[0][-1] + 1
    model_results = model_results[:count]

    final_state = model_results[-1]
    time_ind, pos_potential_ind, neg_potential_ind, curr_ind = 0, 5, 6, 7
    out = model_results[:, [time_ind, pos_potential_ind, curr_ind]]

    # calculate the external voltage from the internal potential differences
    out[:, 1] -= model_results[:, neg_potential_ind]
    return out, final_state, None


def spm_fd_sei(p, initial_state=None, tf=0, internal=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    model_inputs = np.concatenate([p, [0], [tf], initial_state])
    result_dimensionality = len(initial_state) + 1
    # print(len(initial_state), result_dimensionality)
    var = np.zeros((10000, result_dimensionality))
    SPM_fd_sei(model_inputs, var)
    count = np.nonzero(var[:, -2])[0][-1] + 1
    var = var[:count]
    final = var[-1]
    # need to select: time, voltage, current
    out = var[:, [0, -2, -1]]
    out[:, -1] /= 30.0
    # out[:,1] -= var[:,6]
    return out, final, var


def spm_fd(p, initial_state=None, tf=0, internal=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_inputs = np.concatenate([p, [0], [tf], initial_state])
    result_dimensionality = int(p[14] + p[15] + 9)

    var = np.zeros((10000, result_dimensionality))
    # var = np.zeros((10000, ))
    SPM_fd(model_inputs, var)
    count = np.nonzero(var[:, -2])[0][-1] + 1
    var = var[:count]
    final = var[-1]
    # need to select: time, voltage, current
    out = var[:, [0, -2, -1]]
    out[:, -1] /= 30.0

    return out, final, var
