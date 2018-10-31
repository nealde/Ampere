import numpy as np
import os

def spm_par(p, initial=None, tf=0):
    from .SPM_par import model
    if initial is None:
        input1 = np.concatenate([p,[1],[tf]])
    else:
        input1 = np.concatenate([p,[0],[tf], initial])
    var = np.zeros((10000,8))
    model(input1, var)
    count = np.nonzero(var[:,1])[0][-1]+1
    var = var[:count]
    final = var[-1]
    out = var[:,[0,5,7]]
    out[:,1] -= var[:,6]
    return [out, final]

def spm_fd_sei(p, initial=None, tf=0):
    from .SPM_fd_sei import model
    if initial is None:
        input1 = np.concatenate([p,[1],[tf]])
    else:
        input1 = np.concatenate([p,[0],[tf], initial])
    var = np.zeros((10000,int(p[18]+p[19]+15)))
    model(input1, var)
    count = np.nonzero(var[:,-2])[0][-1]+1
    var = var[:count]
    final = var[-1]
    # need to select: time, voltage, current
    out = var[:,[0,-2,-1]]
    out[:,-1] /= 30.
    # out[:,1] -= var[:,6]
    return [out, final]
