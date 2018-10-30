import numpy as np
import os

def spm_par_ida(p, initial=None, tf=0):
    from .cos_module import model
    if initial is None:
        input1 = np.concatenate([p,[1],[tf]])
    else:
        input1 = np.concatenate([p,[0],[tf], initial])
    # if inplace is None:
    var = np.zeros((10000,8))
    # print(input1)
    # else:
    #     var = inplace
    #     var[:,1] = 0
    model(input1, var)
    # print(var.shape)
    # print(np.nonzero(var[:,1]))
    count = np.nonzero(var[:,1])[0][-1]+1
    # print(count)
    # count=0
    # for i in var[:,1]:
    #     count += 1
    #     if i == 0:
    #         break
    # print(count)
    var = var[:count]
    final = var[-1]
    out = var[:,[0,5,7]]
    out[:,1] -= var[:,6]
    return [out, final]
