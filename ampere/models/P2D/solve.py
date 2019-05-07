import numpy as np

def p2d_fd(p, initial=None, tf=0, internal=False):
    from .P2D_fd import model
    if initial is None:
        input1 = np.concatenate([p,[1],[tf]])
    else:
        input1 = np.concatenate([p,[0],[tf], initial])
    # print(list(input1))
    N1 = p[25]
    N2 = p[26]
    N3 = p[27]
    Nr1 = p[28]
    Nr2 = p[29]
    # print(N1, N2, N3, Nr1, Nr2)
    # print(int(5*N1+2*N2+5*N3+N1*Nr1+N3*Nr2+15))
    var = np.zeros((10000, int(5*N1+2*N2+5*N3+N1*Nr1+N3*Nr2+15)))
    # print(zip(range(30),list(input1[:30])))
    # for i in range(35):
    #     print(i, input1[i])
    model(input1, var)
    count = np.nonzero(var[:,-2])[0][-1]+1
    var = var[:count]
    final = var[-1]
    # need to select: time, voltage, current
    out = var[:,[0,-2,-1]]
    out[:,-1] /= 17.1
    # out[:,1] -= var[:,6]
    if not internal:
        return [out, final]
    else:
        return [out, final, var]
