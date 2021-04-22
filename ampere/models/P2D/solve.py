import numpy as np
from .P2D_fd import model as P2D_fd


def p2d_fd(p, initial_state=(), tf=0, internal=False):
    model_inputs = np.concatenate([p, [0], [tf], initial_state])
    N1 = p[25]
    N2 = p[26]
    N3 = p[27]
    Nr1 = p[28]
    Nr2 = p[29]

    model_outputs = np.zeros((10000, int(5*N1+2*N2+5*N3+N1*Nr1+N3*Nr2+15)))
    P2D_fd(model_inputs, model_outputs)

    count = np.nonzero(model_outputs[:,-2])[0][-1]+1
    model_outputs = model_outputs[:count]
    final = model_outputs[-1]
    # need to select: time, voltage, current
    out = model_outputs[:,[0,-2,-1]]
    out[:,-1] /= 17.1
    # out[:,1] -= var[:,6]
    return out, final, model_outputs
