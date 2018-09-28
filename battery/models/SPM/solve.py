import subprocess
import numpy as np
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


def spm_par_ida(p, initial=None, tf=0):
    # parameters, current, cc, initial, values
    # adjust current - given value is current density, in order to scale it to absolute current,
    # we must multiply by (ap*lp*Rp/3) - Rp = 3, ap = 6, Lp = 13   [3, 6, 13]
    # p[-2] = p[-2]*p[3]*p[6]*p[13]/3
    if initial is None:
        run_str = package_directory + "\\spm_flex " + "".join([str(pp)+" " for pp in p]) + "1 " + str(tf)
#         print(run_str)
    else:
        run_str = package_directory + "\\spm_flex " + "".join([str(pp)+" " for pp in p]) + "0 " + str(tf) + " "+"".join([str(pp)+" " for pp in initial])
        # print(run_str)
    # print(run_str)
    var = subprocess.Popen(run_str, stdout=subprocess.PIPE).stdout.read().decode("utf-8").split("\r\n")
    # get everything
    out = []
    for x in var:
        tmp = [a for a in x.split(" ") if a != '']
#         print(len(tmp))
        if len(tmp) == 8:
            out.append([float(tmp[0]), float(tmp[5])-float(tmp[6]), float(tmp[7])]) # time, pos_phi-neg_phi, current
    final = np.array([float(a) for a in var[-2].split(" ") if a != ''])
    return [np.array(out), final]
