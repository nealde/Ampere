import numpy as np
import os

from scipy.interpolate import interp1d

package_directory = os.path.dirname(os.path.abspath(__file__))

# def P2D(p, t):
#     # pass parameters to julia
#     run_str = "julia "+package_directory+"/models/P2D_EB/try1.jl "+ "".join(["{} ".format(par) for par in p])
#     a = os.popen(run_str).read()
#     data = str_to_array(a)
#     f = interp1d(data[0],data[1], bounds_error=False, fill_value=2.5)
#     # need a solution to extrapolating
#     voltage = f(t)
#     return voltage

# def P2D_ida(p, t, initial=None):
#     '''This function wraps the P2D exe which allows for continuous battery operation.
#     It handles switching between CC/CV if needed. This is determined by the time steps
#     and the current.
#     For the parameters passed to the exe, p[:14] are model parameters, p[14] is current,
#     and p[15] is whether to be in CC or CV mode.
#     p[16] is whether or not the model will inherit initial conditions, and the next 7 expected
#     parameters are the initial conditions, if initial = 0.'''
#     from .models.SPM.solve import spm_par_ida
#     data = spm_par_ida(p, initial=initial, tf=t[-1])
#     print(data[1][5]-data[1][6])
#     if data[0][-1, 0] < t[-1] and np.isclose(data[0][-1, 1], 4.2, rtol=1e-2):
#         pp = np.copy(p)
#         pp[15] = 0
#         data2 = spm_par_ida(pp, initial=data[1][1:]) # leave off time
#         data2[0][:,0] += data[0][-1,0]
#         data[0] = np.concatenate((data[0][:-1,:], data2[0]), axis=0)
#         # print(data[0])
#         data[1] = data2[1]
#
#     # returns a list with [][array_of_data] and [final values]]
#     if p[14] > 0:
#         voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=4.2)
#         current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
#     else:
#         voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=2.5)
#         current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
#     final_values = data[1]
#     return [voltage(t), current(t), final_values]

def SPM_par(p, t, initial=None):
    '''This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the exe, p[:14] are model parameters, p[14] is current,
    and p[15] is whether to be in CC or CV mode.
    p[16] is whether or not the model will inherit initial conditions, and the next 7 expected
    parameters are the initial conditions, if initial = 0.'''
    from .models.SPM.solve import spm_par
    data = spm_par(p, initial=initial, tf=t[-1])
    # print(data[0][-5:])
    # print(data)
    # print(data[1][5]-data[1][6])
    if data[0][-1, 0] < t[-1] and np.isclose(data[0][-1, 1], 4.2, rtol=1e-2):
        # print('true')
        pp = np.copy(p)
        pp[15] = 0
        data2 = spm_par(pp, initial=data[1][1:]) # leave off time
        # print(data2)
        data2[0][:,0] += data[0][-1,0]
        data[0] = np.concatenate((data[0][:-1,:], data2[0]), axis=0)
        # print(data[0])
        data[1] = data2[1]

    # returns a list with [][array_of_data] and [final values]]
    # print(data)
    if p[14] > 0:
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=4.2)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    else:
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=2.5)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    final_values = data[1]
    return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values]

def SPM_fd_sei(p, t, initial=None, internal=False):
    '''This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the exe, p[:14] are model parameters, p[14] is current,
    and p[15] is whether to be in CC or CV mode.
    p[16] is whether or not the model will inherit initial conditions, and the next 7 expected
    parameters are the initial conditions, if initial = 0.'''
    from .models.SPM.solve import spm_fd_sei
    # print(p)
    data = spm_fd_sei(p, initial=initial, tf=t[-1])

    if data[0][-1, 0] < t[-1] and np.isclose(data[0][-1, 1], 4.2, rtol=1e-2):
        # print('true')
        pp = np.copy(p)
        pp[21] = 0 # cc is p[23] now
        data2 = spm_fd_sei(pp, initial=data[1][1:]) # leave off time
        # print(data2)
        data2[0][:,0] += data[0][-1,0]
        data[0] = np.concatenate((data[0][:-1,:], data2[0]), axis=0)
        if internal:
            data[2] = np.concatenate((data[2][:-1,:], data2[2]), axis=0)
        # print(data[0])
        data[1] = data2[1]

    # returns a list with [][array_of_data] and [final values]]
    # print(data)
    if p[20] > 0: # if current is positive - current is now index 20
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=4.2)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    else:
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=2.5)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    final_values = data[1]
    if not internal:
        return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values]
    else:
        return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values, data[2]]


def SPM_fd(p, t, initial=None, internal=False):
    '''This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the exe, p[:14] are model parameters, p[14] is current,
    and p[15] is whether to be in CC or CV mode.
    p[16] is whether or not the model will inherit initial conditions, and the next 7 expected
    parameters are the initial conditions, if initial = 0.'''
    from .models.SPM.solve import spm_fd
    # print(p)
    data = spm_fd(p, initial=initial, tf=t[-1], internal=internal)


    if data[0][-1, 0] < t[-1] and np.isclose(data[0][-1, 1], 4.2, rtol=1e-2):
        # print('true')
        pp = np.copy(p)
        pp[17] = 0 # cc is p[17] now
        data2 = spm_fd(pp, initial=data[1][1:], internal=internal) # leave off time
        # print(data2)
        data2[0][:,0] += data[0][-1,0]
        data[0] = np.concatenate((data[0][:-1,:], data2[0]), axis=0)
        if internal:
            data[2] = np.concatenate((data[2][:-1,:], data2[2]), axis=0)
        # print(data[0])
        data[1] = data2[1]

    # returns a list with [][array_of_data] and [final values]]
    # print(data)
    if p[16] > 0: # if current is positive - current is now index 16
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=4.2)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    else:
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=2.5)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    final_values = data[1]
    if not internal:
        return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values]
    else:
        return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values, data[2]]

def P2D_fd(p, t, initial=None, internal=False):
    '''This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the exe, p[:14] are model parameters, p[14] is current,
    and p[15] is whether to be in CC or CV mode.
    p[16] is whether or not the model will inherit initial conditions, and the next 7 expected
    parameters are the initial conditions, if initial = 0.'''
    from .models.P2D.solve import p2d_fd
    # print(p)
    data = p2d_fd(p, initial=initial, tf=t[-1])

    cc_ind = 31
    curr_ind = 30

    if data[0][-1, 0] < t[-1] and np.isclose(data[0][-1, 1], 4.2, rtol=1e-2):
        # print('true')
        pp = np.copy(p)
        pp[cc_ind] = 0 # cc is p[17] now
        data2 = p2d_fd(pp, initial=data[1][1:]) # leave off time
        # print(data2)
        data2[0][:,0] += data[0][-1,0]
        data[0] = np.concatenate((data[0][:-1,:], data2[0]), axis=0)
        if internal:
            data[2] = np.concatenate((data[2][:-1,:], data2[2]), axis=0)
        # print(data[0])
        data[1] = data2[1]

    # returns a list with [][array_of_data] and [final values]]
    # print(data[0][:,0].shape)
    if p[curr_ind] > 0: # if current is positive - current is now index 16
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=4.2)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    else:
        voltage = interp1d(data[0][:,0], data[0][:,1], kind='cubic', bounds_error=False, fill_value=2.5)
        current = interp1d(data[0][:,0], data[0][:,2], kind='cubic', bounds_error=False, fill_value=0)
    final_values = data[1]
    if not internal:
        return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values]
    else:
        return [np.concatenate(([t], [voltage(t)], [current(t)]), axis=0), final_values, data[2]]
# def str_to_array(a):
#     """
#     parses a string and returns the result of a julia solution in [time, voltage]
#     """
#     t1 = a.split('[')[1][:-1]
#     t2 = a.split('[')[2][:-1]
#     out = np.array([[float(t) for t in t2.split(' ')],[float(t) for t in t1.split(',')]])
#     return out
