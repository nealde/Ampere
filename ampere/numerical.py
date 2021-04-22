import numpy as np

from collections import namedtuple
from typing import List

from scipy.interpolate import interp1d

SimulationResult = namedtuple('SimulationResult', ('voltage_data', 'final_state', 'internal_states'))


def run_simulation(model, parameters: List[float], initial_state: List[float], time: List[float], charging: bool, return_internal: bool = False) -> SimulationResult: # noqa
    final_time = time[-1]
    voltage_time_data, final_state, internal_states = model(parameters, initial_state, tf=final_time, internal=return_internal)

    # if we are charging and we are still expecting time remaining in the simulation, re-run the charge
    # simulation in Constant Voltage mode
    if voltage_time_data[-1, 0] < final_time and np.isclose(voltage_time_data[-1, 1], 4.2, rtol=1e-2):
        pp = np.copy(parameters)
        pp[-1] = 0
        cv_voltage_time_data, cv_final_state, cv_internal_states = model(pp, initial_state=final_state[1:]) # leave off time

        cv_voltage_time_data[:,0] += voltage_time_data[-1,0]
        voltage_time_data = np.concatenate((voltage_time_data[:-1,:], cv_voltage_time_data), axis=0)
        final_state = cv_final_state

    fill_value = 4.2 if charging else 2.5
    voltage = interp1d(voltage_time_data[:,0], voltage_time_data[:,1], kind='cubic', bounds_error=False, fill_value=fill_value)
    current = interp1d(voltage_time_data[:,0], voltage_time_data[:,2], kind='cubic', bounds_error=False, fill_value=0)
    return SimulationResult(np.concatenate(([time], [voltage(time)], [current(time)]), axis=0), final_state, internal_states)


def spm_parabolic(p, t, initial=None, internal=False) -> SimulationResult:
    """This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the library, p[:-2] are model parameters, p[-2] is current, and p[-1] is
    whether the model is in CC or CV mode"""
    from .models.SPM.solve import spm_parabolic
    # for all models, -2 is the defined current. A positive current indicates charging.
    charging = p[-2] > 0
    return run_simulation(spm_parabolic, p, initial, t, charging, internal)


def spm_fd_sei(p, t, initial=None, internal=False) -> SimulationResult:
    """This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the library, p[:-2] are model parameters, p[-2] is current, and p[-1] is
    whether the model is in CC or CV mode."""
    from .models.SPM.solve import spm_fd_sei
    # for all models, -2 is the defined current. A positive current indicates charging.
    charging = p[-2] > 0
    return run_simulation(spm_fd_sei, p, initial, t, charging, internal)


def spm_fd(p, t, initial=None, internal=False) -> SimulationResult:
    """This function wraps the SPM exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the library, p[:-2] are model parameters, p[-2] is current, and p[-1] is
    whether the model is in CC or CV mode"""
    from .models.SPM.solve import spm_fd
    # for all models, -2 is the defined current. A positive current indicates charging.
    charging = p[-2] > 0
    return run_simulation(spm_fd, p, initial, t, charging, internal)


def p2d_fd(p, t, initial=None, internal=False) -> SimulationResult:
    """This function wraps the P2D exe which allows for continuous battery operation.
    It handles switching between CC/CV if needed. This is determined by the time steps
    and the current.
    For the parameters passed to the library, p[:-2] are model parameters, p[-2] is current, and p[-1] is
    whether the model is in CC or CV mode"""
    from .models.P2D.solve import p2d_fd
    # for all models, -2 is the defined current. A positive current indicates charging.
    charging = p[-2] > 0
    return run_simulation(p2d_fd, p, initial, t, charging, internal)

