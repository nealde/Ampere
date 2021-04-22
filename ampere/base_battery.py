import numpy as np
import time

from collections import namedtuple
from typing import List, Dict

from .numerical import SimulationResult

ChargeResult = namedtuple('ChargeResult', ('time', 'voltage', 'current', 'internal'))
SampleResult = namedtuple('SampleResult', ('inputs', 'voltages', 'failed', 'currents'))


class BaseBattery:
    """Base class for physics-based battery models"""

    def __init__(
        self,
        verbose: bool
    ):

        self.estimate_parameters = []
        self.initial_parameters = dict()
        self.available_parameters = []
        self.initial_fit = 0
        self.bounds = None
        self.verbose = verbose
        self.initial = None
        self.conf_ = None
        self.inplace = None
        self.hist = []
        self.current_state = np.array([])
        self.charge_ICs = np.array([])
        self.discharge_ICs = np.array([])
        self.estimate_indices = []
        self.model = lambda p, t, initial=None, internal=False: SimulationResult
        self.internal_structure = dict()

        # parameters for fitting
        self.fitted_parameters = []
        self.initial_fit = False
        self.opt_result = None
        self.currents = []
        self.currents_type = []
        self.tol = 0
        self.maxiter = 0
        self.count = 0
        self.parameters_ = None
        self.loglist = []

        self.t_exp = []
        self.v_exp = []

    def _validate_estimate_parameters(self, estimate_parameters: List[str]):
        if estimate_parameters is not None:
            for key in estimate_parameters:
                assert key in self.initial_parameters, f"Invalid estimate key entered - double check {key}, not in {self.initial_parameters.keys()}"
            self.estimate_parameters = estimate_parameters
            self.estimate_indices = [i for i, p in enumerate(self.available_parameters) if p in estimate_parameters]
            message = 'Estimating the following parameters:\n' + '\n'.join(p for p in estimate_parameters)
            print(message)
        else:
            self.estimate_indices = list(range(len(self.initial)))
            if self.verbose:
                message = 'Estimating the following parameters:\n' + '\n'.join(p for p in self.initial_parameters)
                print(message)

    def charge(
        self,
        t=None,
        current=0.5,
        from_current_state=False,
        p=None,
        trim=False,
        internal=False,
    ) -> ChargeResult:
        """The base wrapper for the model, used for simple charging. Automatically
        transitions from cc to cv depending upon final time.

        Inputs              Description
        ------              ------------------------
        t                   A list or numpy array of times to sample at (optional)
        current             A value for charge current
        from_current_state  Whether or not the simulation should start from the
                                current state or not - defaults to False, which
                                simulates the battery from a discharged state
        p                   Defaults to none - present only for wrapping in an optimizer
        trim                Defaults to False, a value of true will remove the padded numbers
                                at the end, where False will allow padded values.

        Output              Description
        ------              ------------------------
        out                 A list of values for [time, voltage, current] of the simulation"""
        if t is None:
            t = np.linspace(0, 5000 * (1 / current), 500)
        if p is None:
            if self.initial_fit == 0:
                p = self.initial
            else:
                p = self.initial
                p[self.estimate_indices] = self.fitted_parameters
        initial_state = self.charge_ICs
        if from_current_state:
            initial_state = self.current_state

        result = self.model([*p, current, 1], t, initial=initial_state, internal=internal)
        self.current_state = result.final_state[1:]
        self.hist.append(result.voltage_data)
        if trim:
            result = SimulationResult(
                self._charge_trim(result.voltage_data),
                result.final_state,
                result.internal_states,
            )

        return ChargeResult(result.voltage_data[0], result.voltage_data[1], result.voltage_data[2], result.internal_states if internal else None)

    def _charge_trim(self, voltage_data):
        return voltage_data[:, voltage_data[2] > 0.01]

    def _discharge_trim(self, voltage_data):
        return voltage_data[:, ~(voltage_data[2] == 0)]

    def discharge(
        self,
        t=None,
        current=0.5,
        from_current_state=False,
        p=None,
        trim=False,
        internal=False,
    ):
        """The base wrapper for the model, used for simple discharging.

        Inputs              Description
        ------              ------------------------
        t                   A list or numpy array of times to sample at (optional)
        current             A value for charge current
        from_current_state  Whether or not the simulation should start from the
                                current state or not - defaults to False, which
                                simulates the battery from a discharged state
        p                   Defaults to none - present only for wrapping in an optimizer
        trim                Defaults to False, a value of true will remove the padded numbers
                                at the end, where False will allow padded values.

        Output              Description
        ------              ------------------------
        out                 A list of values for [time, voltage, current] of the simulation"""
        if t is None:
            t = np.linspace(0, 4000 * (1 / current), 500)
        if p is None:
            if self.initial_fit == 0:
                p = self.initial
            else:
                p = self.initial
                p[self.estimate_indices] = self.fitted_parameters
        initial_state = self.discharge_ICs
        if from_current_state:
            initial_state = self.current_state

        result = self.model([*p, current * -1, 1], t, initial=initial_state, internal=internal)
        self.current_state = result.final_state[1:]
        self.hist.append(result.voltage_data)
        if trim:
            result = SimulationResult(
                self._discharge_trim(result.voltage_data),
                result.final_state,
                result.internal_states,
            )
        return ChargeResult(result.voltage_data[0], result.voltage_data[1], result.voltage_data[2], result.internal_states if internal else None)

    def cycle(self, current=0.5, n=500, charge_first=False, p=None, trim=False):
        """This function calls either a charge then discharge, or a discharge followed
        by a change. When charge_first is set to False, it will start from a charged
        state and discharge, follower by a charge.  Otherwise, it will do them
        in reverse order.

        Inputs              Description
        ------              ------------------------
        current             A value for charge and discharge current. These must be the same.
        charge_first        Whether the charge simulation should be run first. Defaults to False.
        n                   The number of points to sample in each charge / discharge cycle. Time is
                                automatically calculated as tf=4000*(1/current), to ensure the
                                entire charge or discharge cycle is captured.
        p                   Defaults to none - present only for wrapping in an optimizer
        trim                Defaults to False, a value of true will remove the padded numbers
                                at the end, where False will allow padded values.

        Output              Description
        ------              ------------------------
        out                 A list of values for [time, voltage, current] of the simulation"""
        current = abs(current)
        if charge_first:
            solve = [
                self.charge(np.linspace(0, 5000 * (1 / current), n), p=p, trim=trim),
                self.discharge(np.linspace(0, 5000 * (1 / current), n), from_current_state=True, p=p, trim=trim)
            ]

            solve[-1].time += solve[0].time[-1]
            return np.concatenate(solve, axis=1)
        else:
            solve = [
                self.discharge(np.linspace(0, 5000 * (1 / current), n), p=p, trim=trim),
                self.charge(np.linspace(0, 5000 * (1 / current), n), from_current_state=True, p=p, trim=trim)
            ]

            solve[-1].time += solve[0].time[-1]
            return ChargeResult(np.concatenate([x.voltage for x in solve]),
                                np.concatenate([x.time for x in solve]),
                                np.concatenate([x.current for x in solve]),
                                None)

    def piecewise_current(
        self,
        times,
        currents,
        n_steps=50,
        from_current_state=False,
        p=None,
        internal=False,
    ):
        """This function wraps charge and discharge in order to chain them together to
        create the ability to simulate arbitrary piecewise currents. Only supports
        stair-style current stepping, ramps are not supported.
        Inputs
        -------     ----------------------
        times:      A list of values representing number of seconds at that current
        currents:   A list of values representing value of the current

        Example:

        spm.piecewise_current([50,100,30,40],[1,-0.5,.5,.3])

        where a negative current represents charging and a positive current
        represents discharge."""
        assert isinstance(times, list), "times must be a list"
        assert isinstance(currents, list), "Currents must be a list"
        assert len(times) == len(currents), "times and currents must be the same length"

        solve = []
        count = 0
        if not from_current_state:
            self.current_state = self.discharge_ICs
        for t, c in zip(times, currents):
            tt = np.linspace(0, t, n_steps)
            if c > 0:
                out = self.discharge(tt, current=c, from_current_state=True, p=p, internal=internal)
            else:
                out = self.charge(tt, current=c * -1, from_current_state=True, p=p, internal=internal)
            if count > 0:
                # add times together
                out = ChargeResult(out.time + solve[-1].time[-1], out.voltage, out.current, out.internal)
            solve.append(out)
            count += 1

        solve = ChargeResult(np.concatenate([x.time for x in solve]),
                             np.concatenate([x.voltage for x in solve]),
                             np.concatenate([x.current for x in solve]),
                             np.concatenate([x.internal for x in solve]) if internal else None)
        self.hist.append(solve)
        return solve

    def summary(self):
        """
        Returns a Latex render of the equations and node spacings
        used for the current model, in addition to a table
        containing the names and values of the parameters
        """

        return

    def opt_wrap(self, x0, currents_type, verbose=False):
        """The general optimization wrapping function - this function
        serves to call the numerical solution and calculate the
        root mean squared error at each experimental timepoint
        using cubic interpolation.  This ensures that the
        solver has sufficient accuracy while also allowing for
        direct comparisons with the experimental data."""
        from .fitting import rmse

        x = np.copy(self.initial)
        x[self.estimate_indices] = x0
        error = 0
        self.count += 1
        try:
            if currents_type == "constant":
                for t, v, c in zip(self.t_exp, self.v_exp, self.currents):
                    if c > 0:
                        self.current_state = self.discharge_ICs
                        solve = self.discharge(t, current=c, from_current_state=True, p=x)
                        error += rmse(solve.voltage, v)
                    else:
                        self.current_state = self.charge_ICs
                        solve = self.charge(t, current=-c, from_current_state=True, p=x)
                        error += rmse(solve.voltage, v)
            else:
                solve = self.piecewise_current(self.t_exp, self.currents, p=x)
                error += rmse(solve.voltage, self.v_exp)
            if verbose:
                print(error, x0)
        except IndexError:
            error = 100
            print(f"combination of parameters {x[self.estimate_indices]} failed")
        return error

    def fit(self, t_exp, v_exp, currents, currents_type="constant", method="Nelder-Mead", bounds=None, re=0, maxiter=100, tol=None):
        """Model-specific fitting function
        Parameters
        ----------
        t_exp: A numpy array containing the series of time values for experimental data

        v_exp: A numpy array containing the series of voltage values for experimental data

        method: A method to pass to Scipy.optimize.minimize, suggestions include:
            - 'Nelder-Mead'
            - 'SLSQP'
            - 'L-BFGS-B'

        bounds: Bounds for the given parameters
        """

        # enforce typing
        assert isinstance(t_exp, (np.ndarray, list)), "time array is not of type np.ndarray"
        assert isinstance(v_exp, (np.ndarray, list)), "voltage array is not of type np.ndarray"
        assert currents_type in [
            "constant",
            "piecewise",
        ], "currents_type should be either constant or piecewise"
        if currents_type == "constant":
            assert len(t_exp) == len(currents), "time and current lengths should match"
            assert len(t_exp) == len(v_exp), "time and voltage lengths should match"
        if currents_type == "piecewise":
            assert len(t_exp) == len(currents), "time samples and current samples should be of the same length"

        self.currents = currents
        self.currents_type = currents_type
        self.count = 0

        if re == 0:
            self.t_exp = t_exp
            self.v_exp = v_exp

        # # call the optimizer
        from scipy.optimize import minimize
        print(self.estimate_indices)
        print(self.initial[self.estimate_indices],)
        res = minimize(
            self.opt_wrap,
            self.initial[self.estimate_indices],
            args=(self.currents_type, self.verbose),
            bounds=bounds,
            method=method,
            options={"maxiter": int(maxiter)},
            tol=tol,
        )
        self.fitted_parameters = res.x
        self.initial_fit = True
        self.opt_result = res

        return self.opt_result

    def refit(self, method="Nelder-Mead", maxiter=100):
        assert self.initial_fit, "Please call fit before calling refit"
        self.initial[self.estimate_indices] = self.fitted_parameters
        self.fit(self.t_exp, self.v_exp, self.currents, method=method, re=1, maxiter=maxiter)
        return

    def generate_data(
        self,
        n,
        parameters,
        currents,
        loglist="auto",
        bounds=None,
        sample_type="sobol",
        distribution="uniform",
        sample_time=None,
        time_samples=100,
        summary=True,
        just_sample=False,
        verbose=False,
    ) -> SampleResult:
        """
        This function uses the existing Julia kernel to generate a set of data, similar to how the
        optimization function works. Since this julia kernel already exists, the calculations are note made in parallel.
        In the future, a separate file may exist which generates the data in parallel. It is recommended to call fit() first,
        in order to establish t_exp. Otherwise, time can be manually input.
        Parameters
        ----------
        n : the number of samples to make. For sobol, this is the total number, and for grid, this is the number of
                samples per grid dimension. Be careful using grid, as this can get out of hand very quickly (n^dim)

        loglist: A list of parameters which should be log spaced. Log spacing is advised for any variable changing
                    by more than 100x, i.e. max/min > 100.  If 'auto' is given, this will be detected automatically.
                    defaults to 'auto'.

        parameters: A list of parameters if desired modified parameters are different than the initialized set.

        bounds: Hard upper and lower limits for generating the data. If this is not given, +/- 30% is used. If a float is given,
                    +/- that amount is used. Custom bounds are recommended for good results, i.e., fewer failed samples.

        sample_type: defaults to 'sobol', options include 'grid' and 'saltelli'.

        distribution: defaults to 'uniform', describes the distribution of the sampling. Options include 'normal', 'uniform', and 'bi-modal'
                            (to be implemented)
        sample_time: The times to be interpolated for sampling. Filled values will be 2.5V, in keeping with the final
                        discharge voltage.  Defaults to linear spacing at 20% longer than t_exp
        time_samples: number of points in the linear timespacing, defaults to 100.
        """
        assert isinstance(n, int), "n must be an integer"
        assert isinstance(loglist, (str, list, bool)), 'loglist must be a list, False, or "auto"'
        assert sample_type in [
            "grid",
            "sobol",
            "saltelli",
            "random",
        ], "Available arguments for type are {}".format(["grid", "sobol", "saltelli", "random"])

        num_currents = len(currents)
        if num_currents > 1:
            print("multiple currents detected")
            # sort high to low in order to have failures (which are typically at high currents) block that data point
            currents = sorted(currents, reverse=True)

        # step 1 - handle the arguments
        generate_pars = parameters
        internal = False
        generate_inds = [i for i, x in enumerate(self.available_parameters) if x in generate_pars]
        assert generate_pars is not None
        print(generate_inds)

        # set up bounds (used for loglist)
        if bounds is None:
            self.bounds = [(x / 1.2, x * 1.2) for x in self.initial[generate_inds]]
        else:
            d1 = dict(zip(parameters, range(len(parameters))))
            bounds_inds = [d1[i] for i in [self.available_parameters[j] for j in generate_inds]]
            print(bounds_inds)
            self.bounds = [bounds[i] for i in bounds_inds]

        # set up log-spacing
        if loglist is False:
            self.loglist = list(np.zeros(len(generate_pars)))
        if isinstance(loglist, list):
            assert len(loglist) == len(
                generate_pars
            ), "expected loglist to be same length as generate_pars, \
            but got {} and {}".format(
                len(loglist), len(generate_pars)
            )
            self.loglist = loglist
        if loglist == "auto":
            self.loglist = [1 if x[0] / x[1] >= 100 else 0 for x in self.bounds]

        scaled_sample = self._generate_sample(n, sample_type, generate_pars)

        if just_sample:
            return SampleResult(scaled_sample, None, None, currents)

        voltages = []
        succeeded = []
        failed = []

        for parameter_set in scaled_sample:
            per_current_trial_voltage = []
            simulate_pars = np.copy(self.initial)
            simulate_pars[generate_inds] = self.log_descale_for_model(parameter_set)
            try:
                for i in currents:
                    if i > 0:
                        result = self.discharge(current=i, p=simulate_pars, internal=internal)
                        per_current_trial_voltage.append(result.voltage[:, np.newaxis])
                    else:
                        result = self.charge(current=-1 * i, p=simulate_pars, internal=internal)
                        per_current_trial_voltage.append(result.voltage[:, np.newaxis])
                succeeded.append(simulate_pars[generate_inds])
                voltages.append(np.concatenate(per_current_trial_voltage, axis=1))
            except IndexError:
                failed.append(simulate_pars[generate_inds])

        return SampleResult(np.array(succeeded), np.array(voltages), np.array(failed), currents)

    def _generate_sample(self, n, sample_type, generate_pars):
        # create the array using the spacing method of choice
        raw_sample = None
        if sample_type == "sobol":
            from sobol_seq import i4_sobol_generate

            raw_sample = i4_sobol_generate(len(generate_pars), n)
        elif sample_type == "saltelli":
            from SALib.sample import saltelli

            problem = {
                "names": generate_pars,
                "bounds": [[0, 1] for x in generate_pars],
                "num_vars": len(generate_pars),
            }
            raw_sample = saltelli.sample(problem, n, True)

        elif sample_type == "grid":
            from sklearn.utils.extmath import cartesian

            temp = np.linspace(0, 1, n)
            raw_sample = cartesian([temp for i in range(len(generate_pars))])

        elif sample_type == "random":
            raw_sample = np.random.random((n, len(generate_pars)))
        assert raw_sample is not None, "something went wrong - check that type is correct"
        print("expected shape is {}".format(raw_sample.shape))
        # map the raw array to bounds, adhering to log scaling rules
        scaled_sample = self.log_scale_matrix(raw_sample)
        return scaled_sample

    def abscale(self, matrix, a=-1, b=1):
        out = 0
        if matrix.shape == (matrix.shape[0],):
            matrix = matrix.reshape(-1, 1)
            out = 1
        new = np.zeros(matrix.shape)
        for i in range(matrix.shape[1]):
            new[:, i] = (b - a) * (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min()) + a
        if out == 0:
            return new
        else:
            return new[:, 0]

    def log_scale_matrix(self, matrix):
        """This function is used to resample the sampling methods into log
        space, if the value in loglist is a 1."""
        out = np.zeros(matrix.shape)
        for i in range(len(self.loglist)):
            if self.loglist[i] == 1:
                ub = np.log(self.bounds[i][1])
                lb = np.log(self.bounds[i][0])
                out[:, i] = self.abscale(matrix[:, i], ub, lb)
            else:
                ub = self.bounds[i][1]
                lb = self.bounds[i][0]
                out[:, i] = self.abscale(matrix[:, i], ub, lb)
        return out

    def log_descale_for_model(self, matrix):
        """This function takes in a matrix that has been log scaled and
        turns the log values back into normal values to pass to the model."""
        out = np.zeros(matrix.shape)
        for i in range(len(self.loglist)):
            if self.loglist[i] == 1:
                out[i] = np.exp(matrix[i])
            else:
                out[i] = matrix[i]
        return out

    def create_database(self, sample_result: SampleResult, filename: str, train_test_split: float = 0.3):
        """This function creates a dataset using pre-split data and saves it
        into a compressed h5 file with name filename. x, y, xt, yt are assumed
        to be numpy arrays."""
        import h5py

        with h5py.File(filename + ".hdf5", "w") as f:
            f.create_dataset("x", data=x, compression="gzip")
            f.create_dataset("xt", data=xt, compression="gzip")
            f.create_dataset("y", data=y, compression="gzip")
            f.create_dataset("yt", data=yt, compression="gzip")
        return

    def demonstrate_convergence(self, function):
        """This function seeks to demonstrate grid independence for the given
        charge / discharge pattern.  It will continue doubling the number of nodes
        in each region until the absolute error between subsequent discharges is less than 1e-6,
        giving an error if that does not occur.

        Parameters
        ----------
        function: A string describing the desired test

        Outputs
        -------
        Error as a function of nodes, as well as the minimum number of nodes needed
        for grid independence, or numerical convergence, at these conditions.

        """
        self.initial_discretization = self.initial[25:]
        self.nodes = []
        self.errors = []
        print(self.initial_discretization)
        for i in range(1, 10):
            self.nodes.append(self.initial_discretization * i)