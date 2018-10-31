#from .fitting import circuit_fit, computeCircuit, calculateCircuitLength
#from .plotting import plot_nyquist
import matplotlib.pyplot as plt
import numpy as np

### to find confidence intervals for fit parameters:
# SE(Pi) = sqrt[(SS/DF) * conv(i,i)]
# Pi : ith adjustable parameter
# SS: sum of squarewd residuals
# DF: degrees of fredom (number of data points - number of parameters)
# Conv(i,i) : i-thj diagonal element of covariance matrix


class BaseBattery:
    """Base class for physics-based battery models"""
    def __init__(self, initial_parameters=None, estimate_parameters=None, name=None,
                algorithm=None, bounds=None, chemistry=None, verbose=False):
        # initalize class attributes
        # assert isinstance(initial_parameters, dict), \
        #     "Initial parameters must be a dictionary of name-value pairs"
#        self.initial_guess = initial_guess
        self.estimate_parameters = estimate_parameters
        # from .fitting import opt_wrap
        self.initial_fit = 0
        self.name = name
        self.bounds = bounds
        self.verbose = verbose
        self.initial = None
        self.conf_ = None
        self.inplace = None

        # self.opt = opt_wrap

    def _is_fit(self):
        """ check if model has been fit (parameters_ is not None) """
        if self.parameters_ is not None:
            return True
        else:
            return False

    def charge(self, t=None, current=0.5, from_current_state=False, p=None, trim=False):
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
            t = np.linspace(0,5000*(1/current),500)
        if p is None:
            if self.initial_fit == 0:
                p = self.initial
            else:
                p = self.initial
                p[self.estimate_inds] = self.fitted_parameters
        if from_current_state:
            solve = self.model([*p, current, 1], t, initial=self.current_state)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if not trim:
                return solve[0]
            else:
                return solve[0][:,solve[0][2]>0.01]
        else:
            solve = self.model([*p, current, 1], t, initial=self.charge_ICs)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if not trim:
                return solve[0]
            else:
                return solve[0][:,solve[0][2]>0.01]

    def discharge(self, t=None, current=0.5, from_current_state=False, p=None, trim=False):
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
            t = np.linspace(0,4000*(1/current),500)
        if p is None:
            if self.initial_fit == 0:
                p = self.initial
            else:
                p = self.initial
                p[self.estimate_inds] = self.fitted_parameters
        if from_current_state:
            solve = self.model([*p, current*-1, 1], t, initial=[*self.current_state[:-1], current*-1])
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if not trim:
                return solve[0]
            else:
                return solve[0][:,solve[0][2]<-0.01]
        else:
            solve = self.model([*p, current*-1, 1], t, initial=self.discharge_ICs)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if not trim:
                return solve[0]
            else:
                return solve[0][:,solve[0][2]<-0.01]

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
            solve = [self.charge(np.linspace(0,4000*(1/current), n), p=p, trim=trim)]
            solve.append(self.discharge(np.linspace(0,4000*(1/current), n), from_current_state=True, p=p, trim=trim))
            solve[-1][0] += solve[0][0,-1]
            return np.concatenate(solve, axis=1)
        else:
            solve = [self.discharge(np.linspace(0,4000*(1/current), n), p=p, trim=trim)]
            solve.append(self.charge(np.linspace(0,4000*(1/current), n), from_current_state=True, p=p, trim=trim))
            solve[-1][0] += solve[0][0,-1]
            return np.concatenate(solve, axis=1)

    def piecewise_current(self, times, currents, n_steps=50, from_current_state=False, p=None):
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
        assert isinstance(times, list), 'times must be a list'
        assert isinstance(currents, list), 'Currents must be a list'
        assert len(times) == len(currents), 'times and currents must be the same length'

        solve = []
        count = 0
        if not from_current_state:
            self.current_state = self.discharge_ICs
        for t, c in zip(times, currents):
            tt = np.linspace(0, t, n_steps)
            if c > 0:
                try:
                    out = self.discharge(tt, current=c, from_current_state=True, p=p)
                except IndexError:
                    out = [tt, np.ones(len(tt))*2.5, np.ones(len(tt))*c]
            else:
                try:
                    out = self.charge(tt, current=c*-1, from_current_state=True, p=p)
                except IndexError:
                    out = np.array([tt, np.ones(len(tt))*4.2, np.ones(len(tt))*c])
                    # print(solve[-1])
                    # print(out)
            if count > 0:
                out[0] += solve[-1][0,-1]

            solve.append(out)
            count += 1
        solve = np.concatenate(solve, axis=1)
        self.hist.append(solve)
        return solve




#
#     def summary(self):
#         """
#         Returns a Latex render of the equations and node spacings
#         used for the current model, in addition to a table
#         containing the names and values of the parameters
#         """
#
#         return
#
    def opt_wrap(self, x0, currents_type, verbose=False):
        """The general optimization wrapping function - this function
        serves to call the numerical solution and calculate the
        root mean squared error at each experimental timepoint
        using cubic interpolation.  This ensures that the
        solver has sufficient accuracy while also allowing for
        direct comparisons with the experimental data."""
        from .fitting import rmse
        x = np.copy(self.initial)
        # print(self.estimate_inds)
        x[self.estimate_inds] = x0
        error = 0
        if currents_type == 'constant':
            for t, v, c in zip(self.t_exp, self.v_exp, self.currents):
                if c > 0:
                    self.current_state = self.discharge_ICs
                    solve = self.discharge(t, current=c, from_current_state=True, p=x)
                    error += rmse(solve[1], v)
                else:
                    self.current_state = self.charge_ICs
                    solve = self.charge(t, current=-c, from_current_state=True, p=x)
                    error += rmse(solve[1], v)
        else:
            solve = self.piecewise_current(self.t_exp, self.currents, p=x)
            error += rmse(solve[1], self.v_exp)
        if verbose:
            print(error, x0)
        return error

    def fit(self, t_exp, v_exp, currents, currents_type='constant', method="Nelder-Mead", bounds=None, re=0, maxiter=100, tol=None, **kwargs):
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
        assert isinstance(t_exp,(np.ndarray, list)),\
            'time array is not of type np.ndarray'
        # assert isinstance(t_exp[0], (float, int, np.int32, np.float64)),\
        #     'time array does not contain a number'
        assert isinstance(v_exp, (np.ndarray, list)),\
            'voltage array is not of type np.ndarray'
    #    if not interpolate:
        assert currents_type in ['constant','piecewise'], \
            'currents_type should be either constant or piecewise'
        if currents_type == 'constant':
            assert len(t_exp) == len(currents), \
                'time and current lengths should match'
            assert len(t_exp) == len(v_exp), \
                'time and voltage lengths should match'
        if currents_type == 'piecewise':
            assert len(t_exp) == len(currents), \
                'time samples and current samples should be of the same length'
        self.currents = currents
        self.currents_type = currents_type
        self.tol = tol
        self.maxiter = maxiter

        if re == 0:
            self.t_exp = t_exp
            self.v_exp = v_exp

        # # call the optimizer
        from scipy.optimize import minimize
        res = minimize(self.opt_wrap, self.initial[self.estimate_inds], args=(self.currents_type, self.verbose),
                       bounds=bounds, method=method, tol=self.tol, options={'maxiter': self.maxiter})
        self.fitted_parameters = res.x
        self.error = res.fun
        self.initial_fit = 1
        self.opt_result = res

        return res
#
    def refit(self, method="Nelder-Mead", maxiter=100):
        assert self.initial_fit == 1, \
            'Please call fit before calling refit'
        self.initial[self.estimate_inds] = self.fitted_parameters
        self.fit(self.t_exp, self.v_exp, self.currents, method=method, re=1, maxiter=maxiter)
        return
#
#     def plot(self, time=None, exp=True, legend=True, initial=False):
#         """
#         Returns a matplotlib (or Altair) plot of the fit data and/or
#         experimental data.
#
#         Parameters
#         ----------
#         time: A list or numpy array of new time points at which the model should be solved
#
#         exp: A boolean which plots the experimental data (True) or does not (False)
#
#         legend: A boolean which determines if the legend should be shown.
#         """
#         import matplotlib.pyplot as plt
#         self.plot_vars = np.copy(self.initial)
#         self.plot_vars[self.estimate_inds] = self.fitted_parameters
#
#         if time is None:
#             time = self.t_exp
#
#         if exp:
#             plt.plot(time, self.v_exp, 'o', label = 'Experimental')
#
#         plt.plot(np.array(time), self.model(self.plot_vars, np.array(time)), label = 'Model')
# #            if time is None:
# #                plt.plot(self.t_exp, self.model(self.plot_vars, self.t_exp), label = 'Model')
# #            else:
# #                plt.plot(np.array(time), self.model(self.plot_vars, np.array(time)), label = 'Model')
# #        else:
# #            if time is None:
# #                plt.plot(self.t_exp, self.model(self.plot_vars, self.t_exp), label = 'Model')
# #            else:
# #                plt.plot(np.array(time), self.model(self.plot_vars, np.array(time)), label = 'Model')
#         if initial:
#             plt.plot(np.array(time), self.model(self.initial, np.array(time)), label = 'Initial')
#         if legend:
#             plt.legend()
#         plt.show()
#
#     def generate_data(self, filename, n, loglist='auto', pars=None, bounds=None,
#                       type='sobol', distribution='uniform', sample_time=None,
#                       time_samples=100):
#         """
#         This function uses the existing Julia kernel to generate a set of data, similar to how the
#         optimization function works. Since this julia kernel already exists, the calculations are note made in parallel.
#         In the future, a separate file may exist which generates the data in parallel. It is recommended to call fit() first,
#         in order to establish t_exp. Otherwise, time can be manually input.
#         Parameters
#         ----------
#         filename: The filename for the h5 file the data is saved in (using h5py)
#
#         n : the number of samples to make. For sobol, this is the total number, and for grid, this is the number of
#                 samples per grid dimension. Be careful using grid, as this can get out of hand very quickly (n^dim)
#
#         loglist: A list of parameters which should be log spaced. Log spacing is advised for any variable changing
#                     by more than 100x, i.e. max/min > 100.  If 'auto' is given, this will be detected automatically.
#                     defaults to 'auto'.
#
#         pars: A list of parameters if desired modified parameters are different than the initialized set.
#
#         bounds: Hard upper and lower limits for generating the data. If this is not given, +/- 30% is used. If a float is given,
#                     +/- that amount is used. Custom bounds are recommended for good results, i.e., fewer failed samples.
#
#         type: defaults to 'sobol', options include 'grid' and 'saltelli'.
#
#         distribution: defaults to 'uniform', describes the distribution of the sampling. Options include 'normal', 'uniform', and 'bi-modal'
#                             (to be implemented)
#         sample_time: The times to be interpolated for sampling. Filled values will be 2.5V, in keeping with the final
#                         discharge voltage.  Defaults to linear spacing at 20% longer than t_exp
#         time_samples: number of points in the linear timespacing,d efaults to 100.
#         """
#         assert isinstance(filename, str), \
#             'Filename must be type string'
#         assert isinstance(n, int), \
#             'n must be an integer'
#         assert isinstance(loglist, (str, list, bool)), \
#             'loglist must be a list, False, or "auto"'
#         assert type in ['grid', 'sobol', 'saltelli']
#         import time
#
#         self.generate_pars = None
#         self.n = n
#         # step 1 - handle the arguments
#         if pars is None:
#             self.generate_pars = self.estimate_pars
#             self.generate_inds = self.estimate_inds
#         else:
#             self.generate_pars = pars
#             self.generate_inds = [i for i, x in enumerate(self.pars) if x in self.generate_pars]
#         assert self.generate_pars is not None
#
#         # set up bounds (used for loglist)
#         if bounds is None:
#             if self.bounds is None:
#                 self.bounds = [(x/1.2, x*1.2) for x in self.initial[self.generate_inds]]
#         else:
#             self.bounds = bounds
#
#         # set up log-spacing
#         if loglist is False:
#             self.loglist = np.zeros(len(self.generate_pars))
#         if isinstance(loglist, list):
#             assert len(loglist) == len(self.generate_pars), \
#                 'expected loglist to be same length as generate_pars, \
#             but got {} and {}'.format(len(loglist), len(self.generate_pars))
#             self.loglist = loglist
#         if loglist == 'auto':
#             self.loglist = [1 if x[0]/x[1] > 100 else 0 for x in self.bounds]
#
#         # create the array using the spacing method of choice
#         self.raw_sample = None
#         if type == 'sobol':
#             from sobol_seq import i4_sobol_generate
#             self.raw_sample = i4_sobol_generate(len(self.generate_pars),
#                                                 self.n)
#         elif type == 'saltelli':
#             from SALib.sample import saltelli
#             problem = {'names': self.estimate_pars,
#                        'bounds': [[0, 1] for x in self.estimate_pars],
#                        'num_vars': len(self.estimate_pars)}
#             self.raw_sample = saltelli.sample(problem, self.n, True)
#
#         elif type == 'grid':
#             from sklearn.utils.extmath import cartesian
#             temp = np.linspace(0, 1, self.n)
#             self.raw_sample = cartesian([temp for i in range(len(self.generate_pars))])
#
#         elif type == 'random':
#             self.raw_sample = np.random.random((n,len(self.generate_pars)))
#
#         assert self.raw_sample is not None, \
#             'something went wrong - check that type is correct'
#         print('expected shape is {}'.format(self.raw_sample.shape))
#
#         # map the raw array to bounds, adhering to log scaling rules
#         self.scaled_sample = self.log_scale_matrix(self.raw_sample)
#
#         # pass the values to the solver and collect the outputs into arrays,
#         # making sure to handle exceptions. Scale time at the end
#         self.generate_time = None
#         if sample_time is None:
#             try:
#                 self.generate_time = np.linspace(0, self.t_exp[-1]*1.2, time_samples)
#             except NameError:
#                 self.generate_time = np.linspace(0, 10000, time_samples)
#         else:
#             self.generate_time = np.array(sample_time)
#         assert self.generate_time is not None, \
#             'Error - generate_time was not created properly'
#         outs = []
#         ins = []
#         # outs = np.zeros((self.generate_time.shape[0], time_samples))
#         # ins = np.zeros(self.generate_time.shape[0], len(self.generate_inds))
#
#         count = 0
#         st = time.time()
#         for parameter_set in self.scaled_sample:
#             current_pars = np.copy(self.initial)
#             current_pars[self.generate_inds] = parameter_set
#             outs.append(self.model(current_pars, self.generate_time))
#             ins.append(parameter_set)
#             count += 1
#             if count % (self.scaled_sample.shape[0]//20) == 0 and self.verbose:
#                 print('{} solutions completed of {} in {} seconds - {} total hours predicted'.format(count, self.scaled_sample.shape[0], time.time()-st, (time.time()-st)/3600/(count/self.scaled_sample.shape[0])))
#
#         # save the values to an h5 file
#         outs = np.array(outs)
#         ins = np.array(ins)
#
#         # break into test and train splits
#         inds = np.arange(outs.shape[0])
#         np.random.shuffle(inds)
#         train = inds[:outs.shape[0]*3//4]
#         test = inds[outs.shape[0]*3//4:]
#
#         x = outs[train]
#         xt = outs[test]
#         y = ins[train]
#         yt = ins[test]
#
#         self.create_database(x, y, xt, yt, filename)
#         return
#
#
#     def create_database(self, x, y, xt, yt, filename):
#         """This function creates a dataset using pre-split data and saves it
#         into a compressed h5 file with name filename. x, y, xt, yt are assumed
#         to be numpy arrays."""
#         import h5py
#         with h5py.File(filename+".hdf5", "w") as f:
#             f.create_dataset('x', data=x, compression='gzip')
#             f.create_dataset('xt', data=xt, compression='gzip')
#             f.create_dataset('y', data=y, compression='gzip')
#             f.create_dataset('yt', data=yt, compression='gzip')
#         return
#
#     def abscale(self, matrix, a=-1, b=1):
#         out = 0
#         if matrix.shape == (matrix.shape[0],):
#             matrix = matrix.reshape(-1, 1)
#             out = 1
#         new = np.zeros(matrix.shape)
#         for i in range(matrix.shape[1]):
#             new[:, i] = (b-a)*(matrix[:, i]-matrix[:, i].min())/(matrix[:, i].max()-matrix[:, i].min())+a
#         if out == 0:
#             return new
#         else:
#             return new[:, 0]
#
#     def log_scale_matrix(self, matrix):
#         """This function is used to resample the sampling methods into log
#         space, if the value in loglist is a 1."""
#         out = np.zeros(matrix.shape)
#         for i in range(len(self.loglist)):
#             if self.loglist[i] == 1:
#                 ub = np.log(self.bounds[i][1])
#                 lb = np.log(self.bounds[i][0])
#                 out[:, i] = self.abscale(matrix[:, i], ub, lb)
#             else:
#                 ub = self.bounds[i][1]
#                 lb = self.bounds[i][0]
#                 out[:, i] = self.abscale(matrix[:, i], ub, lb)
#         return out

class SingleParticleParabolic(BaseBattery):
    """An implementation of the Single Particle Model, solved with IDA.  This
    version supports CC-discharge, CC-CV charging, and access to the internal
    states of the battery.  It also allows for sequential cycling. """
    def __init__(self, initial_parameters=None, verbose=False, **kwargs):
        """Constructor for the Single Particle Model base class
        Parameters
        ----------
        initial_parameters: A dictionary of parameter names and values. Acceptable names for the
        parameters can be found below:
        | name   | description                                 | default value | Units           |
        |--------|---------------------------------------------|---------------|-----------------|
        | Dn     | Li+ Diffusivity in negative particle        | 3.9e-14       | cm^2/s          |
        | Dp     | Li+ Diffusivity in positive particle        | 1e-14         | cm^2/s          |
        | Rn     | Negative particle radius                    | 2e-6          | m               |
        | Rp     | Positive particle radius                    | 2e-6          | m               |
        | T      | Ambient Temperature                         | 303.15        | K               |
        | an     | Surface area of negative electrode          | 723600        | m^2/m^3         |
        | ap     | Surface area of positive electrode          | 885000        | m^2/m^3         |
        | ce     | Starting electrolyte Li+ concentration      | 1000          | mol/m^3         |
        | csnmax | Maximum Li+ concentration of negative solid | 30555         | mol/m^3         |
        | cspmax | Maximum Li+ concentration of positive solid | 51555         | mol/m^3         |
        | kn     | Negative electrode reaction rate            | 5.0307e-9     |m^2.5/(mol^0.5s) |
        | kp     | Positive electrode reaction rate            | 2.334e-9      |m^2.5/(mol^0.5s) |
        | ln     | Negative electrode thickness                | 88e-6         | m               |
        | lp     | Positive electrode thickness                | 80e-6         | m               |

        estimate_parameters: A list of strings representing the parameters that you wish to estimate.
        Defaults to None, which will allow for the estimation of all parameters except temperature.

        For both intiial_parameters and estimate_parameters, order does not matter.

        Example usage:
        spm = SingleParticle(initial_parameters=dictionary_of_parameter_label_value_pairs, est_pars=list_of_parameter_labels)

        A list of available keyword agruments (kwargs):



        """
        from .numerical import SPM_par
        from .fitting import rmse
        super().__init__(initial_parameters, **kwargs)
        self.model = SPM_par
        self.opt = self.opt_wrap
        self.verbose = verbose
        self.initial_fit = 0
        # self.inplace = np.zeros((10000,8))
        self.available_parameters = ['Dn','Dp','Rn','Rp','T','an','ap','ce','csnmax','cspmax','kn','kp','ln','lp']
        self.default_values = [3.9e-14, 1e-14, 2e-6, 2e-6, 303.15, 723600, 885000, 1000, 30550, 51555, 5.0307e-9, 2.334e-9, 88e-6, 80e-6]
        self.initial_parameters = dict(zip(self.available_parameters, self.default_values))
        if initial_parameters is not None:
            for key in initial_parameters.keys():
                assert set({key}).issubset(self.initial_parameters.keys()),\
                        "Invalid initial key entered - double check %s" % str(key)
            for key in initial_parameters.keys():
                self.initial_parameters[key] = initial_parameters[key]
        # initial enforces the order parameters are given to the model
        self.initial = np.array([self.initial_parameters[i] for i in self.available_parameters])
        if self.estimate_parameters is not None:
            for key in self.estimate_parameters:
                assert set({key}).issubset(self.initial_parameters.keys()),\
                        "Invalid estimate key entered - double check %s" % str(key)
            self.estimate_inds = [i for i, p in enumerate(self.available_parameters) if p in self.estimate_parameters]
            if self.verbose:
                print(self.estimate_parameters, self.estimate_inds)
        else:
            self.estimate_inds = list(range(len(self.initial)))
            if self.verbose:
                print(self.estimate_parameters, self.estimate_inds)
        self.charge_ICs = [4.95030611e+04, 3.05605527e+02, 4.93273985e+04, 3.55685791e+02, 3.78436346e+00, 7.86330739e-01, 1.00000000e+00]
        self.discharge_ICs = [2.51584754e+04, 2.73734963e+04, 2.51409091e+04, 2.73785043e+04, 4.26705391e+00, 6.70539113e-02, -1.00000000]
        self.hist = []


class SingleParticleFDSEI(BaseBattery):
    """An Finite Difference implementation of the Single Particle Model with SEI, solved with IDA.  This
    version supports CC-discharge, CC-CV charging, and access to the internal
    states of the battery.  It also allows for sequential cycling. See .charge, .discharge,
    .cycle, and .piecewise_current for more."""
    def __init__(self, initial_parameters=None, verbose=False, **kwargs):
        """Constructor for the Single Particle Model base class
        Parameters
        ----------
        initial_parameters: A dictionary of parameter names and values. Acceptable names for the
        parameters can be found below:
        | name      | description                                   | default value | Units            |
        |-----------|-----------------------------------------------|---------------|------------------|
        | Dp        | Li+ Diffusivity in positive particle          | 3.9e-14       | cm^2/s           |
        | Dn        | Li+ Diffusivity in negative particle          | 3.9e-14       | cm^2/s           |
        | cspmax    | Maximum Li concentration of positive solid    | 30555         | mol/m^3          |
        | csnmax    | Maximum Li concentration of negative solid    | 30555         | mol/m^3          |
        | lp        | Positive electrode thickness                  | 80e-6         | m                |
        | ln        | Negative electrode thickness                  | 88e-6         | m                |
        | Rp        | Positive particle radius                      | 2e-6          | m                |
        | Rn        | Negative particle radius                      | 2e-6          | m                |
        | T         | Ambient Temperature                           | 303.15        | K                |
        | ce        | Starting electrolyte Li+ concentration        | 1000          | mol/m^3          |
        | ap        | Surface area of positive electrode per volume | 885000        | m^2/m^3          |
        | an        | Surface area of negative electrode per volume | 723600        | m^2/m^3          |
        | M_sei     | Molecular weight of SEI                       | 0.026         | Kg/mol           |
        | rho_sei   | SEI Density                                   | 2.1e3         | Kg/m^3           |
        | Kappa_sei | SEI Ionic conductivity                        | 1             | S/m              |
        | k_sei     | rate constant of side reaction                | 1.5e-6        | C m/(mol*s)      |
        | kp        | Positive electrode reaction rate              | 2.334e-9      | m^2.5/(mol^0.5s) |
        | kn        | Negative electrode reaction rate              | 5.0307e-9     | m^2.5/(mol^0.5s) |
        | N1        | Number of FD nodes in positive particle       | 15            |                  |
        | N2        | Number of FD nodes in negative particle       | 15            |                  |

        estimate_parameters: A list of strings representing the parameters that you wish to estimate.
        Defaults to None, which will allow for the estimation of all parameters except temperature.

        For both intiial_parameters and estimate_parameters, order does not matter.

        Example usage:
        spm = SingleParticle(initial_parameters=dictionary_of_parameter_label_value_pairs, est_pars=list_of_parameter_labels)

        A list of available keyword agruments (kwargs):



        """
        from .numerical import SPM_fd_sei
        from .fitting import rmse
        super().__init__(initial_parameters, **kwargs)
        self.model = SPM_fd_sei
        self.opt = self.opt_wrap
        self.verbose = verbose
        self.initial_fit = 0
        TC = 30
        # self.inplace = np.zeros((10000,8))
        self.available_parameters = ['Dp','Dn','cspmax','csnmax','lp','ln','Rp','Rn','T','ce','ap','an','M_sei','rho_sei','Kappa_sei','kp','kn','ksei','N1','N2']
        self.default_values = [1e-14, 1e-14, 51555.0, 30555.0, 8e-05, 8.8e-05, 2e-06, 2e-06, 303.15, 1000.0, 885000.0, 723600.0, 0.026, 2100.0, 1.0, 2.334e-11, 8.307e-12, 1.5e-06, 30, 30]
        self.initial_parameters = dict(zip(self.available_parameters, self.default_values))
        if initial_parameters is not None:
            for key in initial_parameters.keys():
                assert set({key}).issubset(self.initial_parameters.keys()),\
                        "Invalid initial key entered - double check %s" % str(key)
            for key in initial_parameters.keys():
                self.initial_parameters[key] = initial_parameters[key]
        # initial enforces the order parameters are given to the model
        self.initial = np.array([self.initial_parameters[i] for i in self.available_parameters])
        if self.estimate_parameters is not None:
            for key in self.estimate_parameters:
                assert set({key}).issubset(self.initial_parameters.keys()),\
                        "Invalid estimate key entered - double check %s" % str(key)
            self.estimate_inds = [i for i, p in enumerate(self.available_parameters) if p in self.estimate_parameters]
            if self.verbose:
                print(self.estimate_parameters, self.estimate_inds)
        else:
            self.estimate_inds = list(range(len(self.initial)))
            if self.verbose:
                print(self.estimate_parameters, self.estimate_inds)
        self.charge_ICs = []
        N1 = int(self.initial[18])
        N2 = int(self.initial[19])
        for i in range(N1+2):
            self.charge_ICs.append(49503.111)
        for i in range(N1+2, N1+N2+4):
            self.charge_ICs.append(305.55)
        self.charge_ICs.append(3.67873289259766)    #phi_p
        self.charge_ICs.append(.182763748093840)    #phi_n
        self.charge_ICs.append(30)                  #iint
        self.charge_ICs.append(0)                   #isei
        self.charge_ICs.append(1e-10)               #delta_sei
        self.charge_ICs.append(0)                  #Q
        self.charge_ICs.append(0)                  #cm
        self.charge_ICs.append(0)                  #cf
        self.charge_ICs.append(3.0596914450382)   #pot
        self.charge_ICs.append(TC*1)   	  	  #it
        # self.charge_ICs = [4.95030611e+04, 3.05605527e+02, 4.93273985e+04, 3.55685791e+02, 3.78436346e+00, 7.86330739e-01, 1.00000000e+00]
        self.discharge_ICs=[]
        for i in range(N1+2):
            self.charge_ICs.append(2.51417672e+04)
        for i in range(N1+2, N1+N2+4):
            self.charge_ICs.append(2.73921225)
        self.discharge_ICs.append(4.26700382e+00)
        self.discharge_ICs.append(6.70038247e-02)
        self.discharge_ICs.append(2.65295200e-03)
        self.discharge_ICs.append(7.34704800e-03)
        self.discharge_ICs.append(1.63513920e-10)
        self.discharge_ICs.append(3.08271510e+01)
        self.discharge_ICs.append(3.08183958e+01)
        self.discharge_ICs.append(8.75512593e-03)
        self.discharge_ICs.append(4.20000000e+00)
        self.discharge_ICs.append(1.00000000e-02)
        # self.discharge_ICs = [2.51584754e+04, 2.73734963e+04, 2.51409091e+04, 2.73785043e+04, 4.26705391e+00, 6.70539113e-02, -1.00000000]
        self.hist = []








# class AnalyticalSPM(BaseBattery):
#     """Analytical version of the Single Particle Model"""
#     def __init__(self, initial_parameters, **kwargs):
#         """ Constructor for the Analytical SPM base class
#         Parameters
#         ----------
#         initial_parameters: A dictionary of parameter names and values. Acceptable names for the
#         parameters can be found below:
#             kn, kp, cspmax1, csnmax2, cspmax2, csnmax2
#             order does not matter.
#
#         exp: A boolean which plots the experimental data (True) or does not (False)
#         """
#         super().__init__(initial_parameters, **kwargs)
#         from .analytical import ASingleParticle
#         from .fitting import rmse, opt_wrap
#         self.model = ASingleParticle
#         self.opt = opt_wrap
#
# #        self.initial = initial_parameters
#         self.initial_fit = 0
#         self.available_parameters = ["kn","kp","cspmax1","csnmax1","cspmax2","csnmax2"]
#         if self.estimate_pars is not None:
#             assert set(self.estimate_pars).issubset(set(self.available_parameters)), \
#                 'Check estimate_pars: should be a subset of {}'.format(self.available_parameters)
#         self.initial = np.array([initial_parameters[i] for i in self.available_parameters])
#         if self.estimate_pars is not None:
#             self.estimate_inds = [i for i, p in enumerate(self.available_parameters) if p in self.estimate_pars]
#         else:
#             self.estimate_inds = list(range(len(self.initial)))
#         print(self.estimate_inds)
#        self.initial_values = np.array([initial_parameters[i] for i in self.available_parameters])
#
# class P2D_EB(BaseBattery):
#     """Analytical version of the Single Particle Model"""
#     def __init__(self, initial_parameters, **kwargs):
#         """ Constructor for the Analytical SPM base class
#         Parameters
#         ----------
#         initial_parameters: A dictionary of parameter names and values. Acceptable names for the
#         parameters can be found below:
#             D1 Dsn Dsp Rpn Rpp brugp brugs brugn ctn ctp efn efp en ep es current kn kp lp ln ls sigmn sigmp t1
#             order does not matter.
#
#         exp: A boolean which plots the experimental data (True) or does not (False)
#         """
#         super().__init__(initial_parameters, **kwargs)
#         from .numerical import P2D
#         from .fitting import rmse, opt_wrap
#         self.model = P2D
#         self.opt = opt_wrap
#
# #        self.initial = initial_parameters
#         self.initial_fit = 0
#         self.available_parameters = ['D1','Dsn','Dsp','Rpn','Rpp','brugp','brugs','brugn','ctn','ctp','efn','efp','en','ep','es','current','kn','kp','lp','ln','ls','sigmn','sigmp','t1']
#         assert len(self.available_parameters) == len(initial_parameters), \
#             'Check the length of the initial parameters - there should be {} but onle {} are present'.format(len(self.available_parameters, len(initial_parameters)))
#         if self.estimate_pars is not None:
#             assert set(self.estimate_pars).issubset(set(self.available_parameters)), \
#                 'Check estimate_pars: should be a subset of {}'.format(self.available_parameters)
#         self.initial = np.array([initial_parameters[i] for i in self.available_parameters])
#         if self.estimate_pars is not None:
#             self.estimate_inds = [i for i, p in enumerate(self.available_parameters) if p in self.estimate_pars]
#         else:
#             self.estimate_inds = list(range(len(self.initial)))
#         print(self.estimate_inds)
#
# class P2D_DE(BaseBattery):
#     """Analytical version of the Single Particle Model"""
#     def __init__(self, initial_parameters, **kwargs):
#         """ Constructor for the Analytical SPM base class
#         Parameters
#         ----------
#         initial_parameters: A dictionary of parameter names and values. Acceptable names for the
#         parameters can be found below:
#             D1 Dsn Dsp Rpn Rpp brugp brugs brugn ctn ctp efn efp en ep es current kn kp lp ln ls sigmn sigmp t1
#             order does not matter.
#
#         exp: A boolean which plots the experimental data (True) or does not (False)
#         """
#         super().__init__(initial_parameters, **kwargs)
#         from .numerical import P2D_ED
#         from .fitting import rmse, opt_wrap
#         import subprocess
#         import os
#         import time
#
#         self.model = P2D_ED
#         self.opt = opt_wrap
#         self.p = None
#
# #        self.initial = initial_parameters
#         self.initial_fit = 0
#         self.available_parameters = ['D1','Dsn','Dsp','Rpn','Rpp','brugp','brugs','brugn','ctn','ctp','efn','efp','en','ep','es','current','kn','kp','lp','ln','ls','sigmn','sigmp','t1']
#         assert len(self.available_parameters) == len(initial_parameters), \
#             'Check the length of the initial parameters - there should be {} but only {} are present'.format(len(self.available_parameters, len(initial_parameters)))
#         if self.estimate_pars is not None:
#             assert set(self.estimate_pars).issubset(set(self.available_parameters)), \
#                 'Check estimate_pars: should be a subset of {}'.format(self.available_parameters)
#         self.initial = np.array([initial_parameters[i] for i in self.available_parameters])
#         if self.estimate_pars is not None:
#             self.estimate_inds = [i for i, p in enumerate(self.available_parameters) if p in self.estimate_pars]
#         else:
#             self.estimate_inds = list(range(len(self.initial)))
#         print(self.estimate_inds)
#         # initialize the julia kernel
#         package_directory = os.path.dirname(os.path.abspath(__file__))
#         self.p = subprocess.Popen("julia "+package_directory+"/models/P2D_DE/test.jl")
#         time.sleep(45)
#     def shutdown(self):
#         if self.p != None:
#             p.kill()
# #    def fit(*args, **kwargs):
# #        if self.p
# #        super().fit(t_exp, v_exp, method="Nelder-Mead", bounds=None, re = 0, maxiter=100, **kwargs)
# class P2D_RFM_Cython(BaseBattery):
#     """Reformulated P2D model solved in Cython with analytical Jacobian and node spacing 2-1-2-5-5"""
#     def __init__(self, initial_parameters, **kwargs):
#         """ Constructor for the Analytical SPM base class
#         Parameters
#         ----------
#         initial_parameters: A dictionary of parameter names and values. Acceptable names for the
#         parameters can be found below:
#             D1 Dsn Dsp Rpn Rpp brugp brugs brugn ctn ctp efn efp en ep es current kn kp lp ln ls sigmn sigmp t1
#             order does not matter.
#
#         exp: A boolean which plots the experimental data (True) or does not (False)
#         """
#         super().__init__(initial_parameters, **kwargs)
#         from .numerical import P2D_rfm_cython
#         from .fitting import rmse, opt_wrap
#         import time
#
#         self.model = P2D_rfm_cython
#         self.opt = opt_wrap
#         self.p = None
#
# #        self.initial = initial_parameters
#         self.initial_fit = 0
#         self.available_parameters = ['D1','Dsn','Dsp','Rpn','Rpp','brugp','brugs','brugn','ctn','ctp','efn','efp','en','ep','es','current','kn','kp','lp','ln','ls','sigmn','sigmp','t1']
#         assert len(self.available_parameters) == len(initial_parameters), \
#             'Check the length of the initial parameters - there should be {} but only {} are present'.format(len(self.available_parameters, len(initial_parameters)))
#         if self.estimate_pars is not None:
#             assert set(self.estimate_pars).issubset(set(self.available_parameters)), \
#                 'Check estimate_pars: should be a subset of {}'.format(self.available_parameters)
#         self.initial = np.array([initial_parameters[i] for i in self.available_parameters])
#         if self.estimate_pars is not None:
#             self.estimate_inds = [i for i, p in enumerate(self.available_parameters) if p in self.estimate_pars]
#         else:
#             self.estimate_inds = list(range(len(self.initial)))
#         print(self.estimate_inds)
#         # initialize the julia kernel
#         # package_directory = os.path.dirname(os.path.abspath(__file__))
#         # self.p = subprocess.Popen("julia "+package_directory+"/models/P2D_DE/test.jl")
#         # time.sleep(45)
