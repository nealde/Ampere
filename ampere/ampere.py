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

    def charge(self, t=None, current=0.5, from_current_state=False, p=None, trim=False, internal=False):
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
            solve = self.model([*p, current, 1], t, initial=self.current_state, internal=internal)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if internal:
                if trim:
                    return [solve[0][:,solve[0][2]<-0.01], solve[-1]]
                else:
                    return [solve[0], solve[-1]]
            else:
                if trim:
                    return solve[0][:,solve[2]<-0.01]
                else:
                    return solve[0]
        else:
            solve = self.model([*p, current, 1], t, initial=self.charge_ICs, internal=internal)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if internal:
                if trim:
                    return [solve[0][:,solve[0][2]<-0.01], solve[-1]]
                else:
                    return [solve[0], solve[-1]]
            else:
                if trim:
                    return [solve[0][:,solve[2]<-0.01]]
                else:
                    return [solve[0]]

    def discharge(self, t=None, current=0.5, from_current_state=False, p=None, trim=False, internal=False):
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
            solve = self.model([*p, current*-1, 1], t, initial=[*self.current_state[:-1], current*-1], internal=internal)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if internal:
                if trim:
                    return [solve[0][:,:np.where(solve[0][2]==0)[0][0]+1], solve[-1]]
                else:
                    return [solve[0], solve[-1]]
            else:
                if trim:
                    return [solve[0][:,:np.where(solve[0][2]==0)[0][0]+1]]
                else:
                    return [solve[0]]
        else:
            # print([*p, current*-1, 1], t[-1], self.discharge_ICs, internal)
            solve = self.model([*p, current*-1, 1], t, initial=self.discharge_ICs, internal=internal)
            self.current_state = solve[1][1:]
            self.hist.append(solve[0])
            if internal:
                if trim:
                    return [solve[0][:,:np.where(solve[0][2]==0)[0][0]+1], solve[-1]]
                else:
                    return [solve[0], solve[-1]]
            else:
                if trim:
                    return [solve[0][:,:np.where(solve[0][2]==0)[0][0]+1]]
                    # return solve[0][:,solve[2]<-0.01]
                else:
                    return [solve[0]]

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

    def piecewise_current(self, times, currents, n_steps=50, from_current_state=False, p=None, internal=False):
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
        curr = []
        count = 0
        if not from_current_state:
            self.current_state = self.discharge_ICs
        for t, c in zip(times, currents):
            tt = np.linspace(0, t, n_steps)
            if len(solve)>1:
                if solve[-1][-1,-2] <= 2.5 and c > 0:
                    print(solve[-1][-1,-2])
                    break
            # else:
            curr.append(c)
            # if solve[-1]
            # try:
                # print(solve[-1], t, c)
            # except:
            #     pass
            if c > 0:
                try:
                    out = self.discharge(tt, current=c, from_current_state=True, p=p, internal=internal)
                    # print(len(out))
                except IndexError:
                    pass
                    # out = [tt, np.ones(len(tt))*2.5, np.ones(len(tt))*c]
            else:
                try:
                    if internal:
                        out = self.charge(tt, current=c*-1, from_current_state=True, p=p, internal=internal)
                    else:  # need to nest one layer deeper for downstream code
                        out = [self.charge(tt, current=c*-1, from_current_state=True, p=p, internal=internal)]
                except IndexError:
                    pass
                    # out = np.array([tt, np.ones(len(tt))*4.2, np.ones(len(tt))*c])
            # print(out)
            # add times together
            # if count > 0:
            #     print(solve[-1])
            #     out[0] += solve[-1][0,-1]
            if internal:
                solve.append(out[-1])
            else:
                # add times together
                if count > 0:
                    out[0][0] += solve[-1][0, -1]
                solve.append(out[0])
            count += 1

        # print(solve)
        if internal:
            # print([s.shape for s in solve])
            # print(solve[-1].shape)
            solve = np.concatenate(solve, axis=0)
        else:
            solve = np.concatenate(solve, axis=1)
        self.hist.append(solve)
        return solve, curr




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
        print(self.count)
        self.count+=1
        try:
            if currents_type == 'constant':
                for t, v, c in zip(self.t_exp, self.v_exp, self.currents):
                    if c > 0:
                        self.current_state = self.discharge_ICs
                        solve = self.discharge(t, current=c, from_current_state=True, p=x)
                        print(len(solve[0]))
                        error += rmse(solve[0][1], v)
                    else:
                        self.current_state = self.charge_ICs
                        solve = self.charge(t, current=-c, from_current_state=True, p=x)
                        error += rmse(solve[0][1], v)
            else:
                solve = self.piecewise_current(self.t_exp, self.currents, p=x)
                error += rmse(solve[0][1], self.v_exp)
            if verbose:
                print(error, x0)
        except:
            error = 100
            print('failed')
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
        self.count = 0

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

    def refit(self, method="Nelder-Mead", maxiter=100):
        assert self.initial_fit == 1, \
            'Please call fit before calling refit'
        self.initial[self.estimate_inds] = self.fitted_parameters
        self.fit(self.t_exp, self.v_exp, self.currents, method=method, re=1, maxiter=maxiter)
        return

    def generate_data(self, filename, n, currents, loglist='auto', pars=None, bounds=None,
                      type='sobol', distribution='uniform', sample_time=None,
                      time_samples=100, summary=True, internal=True, just_sample=False, verbose=False):
        """
        This function uses the existing Julia kernel to generate a set of data, similar to how the
        optimization function works. Since this julia kernel already exists, the calculations are note made in parallel.
        In the future, a separate file may exist which generates the data in parallel. It is recommended to call fit() first,
        in order to establish t_exp. Otherwise, time can be manually input.
        Parameters
        ----------
        filename: The filename for the h5 file the data is saved in (using h5py)

        n : the number of samples to make. For sobol, this is the total number, and for grid, this is the number of
                samples per grid dimension. Be careful using grid, as this can get out of hand very quickly (n^dim)

        loglist: A list of parameters which should be log spaced. Log spacing is advised for any variable changing
                    by more than 100x, i.e. max/min > 100.  If 'auto' is given, this will be detected automatically.
                    defaults to 'auto'.

        pars: A list of parameters if desired modified parameters are different than the initialized set.

        bounds: Hard upper and lower limits for generating the data. If this is not given, +/- 30% is used. If a float is given,
                    +/- that amount is used. Custom bounds are recommended for good results, i.e., fewer failed samples.

        type: defaults to 'sobol', options include 'grid' and 'saltelli'.

        distribution: defaults to 'uniform', describes the distribution of the sampling. Options include 'normal', 'uniform', and 'bi-modal'
                            (to be implemented)
        sample_time: The times to be interpolated for sampling. Filled values will be 2.5V, in keeping with the final
                        discharge voltage.  Defaults to linear spacing at 20% longer than t_exp
        time_samples: number of points in the linear timespacing, defaults to 100.
        """
        assert isinstance(filename, str), \
            'Filename must be type string'
        assert isinstance(n, int), \
            'n must be an integer'
        assert isinstance(loglist, (str, list, bool)), \
            'loglist must be a list, False, or "auto"'
        assert type in ['grid', 'sobol', 'saltelli', 'random'], \
            'Available arguments for type are {}'.format(['grid', 'sobol', 'saltelli', 'random'])
        try:
            assert isinstance(self.t_exp, (np.ndarray, list)), \
                't_exp must be a list of numpy arrays matching the length of currents given in initial conditions'
            assert len(self.t_exp) == len(self.currents), \
                'Number of currents does not match experimental data - received {} but expected {}'.format(len(self.t_exp), len(self.currents))
        except AttributeError:
            pass

        import time
        self.currents = currents
        self.num_currents = len(self.currents)
        if self.num_currents > 1:
            print('multiple currents detected')
            self.currents = sorted(self.currents, reverse=True)
        self.generate_pars = None
        self.n = n*self.num_currents
        # step 1 - handle the arguments
        if pars is None:
            self.generate_pars = self.estimate_pars
            self.generate_inds = self.estimate_inds
        else:
            self.generate_pars = pars
            self.generate_inds = [i for i, x in enumerate(self.available_parameters) if x in self.generate_pars]
            # self.bounds_inds = [i for i, x in enumerate(self.available_parmeters)]
        assert self.generate_pars is not None
        print(self.generate_inds)

        # set up bounds (used for loglist)
        if bounds is None:
            if self.bounds is None:
                self.bounds = [(x/1.2, x*1.2) for x in self.initial[self.generate_inds]]
        else:
            d1 = dict(zip(pars,range(len(pars))))
            bounds_inds = [d1[i] for i in [self.available_parameters[j] for j in self.generate_inds]]
            self.bounds = [bounds[i] for i in bounds_inds]

        # set up log-spacing
        if loglist is False:
            self.loglist = np.zeros(len(self.generate_pars))
        if isinstance(loglist, list):
            assert len(loglist) == len(self.generate_pars), \
                'expected loglist to be same length as generate_pars, \
            but got {} and {}'.format(len(loglist), len(self.generate_pars))
            self.loglist = loglist
        if loglist == 'auto':
            self.loglist = [1 if x[0]/x[1] >= 100 else 0 for x in self.bounds]

        # create the array using the spacing method of choice
        self.raw_sample = None
        if type == 'sobol':
            from sobol_seq import i4_sobol_generate
            self.raw_sample = i4_sobol_generate(len(self.generate_pars),
                                                self.n)
        elif type == 'saltelli':
            from SALib.sample import saltelli
            problem = {'names': self.estimate_pars,
                       'bounds': [[0, 1] for x in self.estimate_pars],
                       'num_vars': len(self.estimate_pars)}
            self.raw_sample = saltelli.sample(problem, self.n, True)

        elif type == 'grid':
            from sklearn.utils.extmath import cartesian
            temp = np.linspace(0, 1, self.n)
            self.raw_sample = cartesian([temp for i in range(len(self.generate_pars))])

        elif type == 'random':
            self.raw_sample = np.random.random((n,len(self.generate_pars)))

        assert self.raw_sample is not None, \
            'something went wrong - check that type is correct'
        print('expected shape is {}'.format(self.raw_sample.shape))

        # map the raw array to bounds, adhering to log scaling rules
        self.scaled_sample = self.log_scale_matrix(self.raw_sample)
        if just_sample:
            return [None, self.scaled_sample]
        else:

            outs = []
            ins = []
            self.failed=[]
            count = 0
            for i in self.currents:
                for parameter_set in self.scaled_sample:
                    simulate_pars = np.copy(self.initial)
                    simulate_pars[self.generate_inds] = self.log_descale_for_model(parameter_set)
                    # print(simulate_pars)
                    try:
                        if i > 0:
                            outs.append(self.discharge(current=i, p=simulate_pars, internal=internal)[1])
                        else:
                            outs.append(self.charge(current=-1*i, p=simulate_pars, internal=internal)[1])
                        ins.append(self.log_descale_for_model(parameter_set))
                    except IndexError:
                        self.failed.append(simulate_pars)


                # outs.append(self.)
            #     outs.append([])
            # # outs = np.zeros((self.generate_time.shape[0], time_samples))
            # # ins = np.zeros(self.generate_time.shape[0], len(self.generate_inds))
            # # print(self.scaled_sample)
            # # print(self.generate_time)
            # count = 0
            # st = time.time()
            # self.currents = sorted(self.currents)
            # self.failed = []
            # for parameter_set in self.scaled_sample:
            #     try:
            #         succ = 0
            #         # reverse the list because high currents tend to fail more frequently,
            #         # and we want it to fail first if it's going to fail.
            #         for i, curr in enumerate(self.currents[::-1]):
            #
            #             current_pars = np.copy(self.initial)
            #             current_pars[self.generate_inds] = self.log_descale_for_model(parameter_set)
            #             current_pars[self.curr_index] = curr
            #             # print(current_pars)
            #             outs[i].append(self.model(current_pars, self.generate_time[i]))
            #             succ += 1
            #         ins.append(parameter_set)
            #     except (ValueError, IndexError):
            #         # if self.verbose:
            #             # print('failed - ', current_pars, count)
            #         self.failed.append([current_pars, count])
            #         if succ != 0:
            #             for i in range(succ):
            #                 outs[i].pop()
            #     count += 1
            #     if count == 20:
            #         print('{} solutions completed of {} in {} seconds - {} total hours predicted'.format(count, self.scaled_sample.shape[0], time.time()-st, (time.time()-st)/3600/(count/self.scaled_sample.shape[0])))
            #     if count % (self.scaled_sample.shape[0]//20) == 0 and self.verbose:
            #         print('{} solutions completed of {} in {} seconds - {} total hours predicted'.format(count, self.scaled_sample.shape[0], time.time()-st, (time.time()-st)/3600/(count/self.scaled_sample.shape[0])))
            #
            # # save the values to an h5 file
            # outs = [np.array(out) for out in outs]
            # ins = np.array(ins)
            # self.raw_outs = outs
            #
            # outs2 = []
            # for i in range(len(outs[0])):
            #     outs2.append([x[i] for x in outs])
            # outs2 = np.array(outs2)
            # outs = outs2.reshape(outs2.shape[0], -1)
            #
            # # break into test and train splits
            # inds = np.arange(outs.shape[0])
            # np.random.shuffle(inds)
            # train = inds[:outs.shape[0]*3//4]
            # test = inds[outs.shape[0]*3//4:]
            # #
            # x = outs[train]
            # xt = outs[test]
            # y = ins[train]
            # yt = ins[test]
            # self.outs = outs
            # self.ins = ins
            # self.create_database(x, y, xt, yt, filename)
            # if summary:
            #     print("""A total of {} parameter combinations were evaluated. Of
            #     these, {} failed, representing {} percent.  These values can be
            #     found at object.failed""".format(len(self.scaled_sample), len(self.failed), len(self.failed)/len(self.scaled_sample)*100))
            return [outs, ins]

    def create_database(self, x, y, xt, yt, filename):
        """This function creates a dataset using pre-split data and saves it
        into a compressed h5 file with name filename. x, y, xt, yt are assumed
        to be numpy arrays."""
        import h5py
        with h5py.File(filename+".hdf5", "w") as f:
            f.create_dataset('x', data=x, compression='gzip')
            f.create_dataset('xt', data=xt, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('yt', data=yt, compression='gzip')
        return

    def abscale(self, matrix, a=-1, b=1):
        out = 0
        if matrix.shape == (matrix.shape[0],):
            matrix = matrix.reshape(-1, 1)
            out = 1
        new = np.zeros(matrix.shape)
        for i in range(matrix.shape[1]):
            new[:, i] = (b-a)*(matrix[:, i]-matrix[:, i].min())/(matrix[:, i].max()-matrix[:, i].min())+a
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

    def demonstrate_convergence(self, function):
        '''This function seeks to demonstrate grid independence for the given
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

        '''
        self.initial_discretization = self.initial[25:]
        self.nodes = []
        self.errors = []
        print(self.initial_discretization)
        for i in range(1,10):
            self.nodes.append(self.initial_discretization*i)


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
            self.discharge_ICs.append(2.51417672e+04)
        for i in range(N1+2, N1+N2+4):
            self.discharge_ICs.append(2.73921225e+04)
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


class SingleParticleFD(BaseBattery):
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
        from .numerical import SPM_fd
        from .fitting import rmse
        super().__init__(initial_parameters, **kwargs)
        self.model = SPM_fd
        self.opt = self.opt_wrap
        self.verbose = verbose
        self.initial_fit = 0
        TC = 30
        # self.inplace = np.zeros((10000,8))
        self.available_parameters = ['Dp','Dn','cspmax','csnmax','lp','ln','Rp','Rn','T','ce','ap','an','kp','kn','N1','N2']
        self.default_values = [1e-14, 1e-14, 51555.0, 30555.0, 8e-05, 8.8e-05, 2e-06, 2e-06, 303.15, 1000.0, 885000.0, 723600.0, 2.334e-11, 8.307e-12, 30, 30]
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
        N1 = int(self.initial[14])
        N2 = int(self.initial[15])
        for i in range(N1+2):
            self.charge_ICs.append(49503.111)
        for i in range(N1+2, N1+N2+4):
            self.charge_ICs.append(305.55)
        self.charge_ICs.append(3.67873289259766)    #phi_p
        self.charge_ICs.append(.182763748093840)    #phi_n
        self.charge_ICs.append(3.0596914450382)   #pot
        self.charge_ICs.append(TC*1)   	  	  #it
        # self.charge_ICs = [4.95030611e+04, 3.05605527e+02, 4.93273985e+04, 3.55685791e+02, 3.78436346e+00, 7.86330739e-01, 1.00000000e+00]
        self.discharge_ICs=[]
        for i in range(N1+2):
            self.discharge_ICs.append(25817.37)
        for i in range(N1+2, N1+N2+4):
            self.discharge_ICs.append(26885.03)
        self.discharge_ICs.append(4.246347)
        self.discharge_ICs.append(0.046347)
        self.discharge_ICs.append(4.20000000e+00)
        self.discharge_ICs.append(1.00000000e-02)
        # self.discharge_ICs = [2.51584754e+04, 2.73734963e+04, 2.51409091e+04, 2.73785043e+04, 4.26705391e+00, 6.70539113e-02, -1.00000000]
        self.hist = []

class PseudoTwoDimFD(BaseBattery):
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
        from .numerical import P2D_fd
        from .fitting import rmse
        super().__init__(initial_parameters, **kwargs)
        self.model = P2D_fd
        self.opt = self.opt_wrap
        self.verbose = verbose
        self.initial_fit = 0
        self.discretized = True
        # TC = 30
        # self.inplace = np.zeros((10000,8))
        self.available_parameters = ['D1','Dsn','Dsp','Rpn','Rpp','Temp','brugn','brugp','brugs','c0','ctn','ctp','efn','efp','en','ep','es','kn','kp','ln','lp','ls','sigman','sigmap','t1','N1','N2','N3','Nr1','Nr2']
        self.default_values = [.15e-8, .72e-13, .75e-13, .10e-4, .8e-5, 298.15, 1.5, 1.5, 1.5,  1200, 30555, 45829., .3260e-1,.025, .38, .4, .45, .10307e-9, .1334e-9, .465e-4, .43e-4, .16e-4, 100, 10, .363, 7, 3, 7, 3, 3]
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


        N1 = int(self.initial[25])
        N2 = int(self.initial[26])
        N3 = int(self.initial[27])
        Nr1 = int(self.initial[28])
        Nr2 = int(self.initial[29])


        self.charge_ICs = []
        for i in range(N1+N2+N3+4):
            self.charge_ICs.append(1.0)
        for i in range(N1+N2+N3+3):
            self.charge_ICs.append(-.787E-2+.03E-2*i)
        self.charge_ICs.append(0.0)
        for i in range(N1+2):
            self.charge_ICs.append(2.899)
        for i in range(N3+2):
            self.charge_ICs.append(0.09902)
        for i in range(N3*(Nr2+1)):
            self.charge_ICs.append(0.22800075826244027)
        for i in range(N3):
            self.charge_ICs.append(0.21325933957011173)
        for i in range(N1*(Nr1+1)):
            self.charge_ICs.append(0.9532086891149233)
        for i in range(N1):
            self.charge_ICs.append(0.9780166774057617)
        self.charge_ICs.append(2.8)
        self.charge_ICs.append(-17.1)

        # initialize the discharge ICs
        self.discharge_ICs = []
        for i in range(N1+N2+N3+4):
            self.discharge_ICs.append(0.99998+2e-6*i)
        for i in range(N1+N2+N3+3):
            self.discharge_ICs.append(-.3447E-2+.01E-2*i)
        self.discharge_ICs.append(0.0)
        for i in range(N1+2):
            self.discharge_ICs.append(0.422461225901562E1)
        for i in range(N3+2):
            self.discharge_ICs.append(0.822991162960124E-1)
        for i in range(N3*(Nr2+1)):
            self.discharge_ICs.append(0.986699999999968)
        for i in range(N3):
            self.discharge_ICs.append(0.977101677061948)
        for i in range(N1*(Nr1+1)):
            self.discharge_ICs.append(0.424)
        for i in range(N1):
            self.discharge_ICs.append(0.431)
        self.discharge_ICs.append(4.2)
        self.discharge_ICs.append(-17.1)

        self.hist = []
