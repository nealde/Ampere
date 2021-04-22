import numpy as np

from typing import List, Dict

from .base_battery import BaseBattery, SimulationResult, ChargeResult


class SingleParticleParabolic(BaseBattery):
    """An implementation of the Single Particle Model, solved with IDA.  This
    version supports CC-discharge, CC-CV charging, and access to the internal
    states of the battery.  It also allows for sequential cycling."""

    def __init__(self, initial_parameters: Dict[str, float] = None, estimate_parameters : List[str] = None, verbose=False): # noqa
        """Constructor for the Single Particle Model class with the parabolic intra-particle concentration approximation.

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

        estimate_parameters: List[str]
            A list of parameters that this instance is allowed to estimate while fitting. Having each instance
                of a model only allow certain values to be estimated helps encourage good code organization.

        Example usage:
        spm = SingleParticleParabolic(initial_parameters={'Rp': 1e-5})
        trial1 = spm.charge(current=1.0)
        trial2 = spm.discharge(current=1.0)
        """
        from .numerical import spm_parabolic
        super().__init__(verbose)
        self.model = spm_parabolic
        self.initial_parameters = {
            "Dn": 3.9e-14,
            "Dp": 1e-14,
            "Rn": 2e-6,
            "Rp": 2e-6,
            "T": 303.15,
            "an": 723600,
            "ap": 885000,
            "ce": 1000,
            "csnmax": 30550,
            "cspmax": 51555,
            "kn": 5.0307e-9,
            "kp": 2.334e-9,
            "ln": 88e-6,
            "lp": 80e-6,
        }
        self.available_parameters = list(self.initial_parameters.keys())

        if isinstance(initial_parameters, dict):
            for key in initial_parameters.keys():
                assert key in self.initial_parameters, "Invalid initial key entered - double check %s" % str(key)
                self.initial_parameters[key] = initial_parameters[key]

        # initial enforces the order parameters are given to the model
        # as of python 3.6, the order of keys in a dictionary can be relied upon due to an implementation detail
        self.initial = np.array([self.initial_parameters[i] for i in self.available_parameters])
        self._validate_estimate_parameters(estimate_parameters)

        # there is no internal structure - the parabolic assumption of internal concentration
        # gradients prevents peering into the particles.
        self.internal_structure = {}

        self.charge_ICs = [
            4.95030611e04,
            3.05605527e02,
            4.93273985e04,
            3.55685791e02,
            3.78436346e00,
            7.86330739e-01,
            1.00000000e00,
        ]
        self.discharge_ICs = [
            2.51584754e04,
            2.73734963e04,
            2.51409091e04,
            2.73785043e04,
            4.26705391e00,
            6.70539113e-02,
            -1.00000000,
        ]


class SingleParticleFDSEI(BaseBattery):
    """An Finite Difference implementation of the Single Particle Model with SEI, solved with IDA.  This
    version supports CC-discharge, CC-CV charging, and access to the internal
    states of the battery.  It also allows for sequential cycling. See .charge, .discharge,
    .cycle, and .piecewise_current for more."""

    def __init__(self, initial_parameters: Dict[str, float]=None, estimate_parameters: List[str] = None, verbose=False):
        """Constructor for the Single Particle Model class with Finite Difference discretization and SEI layer building
        Parameters
        ----------
        initial_parameters: Dict[str, float]
            A dictionary of parameter names and values. Acceptable names for the parameters can be found below:
            | name      | description                                   | default value | Units            |
            |-----------|-----------------------------------------------|---------------|------------------|
            | Dp        | Li+ Diffusivity in positive particle          | 1e-14        | cm^2/s           |
            | Dn        | Li+ Diffusivity in negative particle          | 1e-14        | cm^2/s           |
            | cspmax    | Maximum Li concentration of positive solid    | 51555         | mol/m^3          |
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
            | N1        | Number of FD nodes in positive particle       | 30            |                  |
            | N2        | Number of FD nodes in negative particle       | 30            |                  |

        estimate_parameters: List[str]
            A list of strings representing the parameters that you wish to estimate.
            Defaults to None, which will allow for the estimation of all parameters except temperature.
            For both initial_parameters and estimate_parameters, order does not matter.

        Example usage:
        Example usage:
        spm = SingleParticleFDSEI(initial_parameters={'Rp': 1e-5})
        trial1 = spm.charge(current=1.0)
        trial2 = spm.discharge(current=1.0)
        """
        from .numerical import spm_fd_sei
        super().__init__(verbose)
        self.model = spm_fd_sei
        self.opt = self.opt_wrap
        self.verbose = verbose
        self.initial_fit = 0

        self.initial_parameters = {
            "Dp": 1e-14,
            "Dn": 1e-14,
            "cspmax": 51555,
            "csnmax": 30555,
            "lp": 80e-6,
            "ln": 88e-6,
            "Rp": 2e-6,
            "Rn": 2e-6,
            "T": 303.15,
            "ce": 1000,
            "ap": 885000,
            "an": 723600,
            "M_sei": 0.026,
            "rho_sei": 2100,
            "Kappa_sei": 1.0,
            "kp": 2.334e-11,
            "kn": 8.307e-12,
            "ksei": 1.5e-06,
            "N1": 30,
            "N2": 30,
        }

        self.available_parameters = list(self.initial_parameters.keys())
        if isinstance(initial_parameters, dict):
            for key in initial_parameters.keys():
                assert key in self.initial_parameters, "Invalid initial key entered - double check %s" % str(key)
                self.initial_parameters[key] = initial_parameters[key]

        # initial enforces the order parameters are given to the model
        # as of python 3.6, the order of keys in a dictionary can be relied upon due to an implementation detail
        self.initial = np.array([self.initial_parameters[i] for i in self.available_parameters])
        self._validate_estimate_parameters(estimate_parameters)

        n_positive = self.initial_parameters["N1"]
        n_negative = self.initial_parameters["N2"]

        self.internal_structure = {
            'time': [0],
            'positive_concentration': list(range(1, n_positive + 3)),
            'negative_concentration': list(range(n_positive + 3, n_positive + 3 + n_negative + 2)),
            'SEI thickness': [3 + n_positive + n_negative + 2 + 5]
        }

        # set initial lithium concentrations inside the particles
        self.charge_ICs = [49503.111 for n in range(n_positive + 2)] + [305.55 for n in range(n_negative + 2)]
        self.charge_ICs += [
            3.67873289259766,   # phi_p
            0.182763748093840,  # phi_n
            30,                 # iint
            0,                  # isei
            1e-10,              # delta_sei
            0,                  # Q
            0,                  # cm
            0,                  # cf
            3.0596914450382,    # potential
            30,                 # it
        ]

        # set initial lithium concentrations inside the particles
        self.discharge_ICs = [2.51417672e04 for n in range(n_positive + 2)] + [2.73921225e04 for n in range(n_negative + 2)]
        self.discharge_ICs += [
            4.26700382e00,      # phi_p
            6.70038247e-02,     # phi_n
            2.65295200e-03,     # iint,
            7.34704800e-03,     # isei
            1e-10,              # delta_sei
            3.08271510e01,      # Q
            3.08183958e01,      # cm
            8.75512593e-03,     # cf
            4.20000000e00,      # pot
            30,                 # it
        ]


class SingleParticleFD(BaseBattery):
    """An Finite Difference implementation of the Single Particle Model with SEI, solved with IDA.  This
    version supports CC-discharge, CC-CV charging, and access to the internal
    states of the battery.  It also allows for sequential cycling. See .charge, .discharge,
    .cycle, and .piecewise_current for more."""

    def __init__(self, initial_parameters: Dict[str, float] = None, estimate_parameters: List[str] = None, verbose=False):
        """Constructor for the Single Particle Model class with Finite Difference discretization.
        Parameters
        ----------
        initial_parameters: Dict[str, float]
            A dictionary of parameter names and values. Acceptable names for the parameters can be found below:
            | name      | description                                   | default value | Units            |
            |-----------|-----------------------------------------------|---------------|------------------|
            | Dp        | Li+ Diffusivity in positive particle          | 1e-14        | cm^2/s           |
            | Dn        | Li+ Diffusivity in negative particle          | 1e-14        | cm^2/s           |
            | cspmax    | Maximum Li concentration of positive solid    | 51555         | mol/m^3          |
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
            | N1        | Number of FD nodes in positive particle       | 30            |                  |
            | N2        | Number of FD nodes in negative particle       | 30            |                  |

        estimate_parameters: List[str]
            A list of strings representing the parameters that you wish to estimate.
            Defaults to None, which will allow for the estimation of all parameters except temperature.
            For both initial_parameters and estimate_parameters, order does not matter.

        Example usage:
        Example usage:
        spm = SingleParticleFDSEI(initial_parameters={'Rp': 1e-5})
        trial1 = spm.charge(current=1.0)
        trial2 = spm.discharge(current=1.0)
        """
        from .numerical import spm_fd
        super().__init__(verbose)
        self.model = spm_fd
        self.verbose = verbose
        self.initial_fit = 0

        self.initial_parameters = {
            "Dp": 1e-14,
            "Dn": 1e-14,
            "cspmax": 51555,
            "csnmax": 30555,
            "lp": 80e-6,
            "ln": 88e-6,
            "Rp": 2e-6,
            "Rn": 2e-6,
            "T": 303.15,
            "ce": 1000,
            "ap": 885000,
            "an": 723600,
            "kp": 2.334e-11,
            "kn": 8.307e-12,
            "N1": 30,
            "N2": 30,
        }

        self.available_parameters = list(self.initial_parameters.keys())
        if isinstance(initial_parameters, dict):
            for key in initial_parameters.keys():
                assert key in self.initial_parameters, "Invalid initial key entered - double check %s" % str(key)
                self.initial_parameters[key] = initial_parameters[key]

        # initial enforces the order parameters are given to the model
        # as of python 3.6, the order of keys in a dictionary can be relied upon due to an implementation detail
        self.initial = np.array([self.initial_parameters[i] for i in self.available_parameters])
        self._validate_estimate_parameters(estimate_parameters)

        n_positive = self.initial_parameters["N1"]
        n_negative = self.initial_parameters["N2"]

        self.internal_structure = {
            'time': [0],
            'positive_concentration': list(range(1, n_positive + 3)),
            'negative_concentration': list(range(n_positive + 3, n_positive + 3 + n_negative + 2))
        }

        # set initial lithium concentrations
        self.charge_ICs = [49503.111 for n in range(n_positive + 2)] + [305.55 for n in range(n_negative + 2)]
        self.charge_ICs += [
            3.67873289259766,  # phi_p
            0.182763748093840,  # phi_n
            3.0596914450382,  # pot
            30,  # it
        ]

        self.discharge_ICs = [25817.37 for n in range(n_positive + 2)] + [26885.03 for n in range(n_negative + 2)]
        self.discharge_ICs += [
            4.246347,  # phi_p
            0.046347,  # phi_n
            4.20000000e00,  # pot
            30,  # it
        ]


class PseudoTwoDimFD(BaseBattery):
    """An Finite Difference implementation of the Single Particle Model with SEI, solved with IDA.  This
    version supports CC-discharge, CC-CV charging, and access to the internal
    states of the battery.  It also allows for sequential cycling. See .charge, .discharge,
    .cycle, and .piecewise_current for more."""

    def __init__(self, initial_parameters: Dict[str, float] = None, estimate_parameters: List[str] = None, verbose=False):
        """Constructor for the Single Particle Model base class
        Parameters
        ----------
        initial_parameters: A dictionary of parameter names and values. Acceptable names for the
        parameters can be found below:
        |     name    |                           description                          |  default value  | Units              |
        | ----------- | -----------------------------------------------                | --------------- | ------------------ |
        | D1          | Li+ Diffusivity in electrolyte                                 | 0.15e-8         | cm^2/s             |
        | Dp          | Li+ Diffusivity in positive particle                           | 7.2e-14         | cm^2/s             |
        | Dn          | Li+ Diffusivity in negative particle                           | 7.5e-14         | cm^2/s             |
        | cspmax      | Maximum Li concentration of positive solid                     | 45829           | mol/m^3            |
        | csnmax      | Maximum Li concentration of negative solid                     | 30555           | mol/m^3            |
        | ls          | Separator thickness                                            | 16e-6           | m                  |
        | lp          | Positive electrode thickness                                   | 43e-6           | m                  |
        | ln          | Negative electrode thickness                                   | 46.5e-6         | m                  |
        | es          | Separator volume fraction of pores                             | 0.45            | m^3/m^3            |
        | ep          | Positive electrode volume fraction of pores                    | 0.4             | m^3/m^3            |
        | en          | Negative electrode volume fraction of pores                    | 0.38            | m^3/m^3            |
        | efn         | Negative electrode filler fraction                             | 0.0326          | m^3/m^3            |
        | efp         | Positive electrode filler fraction                             | 0.025           | m^3/m^3            |
        | brugs       | Separator Bruggeman coefficient - pore tortuosity              | 1.5             |                    |
        | brugn       | Negative electrode Bruggeman coefficient - pore tortuosity     | 1.5             |                    |
        | brugp       | Positive electrode coefficient - pore tortuosity               | 1.5             |                    |
        | sigma_n     | Negative electrode electrical conductivity                     | 100             | S/m                |
        | sigma_p     | Positive electrode electrical conductivity                     | 10              | S/m                |
        | t+          | Transference Number - fraction of ionic current carried by Li+ | 0.363           |                    |
        | Rp          | Positive particle radius                                       | 10e-6           | m                  |
        | Rn          | Negative particle radius                                       | 8e-6            | m                  |
        | T           | Ambient Temperature                                            | 303.15          | K                  |
        | ce          | Starting electrolyte Li+ concentration                         | 1200            | mol/m^3            |
        | ap          | Surface area of positive electrode per volume                  | 885000          | m^2/m^3            |
        | an          | Surface area of negative electrode per volume                  | 723600          | m^2/m^3            |
        | kp          | Positive electrode reaction rate                               | 0.10307e-9      | m^2.5/(mol^0.5s)   |
        | kn          | Negative electrode reaction rate                               | 0.1334e-9       | m^2.5/(mol^0.5s)   |
        | N1          | Positive electrode number of FD Nodes                          | 7               |                    |
        | N2          | Separator number of FD Nodes                                   | 3               |                    |
        | N3          | Negative electrode number of FD Nodes                          | 7               |                    |
        | Nr1         | Positive particle number of FD nodes (per particle)            | 3               |                    |
        | Nr2         | Negative particle number of FD nodes (per particle)            | 3               |                    |

        estimate_parameters: A list of strings representing the parameters that you wish to estimate.
        Defaults to None, which will allow for the estimation of all parameters except temperature.

        For both intiial_parameters and estimate_parameters, order does not matter.

        Example usage:
        spm = SingleParticle(initial_parameters=dictionary_of_parameter_label_value_pairs, est_pars=list_of_parameter_labels)

        A list of available keyword agruments (kwargs):



        """
        from .numerical import p2d_fd
        super().__init__(initial_parameters)
        self.model = p2d_fd
        self.verbose = verbose
        self.initial_fit = 0

        self.initial_parameters = {
            "D1": 0.15e-8,
            "Dsn": 7.2e-14,
            "Dsp": 7.5e-14,
            "Rn": 10e-6,
            "Rp": 8e-6,
            "T": 303.15,
            "brugn": 1.5,
            "brugp": 1.5,
            "brugs": 1.5,
            "c0": 1200,
            "csnmax": 30555,
            "cspmax": 45829,
            "efn": 0.0326,
            "efp": 0.025,
            "en": 0.38,
            "ep": 0.4,
            "es": 0.45,
            "kn": 0.10307e-9,
            "kp": 0.1334e-9,
            "ln": 46.5e-6,
            "lp": 43e-6,
            "ls": 16e-6,
            "sigma_n": 100,
            "sigma_p": 10,
            "t1": 0.363,
            "N1": 7,
            "N2": 3,
            "N3": 7,
            "Nr1": 3,
            "Nr2": 3,
        }
        self.available_parameters = list(self.initial_parameters.keys())
        if isinstance(initial_parameters, dict):
            for key in initial_parameters.keys():
                assert key in self.initial_parameters, "Invalid initial key entered - double check %s" % str(key)
                self.initial_parameters[key] = initial_parameters[key]

        # initial enforces the order parameters are given to the model
        # as of python 3.6, the order of keys in a dictionary can be relied upon due to an implementation detail
        self.initial = np.array([self.initial_parameters[i] for i in self.available_parameters])
        self._validate_estimate_parameters(estimate_parameters)

        n_positive_region = int(self.initial[25])
        n_separator_region = int(self.initial[26])
        n_negative_region = int(self.initial[27])
        nr_positive = int(self.initial[28])
        nr_negative = int(self.initial[29])

        # here, the internal structure is more complex.
        # for each "point" along the material regions, there is a simulated particle.
        self.internal_structure = {'time': [0], 'potential': [-2], 'current': [-1]}

        # initialize lithium concentration across the electrolyte
        self.internal_structure.update({
            'electrolyte_lithium_concentration': {
                'positive': [n + 2 for n in range(n_positive_region + 1)],
                'separator': [n + 3 + n_positive_region for n in range(n_separator_region + 1)],
                'negative': [n + 4 + n_positive_region + n_separator_region for n in range(n_negative_region + 1)]
            }
        })
        self.charge_ICs = [1.0 for i in range(n_positive_region + n_separator_region + n_negative_region + 4)]

        # initialize liquid-phase potential across the electrolyte. The slight applied gradient helps the model initialize.
        self.internal_structure.update({
            'liquid_phase_potential': {
                'positive': [n + len(self.charge_ICs) + 1 for n in range(n_positive_region + 1)],
                'separator': [n + len(self.charge_ICs) + 2 + n_positive_region for n in range(n_separator_region + 1)],
                'negative': [n + len(self.charge_ICs) + 3 + n_positive_region + n_separator_region for n in range(n_negative_region + 2)]
            }
        })
        self.charge_ICs += [-0.787e-2 + 0.03e-2 * i for i in range(n_positive_region + n_separator_region + n_negative_region + 3)]

        # set liquid-phase potential to be 0 at the negative electrode current collector - this acts as absolute ground
        self.charge_ICs += [0.0]
        # set initial solid-phase potential for each electrode
        self.internal_structure.update({
            'solid_phase_potential': {
                'positive': [n + len(self.charge_ICs) + 1 for n in range(n_positive_region + 2)],
                'negative': [n + len(self.charge_ICs) + 3 + n_positive_region for n in range(n_negative_region + 2)]
            }
        })
        self.charge_ICs += [2.899 for i in range(n_positive_region + 2)]
        self.charge_ICs += [0.09902 for i in range(n_negative_region + 2)]

        # set initial lithium concentration inside the negative electrode
        # in the equations, the pattern is as follows:
        # * the centers of all of the particles are set
        # * the internals of all of the particles are set, grouped by location along the electrode
        # * the externals of all of the particles are set.
        self.internal_structure.update({
            'solid_lithium_concentration': {f'negative_{n}': [n + len(self.charge_ICs) + 1] for n in range(n_negative_region)}
        })
        count = len(self.charge_ICs) + n_negative_region + 1
        for p in range(nr_negative + 1):
            for n in range(n_negative_region):
                self.internal_structure['solid_lithium_concentration'][f'negative_{n}'] += [count]
                count += 1
        self.charge_ICs += [0.22800075826244027 for i in range(n_negative_region * (nr_negative + 1))]
        self.charge_ICs += [0.21325933957011173 for i in range(n_negative_region)]

        self.internal_structure['solid_lithium_concentration'].update({f'positive_{n}': [n + len(self.charge_ICs) + 1] for n in range(n_negative_region)})
        count = len(self.charge_ICs) + n_positive_region
        for p in range(nr_positive + 1):
            for n in range(n_positive_region):
                self.internal_structure['solid_lithium_concentration'][f'positive_{n}'] += [count]
                count += 1

        self.charge_ICs += [0.9532086891149233 for i in range(n_positive_region * (nr_positive + 1))]
        self.charge_ICs += [0.9780166774057617 for i in range(n_positive_region)]
        self.charge_ICs.append(2.8)  # potential
        self.charge_ICs.append(-17.1)  # initial discharge current

        # initialize the discharge ICs
        self.discharge_ICs = [0.99998 + 2e-6 * i for i in range(n_positive_region + n_separator_region + n_negative_region + 4)]
        self.discharge_ICs += [-0.3447e-2 + 0.01e-2 * i for i in range(n_positive_region + n_separator_region + n_negative_region + 3)]
        self.discharge_ICs += [0.0]
        self.discharge_ICs += [0.422461225901562e1 for i in range(n_positive_region + 2)]
        self.discharge_ICs += [0.822991162960124e-1 for i in range(n_negative_region + 2)]
        self.discharge_ICs += [0.986699999999968 for i in range(n_negative_region * (nr_negative + 1))]
        self.discharge_ICs += [0.977101677061948 for i in range(n_negative_region)]
        self.discharge_ICs += [0.424 for i in range(n_positive_region * (nr_positive + 1))]
        self.discharge_ICs += [0.431 for i in range(n_positive_region)]
        self.discharge_ICs.append(4.2)  # potential
        self.discharge_ICs.append(-17.1)  # initial current
