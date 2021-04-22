import os
import sys
import numpy as np
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from ampere import SingleParticleFDSEI, SingleParticleFD, SingleParticleParabolic, PseudoTwoDimFD


def test_values_unchanged():
    filename = 'ampere/tests/test_data'
    with h5py.File(filename + ".hdf5", "r") as f:
        for model in [SingleParticleParabolic(), PseudoTwoDimFD(), SingleParticleFD(), SingleParticleFDSEI()]:

            print(model.__class__.__name__)
            data = model.discharge(current=1.0)
            name = f'data{model.__class__.__name__}_discharge'
            new = np.concatenate([[data.time], [data.voltage], [data.current]], axis=0)
            # f.create_dataset(name, data=new, compression="gzip")
            expected = f.get(name)
            np.testing.assert_almost_equal(new, expected)

            data = model.charge(current=1.0)
            name = f'data{model.__class__.__name__}_charge'
            new = np.concatenate([[data.time], [data.voltage], [data.current]], axis=0)
            # f.create_dataset(name, data=new, compression="gzip")
            expected = f.get(name)
            np.testing.assert_almost_equal(new, expected)

            data = model.discharge(current=1.0, internal=True)
            if data.internal is not None:
                name = f'data{model.__class__.__name__}_discharge_internal'
                # f.create_dataset(name, data=data.internal, compression="gzip")
                expected = f.get(name)
                np.testing.assert_almost_equal(data.internal, expected)

            data = model.piecewise_current([500,30,60,100], [1.0,-0.2,0.5,-2])
            name = f'data{model.__class__.__name__}_piecewise'
            new = np.concatenate([[data.time], [data.voltage], [data.current]], axis=0)
            # f.create_dataset(name, data=new, compression="gzip")
            expected = f.get(name)
            np.testing.assert_almost_equal(new, expected)


def test_optimization():
    labels = ['Rp', 'lp', 'kp']
    values = [2.0e-6, 78e-6, 2.334e-9]
    p1 = dict(zip(labels, values))

    spm1 = SingleParticleFDSEI()
    spm2 = SingleParticleFDSEI(initial_parameters=p1, estimate_parameters=['Rp', 'lp', 'kp'])

    # create the experimental data
    a = [
        spm1.discharge(current=0.5, trim=True),
        spm1.charge(current=1.0, trim=True)
    ]

    t_exp = [x.time for x in a]
    v_exp = [x.voltage for x in a]
    res = spm2.fit(t_exp, v_exp, currents=[0.5, -1.0], maxiter=1000)

    np.testing.assert_almost_equal(res.x, [7.84309206e-05, 2.01704867e-06, 2.36852421e-09])


def test_generate_data():
    spm = SingleParticleFDSEI()
    data = spm.generate_data(100, currents=[0.5, 1.0], parameters=['Rp', 'Rn'], bounds=[[1e-12, 1e-3], [1e-12, 1e-3]], sample_type='random')
    assert data.inputs.shape == (98, 2)
    assert data.voltages.shape == (98, 500, 2)
    assert data.failed.shape == (2, 2)


test_values_unchanged()
test_optimization()
test_generate_data()