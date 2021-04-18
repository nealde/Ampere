import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from ampere import SingleParticleFDSEI, SingleParticleFD, SingleParticleParabolic, PseudoTwoDimFD

def test_values_unchanged():
    for model in [SingleParticleFDSEI(), SingleParticleFD(), SingleParticleParabolic(), PseudoTwoDimFD()]:
        print(model.__class__.__name__)
        data = model.discharge(current=1.0)
        # np.savetxt(f'ampere/tests/data{model.__class__.__name__}_discharge.csv', data[0])
        expected = np.genfromtxt(f'ampere/tests/data{model.__class__.__name__}_discharge.csv')
        np.testing.assert_almost_equal(data[0], expected)

        data = model.charge(current=1.0)
        # np.savetxt(f'ampere/tests/data{model.__class__.__name__}_charge.csv', data[0])
        expected = np.genfromtxt(f'ampere/tests/data{model.__class__.__name__}_charge.csv')
        np.testing.assert_almost_equal(data[0], expected)


test_values_unchanged()