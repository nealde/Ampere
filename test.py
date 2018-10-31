import sys
sys.settrace

from battery import SingleParticleParabolic, SingleParticleFDSEI
import numpy as np

spm1 = SingleParticleFDSEI()
print(spm1.initial_parameters)

labels = ['Rp','lp','kp']
values = [1.5e-6, 78e-6,2.334e-9]
p1 = dict(zip(labels, values))
# print(p1)

spm2 = SingleParticleFDSEI(initial_parameters=p1, estimate_parameters=['lp'])
print(spm1.initial_parameters)
print(spm2.initial_parameters)

# create the experimental data
data = spm1.charge(current=0.5, trim=True)
data2 = spm2.charge(current=0.5, trim=True)
t_exp = data[0]
v_exp = data[1]
curr_exp = data[2]

spm2.fit([t_exp], [v_exp], currents=[-0.5], maxiter=100, tol=1e-7)
