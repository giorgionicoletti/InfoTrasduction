import numpy as np
import sys

sys.path.append('../modules/')

import funAdapt as fa

import time as measure_time

Nadapt = 10000
Nrepeat = 32
sigma = 1.5
theta_eta = 3.
delta_a = 0.05

Lambda = 0.9

dt = 1e-3

Nsteps_array = np.array([3, 4, 5, 6, 7])
Nsteps_array = 10**Nsteps_array

for Nsteps in Nsteps_array:
    print('Nsteps = %d' % Nsteps)

    t0 = measure_time.time()

    a_adapt, L_adapt = fa.repeat_adaptive_dynamics(Nrepeat, Nadapt, Nsteps, dt, sigma, theta_eta, Lambda, delta_a, p_acc = 0.00)
    
    np.save(f'../data/adapt_a_Nsteps{Nsteps}_sigma{sigma}_theta{theta_eta}_delta{delta_a}_Lambda{Lambda}_Nadapt{Nadapt}.npy', a_adapt)
    np.save(f'../data/adapt_L_Nsteps{Nsteps}_sigma{sigma}_theta{theta_eta}_delta{delta_a}_Lambda{Lambda}_Nadapt{Nadapt}.npy', L_adapt)

    t1 = measure_time.time()
    print('Time: %f' % (t1 - t0))
    print()