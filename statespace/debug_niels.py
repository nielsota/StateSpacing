# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:37:47 2022

@author: arnep
"""

import pandas as pd
import numpy as np
from multivariateFilter import multiStateSpacer
from kalman import state_spacer
import matplotlib.pyplot as plt
from matrix_maker import make_matrices_seasonal_trad, make_matrices_seasonal_trigon, add_seasonal_trigon
from matrix_maker import add_exo_deterministic, add_exo_stochastic, add_cycle, make_first, add_seasonal_trad

import scipy.fft
from scipy.fft import fft, fftfreq, fftshift

import statsmodels.api as sm

import time as timer

class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )

        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        return [np.std(self.endog)]*3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self.ssm['obs_cov',0,0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]



np.random.seed(0)


s = 100
p1, p2, p3 = 5, 3, 100
# load data, shape [s, n]
y = np.zeros((s,1))
trend = np.zeros(s)
level = np.zeros(s)

for i in range(1, len(trend)):
    trend[i] = trend[i-1] +  p1 * np.random.randn()

for i in range(1, len(trend)):
    level[i] = level[i-1] + trend[i-1] +  p2 * np.random.randn()

for i in range(len(y)):
    y[i] = level[i] + p3 * np.random.randn()


plt.plot(y)
plt.show()

kf = state_spacer()
#set the function and initialisation of the matrices
kalman_llik = kf.kalman_llik_cython_diffuse


statesv1 = 2
eta_sizev1 = 2

kf.init_matrices(y_dim=y.shape[1] , states = statesv1, eta_size = eta_sizev1)

kf.matr['T'][0,1] = 1   
kf.matr['R'] = np.eye(statesv1, eta_sizev1)
kf.matr['Z'][0, 1] = 0


np.fill_diagonal(kf.matr['Q'], np.nan)
np.fill_diagonal(kf.matr['H'], np.nan)

#initialise parameters and filter
filter_init =  np.zeros(statesv1), (np.eye(statesv1)*1e8)
param_init = np.array((1, 1, 1))
bnds = ((0.0001, 50000),(0.0001, 50000),(0.0001, 50000))

kalman_llik = kf.kalman_llik_cython_diffuse

start = timer.time()
kf.fit(y, optim_fun=kalman_llik,
            filter_init=filter_init, param_init=param_init, bnds=bnds)
    
stop = timer.time()
print('Fitting time: ' +  str(stop - start))


output = kf.smoother(y, filter_init)
fil = np.einsum('ij,kj->ik',  output['output']["alpha"].reshape(100,2), kf.matr['Z']) + kf.matr['c'].T

plt.plot(y)
plt.plot(fil)
plt.show()

start = timer.time()


mod = LocalLinearTrend(y)

stop = timer.time()
print('Fitting time: ' +  str(stop - start))

# Fit it using MLE (recall that we are fitting the three variance parameters)
res = mod.fit(disp=False)
print(res.summary())
