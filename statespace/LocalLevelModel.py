import pathlib

from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import minimize, show_options

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class LocalLevelModel:
    
    data_path: Optional[Path] = None
    params: Dict[str, float] = None
    y, a, att, a_hat, p, ptt, f, v, k, r = None, None, None, None, None, None, None, None, None, None
    
    def __post_init__(self):
        
        if self.data_path:
            if self.data_path.exists():
                self.y = pd.read_csv(self.data_path).values.astype("float")
            else:
                raise Exception('Path does not exist')
    
    def fit(self, y: Optional[np.ndarray] = None):
        
        # If data added during construction, use that
        
        if self.y is not None:
            y = self.y
        
        # If data not added during construction, should be passed to fit
        else: 
            if y is None:
                raise Exception('Need a data source')
            else:
                self.y = y.astype("float")
        
        options = {
            'eps': 1e-6,
            'disp': True,
            'maxiter':500
        }

        bnds = ((0, None), (0, None))

        var_eta_ini = 500
        var_eps_ini = 500
        param_ini = np.array([var_eta_ini, var_eps_ini])

        res = minimize(self._llik, param_ini, args=(self.y, ), method='L-BFGS-B', options=options, bounds=bnds)

        params_dict = {'var_eta': res.x[0], 'var_eps': res.x[1]} 
        print(f'Parameters: {params_dict}')

        self.a, self.att, self.p, self.ptt, self.f, self.v, self.k = self._kalman_filter(self.y, **params_dict)
        self.a_hat, self.r  = self._kalman_smoother(self.a, self.p, self.f, self.v, self.k, **params_dict)
        
    def _get_nan_positions(self, y: np.ndarray) -> List[int]:
        '''
        Find positions of nan elements for the Kalman filter

                Parameters:
                        y (np.Array[float])     : Observed time series which might contain NaN

                Returns:
                        nan_pos_list (List[int]): List of NaN positions
        '''

        if y.ndim == 2:
            y = np.squeeze(y)

        nan_pos_list = np.squeeze(np.argwhere(np.isnan(y))).tolist()

        return nan_pos_list

    def _kalman_step(self, nan_pos_list: List[int], t: int, *args, **params):
        '''
        Computer one step of the kalman filter for the model
        y_t       = alpha_t + epsilon_t  w/ eps ~ N(0, var_eps)
        alpha_t+1 = alpha_t + eta_t      w/ eta ~ N(0, var_eta)

                Parameters:
                        args: variables filled in up until index t-1

                Returns:
                        args: variables filled in up until index t
        '''

        y, a, att, p, ptt, f, v, k = args

        # retrieve the parameters of the LL model
        var_eta = params['var_eta']
        var_eps = params['var_eps']

        # Always true
        a[t] = att[t-1]
        p[t] = ptt[t-1] + var_eta

        # If observation is present, proceed normally
        if t not in nan_pos_list:

            v[t] = y[t] - a[t]
            f[t] = p[t] + var_eps
            k[t] = p[t] / f[t]

            att[t] = a[t] + k[t] * v[t]
            ptt[t] = p[t] * (1 - k[t])

        # If observation is missing, update accordingly
        else:

            # variance -> inf because value unknown
            v[t] = np.nan
            f[t] = np.inf
            k[t] = 0

            # cannot keep k*v because it will equal nan (want 0)
            att[t] = a[t]
            ptt[t] = p[t]

        return a, att, p, ptt, f, v, k
    
    def _kalman_filter(self, y: np.ndarray, diffuse=True, *args, **params):
        '''
        Computer the kalman filter for a local level model given by
        y_t       = alpha_t + epsilon_t  w/ eps ~ N(0, var_eps)
        alpha_t+1 = alpha_t + eta_t      w/ eta ~ N(0, var_eta)

                Parameters:
                        y (np.Array[float])     : Observed time series
                        diffuse (Boolean)       : Whether or not to perform a diffuse initialization
                        params (Dict[str:float]): parameters state matrices

                Returns:
                        a (np.Array[float]): state mean estimate,  a[0] = a_0
                        p (np.Array[float]): state variance estimate, p[0] = p_0
                        f (np.Array[float]): prediction error, f[t] = var(v_t)
                        v (np.Array[float]): prediction variance, v[t] = y[t] - a[t]
                        k (np.Array[float]): Kalman gain, k[t] = p[t] / f[t]
        '''

        # retrieve the parameters of the LL model
        var_eta = params['var_eta']
        var_eps = params['var_eps']

        # get time horizon from data
        T = len(y)

        # get nan positions
        nan_pos_list = self._get_nan_positions(y)

        # initialize filters
        a, att, p, ptt, f, v, k = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        # REPLACE BY INIT FUNCTION?
        if diffuse:
            a[0] = 0
            p[0] = 1e7

        # Compute values at time t=1
        f[0] = p[0] + var_eps
        v[0] = y[0] - a[0]
        k[0] = p[0] / f[0]
        att[0] = a[0] + k[0] * v[0]
        ptt[0] = p[0] * (1 - k[0])

        # Compute values for t = 2,...,T
        for t in range(1, T):

            a, att, p, ptt, f, v, k = self._kalman_step(nan_pos_list, t, y, a, att, p, ptt, f, v, k, **params)

        #print(np.mean(f))
        return a, att, p, ptt, f, v, k
    
    def _kalman_smoother(self, *args, **params):
    
        '''
        Computer the kalman smoother for a local level model given by
        y_t   = alpha_t + epsilon_t  w/ eps ~ N(0, var_eps)
        m_t+1 = alpha_t + eta_t      w/ eta ~ N(0, var_eta)

        Where 
        r_{t-1} = v_t / f_t + L_t * r_t w/ r_n = 0
        a_hat_t = a_t + p_t * r_{t-1} 

                Parameters:
                        y (np.Array[float])     : Observed time series
                        diffuse (Boolean)       : Whether or not to perform a diffuse initialization
                        params (Dict[str:float]): parameters state matrices
                        f (np.Array[float]): prediction variance, f[t] = var(v_t)


                Returns:
                        a (np.Array[float]): state mean estimate,  a[0] = a_0
                        p (np.Array[float]): state variance estimate, p[0] = p_0
                        f (np.Array[float]): prediction error, v[t] = v_t
                        f (np.Array[float]): prediction variance, f[t] = var(v_t)
        '''

        # get output of kalman filter
        a, p, f, v, k = args

        # get L 
        l = 1 - k

        # get time horizon
        T = len(a)

        # r_T=0 not included, but need it in loop... shit
        r = np.zeros(T)

        # initialize r-and-smoother arrays, need r_0 for a_1, which is at index 0 of a
        r = np.zeros(T+1)
        a_hat = np.zeros(T)

        # a_hat[t]: contains a_hat_{t+1}
        # r[t]    : contains r_t

        # recursively compute smoothed state
        for t in range(T-1, -1, -1):

            if np.isnan(v[t]):
                r[t] = r[t+1]

            else:
                # should start at r_{T-1} at index T, and end at r_0 at index 0
                r[t] = v[t] / f[t] + l[t] * r[t+1]

            # a_hat_t = a_t + p_t * r_{t-1}
            # should start at a_hat{T} at index T-1, and end at a_1 at index 0
            a_hat[t] = a[t] + p[t] * r[t]

        return a_hat, r
    
    def _llik(self, params: np.ndarray, *args) -> float:
    
        '''
        Computes Gaussian log-likelihood for local level model
        y_t   = alpha_t + epsilon_t  w/ eps ~ N(0, var_eps)
        m_t+1 = alpha_t + eta_t      w/ eta ~ N(0, var_eta)

        Where 
        r_{t-1} = v_t / f_t + L_t * r_t w/ r_n = 0
        a_hat_t = a_t + p_t * r_{t-1} 

                Parameters:
                        params (np.ndarray)     : Contains parameters
                        args (tuple): contains y -> given my args in minimizer


                Returns:
                        -llik (float): value of log-likelihood for given parameters
        '''

        # retrieve parameters
        params_dict = {'var_eta': params[0], 'var_eps': params[1]} 

        # retrieve data
        y = args[0]

        # get values necessary from Kalman filter
        _, _, _, _, f, v, _ = self._kalman_filter(y, **params_dict)


        # If an observation at time t is not present, should not include in log-likelihood
        v = v[~np.isnan(v)]
        f = f[~np.isinf(f)]

        # Get the number of present observations
        T = len(f)

        # compute log-likelihood
        llik = np.sum( (-1/2) * ( T * np.log(np.pi) + np.log(f[1:]) + np.square(v[1:]) / f[1:]))

        return -llik