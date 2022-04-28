import torch
import pathlib

from typing import List, Optional, Dict, Tuple
from torch.autograd import Variable
from scipy.optimize import minimize
from collections import deque
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import _initiate_variables, _map_vector_to_matrices, _get_nan_positions, _remove_nan_tensor, _remove_inf_tensor, _get_bounds, read_target_from_path
from StateSpacingKalman import KalmanV1
from StateSpacingProtocols import KalmanProtocol, MathProtocol

@dataclass
class LinearGaussianModel_v1_numpy:
    
    """
    class implementing a linear Gaussian model in StateSpace form
    
        y_t         = Z_t * alpha_t +       epsilon_t        epsilon_t ~ N(0, H_t)
        alpha_{t+1} = T_t * alpha_t + R_t * eta_t            eta_t ~ N(0, Q_t)
        
    """
    
    # Declare observation vector
    y: np.ndarray
    
    # Declare State Matrices 
    T: np.ndarray
    Z: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    H: np.ndarray
        
    # Declare whether init is diffuse
    diffuse: bool
    
    # Declare map for mapping param vector of optimizer to state matrices; e.g. 0: {"matrix" : "Q", "index": (0, 0, 0), "constant": True, "bounds": (None, None)}
    param_map: Dict[int, Dict]
    
    # Declare map for mapping each row of filters to its role {'0': level, '1': trend, '2': season}
    filter_map: Dict[int, Dict]
        
    kalman: KalmanProtocol = KalmanV1()
        
    def __post_init__(self):
        
        # Get p, s, n: dimension state, dimension observation and length of y (time)
        self.p = self.T.shape[0]
        self.s = self.y.shape[0]
        self.n = self.y.shape[2]
        self.shapes = {'p': self.p, 's': self.s, 'n': self.n}
        
        # Check that shapes match
        assert self.__check_shapes(self.T, self.Z, self.R, self.Q, self.H)
        
        # Add time dimension to state matrices
        self.__3Dfy_state_matrices()
        
        # Get filteres (line 1), filter variances (line 2), error and error variance (line 3) signals and Kalman gains
        self.a, self.att, self.a_hat, self.P, self.Ptt, self.P_hat, self.v, self.F, self.K, self.M = _initiate_variables(self.p, self.s, self.n)
    
    def __get_state_matrices(self):
        return self.T, self.Z, self.R, self.Q, self.H
    
    def __set_state_matrices(self, *args):
        T, Z, R, Q, H = args
        self.__check_shapes(T, Z, R, Q, H)
        self.T, self.Z, self.R, self.Q, self.H = T, Z, R, Q, H
    
    def __3Dfy_state_matrices(self):
        """
        takes 2D state matrices and turns into 3D state matrices
        """
        
        shapes = self.__get_shapes()
        n = shapes['n']
        names = ['T', 'Z', 'R', 'Q', 'H']
        
        TwoD_matrices = [*self.__get_state_matrices()]
        
        # use i..range to edit elements in list
        for i in range(len(TwoD_matrices)):
            if TwoD_matrices[i].ndim == 2:
                TwoD_matrices[i] = np.repeat(TwoD_matrices[i][:, :, None], n, axis=2)
            else:
                if TwoD_matrices[i].shape[2] != n:
                    raise ValueError(f' #depth of {names[i]} is {TwoD_matrices[i].shape[2]} and not {n})')
        
        # update into 3D matrices
        self.__set_state_matrices(*TwoD_matrices)      
    
    def __get_filters(self):
        return self.a, self.att, self.a_hat, self.P, self.Ptt, self.P_hat, self.v, self.F, self.K, self.M
    
    def __get_shapes(self):
        return self.shapes
    
    def __get_filter_map(self):
        return self.filter_map
    
    def __check_shapes(self, T, Z, R, Q, H):
        """ checks if shapes of state matrices are compatible 

        Args:
            state matrices T, Z, R, Q, H
    
        Returns:
            True: if all shape compatibility tests are passed
        """
        
        # check that T is square
        if not T.shape[0] == T.shape[1]:
            raise ValueError(f'T not a square square matrix ({T.shape[0]}x{T.shape[1]})')
        
        # check that Q is square
        if not Q.shape[0] == Q.shape[1]:
            raise ValueError(f'Q not a square square matrix ({Q.shape[0]}x{Q.shape[1]})')
        
        # check that H is square
        if not H.shape[0] == H.shape[1]:
            raise ValueError(f'H not a square square matrix ({H.shape[0]}x{H.shape[1]})')
        
        # check that columns of R is rows of Q
        if not R.shape[1] == Q.shape[0]:
            raise ValueError(f' #columns of R not #rows of Q ({R.shape[1]} and {Q.shape[0]})')
            
        # check that rows of T is rows of R
        if not T.shape[0] == R.shape[0]:
            raise ValueError(f' #rows of T not #rows of R ({T.shape[0]} and {R.shape[0]})')
        
        return True
    
    def fit(self):
        """ fits model to data

        Args:
            None: acts on instance of the class
    
        Returns:
            None: but updates all instance variables after training
        """
        
        # Get State Matrices 
        T, Z, R, Q, H = self.__get_state_matrices()

        # Get whether init is diffuse
        diffuse = self.diffuse

        # Get map for mapping param vector of optimizer to state matrices
        param_map= self.param_map

        # Get observation vector
        y = self.y
        
        # Get shapes
        shapes = self.shapes
        
        # set options for minimization
        options = {
            'eps': 1e-8,
            'disp': True,
            'maxiter': 2000
        }
        
        # Get bounds for optimization
        bounds = _get_bounds(param_map)
        
        params_ini = np.ones((len(param_map), 1))
        
        # maximize log-likelihook
        res = minimize(self.kalman.log_likelihood, params_ini, args=(T, Z, R, Q, H, y, param_map, diffuse, shapes), method='L-BFGS-B', options=options, bounds=bounds)
        
        # extract params
        params = res.x
        
        # Update instance state matrices
        T, Z, R, Q, H = _map_vector_to_matrices(params, param_map, T, Z, R, Q, H)
        self.T, self.Z, self.R, self.Q, self.H = T, Z, R, Q, H
        
        # Get filtered and incasted signals and signal variances, 
        a, att, P, Ptt, F, v, K, M  = self.kalman.kalman_filter(T, Z, R, Q, H, y, diffuse, shapes)
        self.a, self.att, self.P, self.Ptt, self.v, self.F, self.K, self.M = a, att, P, Ptt, F, v, K, M 
        
        # Get smoothed signal
        a_hat, P_hat, r, N, L = self.kalman.kalman_smoother(T, Z, R, Q, H, a, P, v, F, K, shapes)
        self.a_hat, self.P_hat, self.r, self.N, self.L = a_hat, P_hat, r, N, L
        
    
    def plot_states(self, signal_components=None, state_only=False):
        """ Plots y + states that should approx y, and plots all states 

        Args:
            signal_components (str): which components of Za go into signal
            state_only (str): Whether or not to only plot a (and not Za)
    
        Returns:
             A plot of filter mean and variance
        """
        
        # get filters for plotting
        a, att, a_hat, P, Ptt, P_hat, F, v, K, M = self.__get_filters()
        
        # get system matrices
        T, Z, R, Q, H = self.__get_state_matrices()
        
        # get shapes
        filter_map = self.__get_filter_map()
        filter_names = list(filter_map.values())
        filter_keys = list(filter_map.keys())
        
        # if no signal passed, assume level at index 0
        if not signal_components:
            print('assuming component at index 0 is main component of y')
        signal_components = signal_components if signal_components else [filter_map[0]]
        
        # check if components in signal are in fact part of filters
        for component in signal_components:
            if not component in filter_names:
                raise ValueError(f'{component} not part of signal')
                
        # get indices of components passed to signal argument
        indxs = []
        for component in signal_components:
            index = filter_names.index(component)
            indxs.append(filter_keys[index])
        
        # number of plots is 1 (includes obeservations) + 1 for each filter in filter_map
        nfilters = len(filter_map.keys())
        num_plots = nfilters + 1
        
        # get z, y and a_hat in correct shape
        Z = np.squeeze(Z)
        y = np.squeeze(self.y)
        a_hat = np.squeeze(a_hat)
        
        # create plot 
        fig, axs = plt.subplots(num_plots, sharex=True)
        fig.set_size_inches(10, 10)
        
        axs[0].set_title('Observation + Level (at index 0)')
        axs[0].plot(range(len(y)), y)
        axs[0].grid()
        
        # machanics of slicing depends on number of states
        if nfilters == 1:
            if not state_only:
                axs[0].plot(Z * a_hat)
            else:
                axs[0].plot(a_hat)
        else:
            if not state_only:
                signal = np.sum([Z[i, :] * a_hat[i, :] for i in indxs], axis=0)
                axs[0].plot(signal)
            else:
                signal = np.sum([a_hat[i, :] for i in indxs], axis=0)
                axs[0].plot(signal)
        
        # for each plot, plot the row of a_hat given in filter_map with the given title
        for idx, i in enumerate(filter_map.keys()):
            
            axs[idx + 1].set_title(filter_map[i])
            axs[idx + 1].grid()
            
            if nfilters == 1:
                axs[idx + 1].plot(a_hat)
            else:
                axs[idx + 1].plot(a_hat[i, :])
        
        
    def plot_state(self, state_name: str, filter_type='smoothed'):
        """ Plots a particular state with is variance

        Args:
            state_name (str): which state you want to plot
            filter_type (str): what type of filter you want to plot
    
        Returns:
             A plot of filter mean and variance
        """
        
        # check if entered type is allowed
        allowed_types = ['smoothed', 'incasted', 'filtered']
        if filter_type not in allowed_types:
            raise ValueError(f'filter type is {filter_type} but must be in {allowed_types}')
        
        # get filters for plotting
        a, att, a_hat, P, Ptt, P_hat, F, v, K, M = self.__get_filters()
        
        # get system matrices
        T, Z, R, Q, H = self.__get_state_matrices()
        
        # check what type of filter to use -> set filter_ and filter_var accordingly
        if filter_type == 'smoothed':
            filter_ = a_hat
            filter_var = P_hat
            filter_label = r'$E(a_t|Y_n)$'
            filter_var_label = r'$var(a_t|Y_n)$'
        elif filter_type == 'incasted':
            filter_ = att
            filter_var = Ptt
            filter_label = r'$E(a_t|Y_t)$'
            filter_var_label = r'$var(a_t|Y_t)$'
        else:
            filter_ = a
            filter_var = P
            filter_label = r'$E(a_t|Y_{t-1})$'
            filter_var_label = r'$var(a_t|Y_{t-1})$'
        
        # get shapes
        filter_map = self.__get_filter_map()
        filter_names = list(filter_map.values())
        filter_keys = list(filter_map.keys())
        
        # for example, if state_name is 'exogenous', check if 'exogenous' in filter_names, ow can't plot it
        if state_name not in filter_names:
            raise ValueError(f'{state_name} not part of filter_map')
        
        # get the index of the state name in the state vector
        index = filter_keys[filter_names.index(state_name)]
        
        # create plot 
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(10, 10)
        
        # plot the filter and filter variance
        axs[0].plot(np.squeeze(filter_[index, :]), label=filter_label)
        axs[1].plot(np.squeeze(filter_var[index, index, :]), label=filter_var_label)
    
        # styling
        for ax in axs:
            ax.grid()
            ax.legend()
@dataclass
class LLM_v1_numpy(LinearGaussianModel_v1_numpy):
    
    def __init__(self, y, dtype = np.float64):

         # Declare State Matrices Local Level Model -> use [[]] for extra dimension
        T = np.array([[1]]).astype(dtype)
        Z = np.array([[1]]).astype(dtype)
        R = np.array([[1]]).astype(dtype)
        Q = np.array([[1]]).astype(dtype)
        H = np.array([[1]]).astype(dtype)
        diffuse = True

        dict_param_llm = {
            0: {"matrix" : "Q", "index": (0, 0, 0), "constant": True, "bounds": (0.1, None)},
            1:  {"matrix" : "H", "index": (0, 0, 0), "constant": True, "bounds": (0.1, None)}
        }
        
        filter_map = {0: "level"}
        
        super().__init__(y, T, Z, R, Q, H, diffuse, dict_param_llm, filter_map)