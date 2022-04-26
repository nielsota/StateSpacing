import pathlib

from typing import List, Optional, Dict
from torch.autograd import Variable
from scipy.optimize import minimize
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _mm(M1: np.ndarray, M2: np.ndarray, dtype=np.float64):
    """Matrix multiplication. (if later change to another package; e.g. torch, then only change utils and not core code)
    shapes [i,j] , [j,k] -> i,k

    Args:
        M1 (np.ndarray): matrix 1
        M2 (np.ndarray): matrix 2
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns:
        M1M2: matrix product
    """
    
    assert M1.shape[1] == M2.shape[0], "Number of cols in M1 should equal rows in M2"

    return np.matmul(M1, M2).astype(dtype)


def _bmm(M1: np.ndarray, M2: np.ndarray, dtype=np.float64):
    """ Batch matrix multiplication. (if later change to another package; e.g. torch, then only change utils and not core code)
    shapes [b,i,j] , [b,j,k] -> b,i,k

    Args:
        M1 (np.ndarray): _description_
        M2 (np.ndarray): _description_
        dtype (_type_, optional): _description_. Defaults to np.float64.

    Returns:
        _type_: _description_
    """
    
    assert M1.shape[2] == M2.shape[1], "Number of cols in M1 should equal rows in M2"
    assert M1.shape[0] == M2.shape[0], "Number of examples in M1 should equal number of examples in M2"

    return np.einsum('Bij, Bjk -> Bik', M1, M2).astype(dtype)


def _T(M1: np.ndarray, dtype=np.float64) -> np.ndarray:
    return M1.T


def _inv(M1: np.ndarray, dtype=np.float64) -> np.ndarray:
    """return inverse

    Args:
        M1 (np.ndarray): matrix M1
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns:
        np.ndarray: inverse of M1
    """
    return np.linalg.inv(M1)


def _mm3(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, dtype=np.float64):
    """computes product of 3 matrices

    Args:
        Mi (np.ndarray): matrix i. i=1,2,3
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns:
        M1M2M3: product of 3 matrices
    """

    assert M1.shape[1] == M2.shape[0], "Number of cols in M1 should equal rows in M2"
    assert M2.shape[1] == M3.shape[0], "Number of cols in M2 should equal rows in M3"

    return np.matmul(np.matmul(M1, M2), M3).astype(dtype)


def _bmm3(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, dtype=np.float64):
    """computes batch matrix product of 3 matrices

    Args:
        Mi (np.ndarray): Mi, i=1,2,3
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns:
        M1M2M3: batch matrix multiplication of 3 matrices
    """
    
    assert M1.shape[2] == M2.shape[1], "Number of cols in M1 should equal rows in M2"
    assert M2.shape[2] == M1.shape[1], "Number of cols in M2 should equal rows in M3"
    assert M1.shape[0] == M2.shape[0], "Number of examples in M1 should equal number of examples in M2"
    assert M2.shape[0] == M3.shape[0], "Number of examples in M1 should equal number of examples in M2"

    return _bmm(_bmm(M1, M2), M3).astype(dtype)


def _initiate_variables(p: int, s: int, n: int, dtype=np.float64):
    """_summary_

    Args:
        p (int): dimension of state vector
        s (int): dimension of observation vector
        n (int): dimension of time direction
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns:
        Initiated versions of a, att, a_hat, P, Ptt, P_hat, v, F, K and M
    """
    return np.zeros((p, 1, n)).astype(dtype), np.zeros((p, 1, n)).astype(dtype), np.zeros((p, 1, n)).astype(dtype), np.zeros((p, p, n)).astype(dtype), np.zeros((p, p, n)).astype(dtype), np.zeros((p, p, n)).astype(dtype), np.zeros((s, 1, n)).astype(dtype), np.zeros((s, s, n)).astype(dtype), np.zeros((p, s, n)).astype(dtype), np.zeros((p, s, n)).astype(dtype)


def _map_vector_to_matrices(params, param_map, *args, dtype=np.float64):
    """ Maps the elements of params to their correct position in the state matrices T, Z, R, Q and H

    Args:
        params (np.ndarray): vector of parameters
        param_map (np.ndarray): dictionary containing key (int) which corresponds to the index in params, and 
                            value containing which matrix it belongs to and at which index 
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns:
         T, Z, R, Q, H, with params put in the correct position
    """
    
    T, Z, R, Q, H = args

    for k, v in param_map.items():

        state_matrix_asstr = v["matrix"]
        i, j, t = v["index"]
        constant_through_time = v["constant"]

        # will work even if shape if 3D, as long as time is last dimension
        if state_matrix_asstr == "Q":
            Q[i, j] = float(params[k])
        if state_matrix_asstr == "H":
            H[i, j] = float(params[k])
        if state_matrix_asstr == "T":
            T[i, j] = float(params[k])
        if state_matrix_asstr == "Z":
            Z[i, j] = float(params[k])
        if state_matrix_asstr == "R":
            R[i, j] = float(params[k])
    
    return T.astype(dtype), Z.astype(dtype), R.astype(dtype), Q.astype(dtype), H.astype(dtype)


def _get_nan_positions(y: np.ndarray) -> List[int]:
    """ Get a list of the moments in time that contain missing values

    Args:
        y (np.ndarray): observation vector

    Returns:
        List[int]: list of indices where values are missing
    """

    if y.ndim > 1:
        y = np.squeeze(y)

    nan_pos_list = np.squeeze(np.argwhere(np.isnan(y))).tolist()

    return nan_pos_list


def _remove_nan_tensor(tensor: np.ndarray):
    """remove nans from a tensor

    Args:
        tensor (np.ndarray): _description_

    Returns:
        output: input with inf's removed
    """
    tensor = tensor.transpose(2, 0, 1)
    return tensor[~np.any(np.isnan(tensor), axis =1)][:, :, None].transpose(1,2,0)


def _remove_inf_tensor(tensor: np.ndarray):
    """remove infs from a tensor

    Args:
        tensor (np.ndarray): input

    Returns:
        output: input with inf's removed 
    """
    tensor = tensor.transpose(2, 0, 1)
    return tensor[~np.any(np.isinf(tensor), axis =1)][:, :, None].transpose(1,2,0)

def _get_bounds(dict_params: Dict[int, Dict]):
    """ get the bounds from a dict_params objects

    Args:
        Dict (_type_): _description_
    """
    bounds = []
    for k, v in dict_params.items():
        bounds.append(v["bounds"])
        
    bounds = tuple(bounds)

    return bounds

def read_target_from_path(path: pathlib.Path, dtype=np.float64, header=None) -> torch.Tensor:
    
    if not path.exists():
        raise ValueError(f"path: {path} does not exist")
    
    # read data
    data = np.squeeze(pd.read_csv(path, header=header).values)
    
    # if array of shape (n,)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    n = data.shape[0]
    p = data.shape[1]
    
    y = data.T
    y = y[:, None, :]
  
    return y.astype(dtype)