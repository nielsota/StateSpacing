from typing import Protocol
from StateSpacingProtocols import MathProtocol

import numpy as np

class StateSpaceMathNumpy(MathProtocol):
    
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
        return np.matmul(M1, M2).astype(dtype)

    def _bmm(M1: np.ndarray, M2: np.ndarray, dtype=np.float64):
        """ Batch matrix multiplication. (if later change to another package; e.g. torch, then only change utils and not core code)
        shapes [b,i,j] , [b,j,k] -> b,i,k

        Args:
            M1 (np.ndarray): _description_
            M2 (np.ndarray): _description_
            dtype (_type_, optional): _description_. Defaults to np.float64.

        Returns:
           batch multiplied matrix
        """
        return np.einsum('Bij, Bjk -> Bik', M1, M2).astype(dtype)
    
    def _T(M1: np.ndarray, dtype=np.float64) -> np.ndarray:
        """compute transpose of a matrix

        Args:
            M1 (np.ndarray): _description_
            dtype (_type_, optional): _description_. Defaults to np.float64.

        Returns:
            np.ndarray: _description_
        """
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
        return np.matmul(np.matmul(M1, M2), M3).astype(dtype)

    def _bmm3(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, dtype=np.float64):
        """computes batch matrix product of 3 matrices

    Args:
        Mi (np.ndarray): Mi, i=1,2,3
        dtype (_type_, optional): type. Defaults to np.float64.

    Returns: ev
        M1M2M3: batch matrix multiplication of 3 matrices
    """
        return np.einsum('Bij, Bjk -> Bik', np.einsum('Bij, Bjk -> Bik', M1, M2).astype(dtype), M3).astype(dtype)
  