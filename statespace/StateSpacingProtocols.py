from typing import List, Protocol

import numpy as np

class MathProtocol(Protocol):
    
    def _mm(M1: np.ndarray, M2: np.ndarray, dtype=np.float64) -> np.ndarray:
        ...

    def _bmm(M1: np.ndarray, M2: np.ndarray, dtype=np.float64)-> np.ndarray:
        ...
    
    def _T(M1: np.ndarray, dtype=np.float64) -> np.ndarray:
        ...

    def _inv(M1: np.ndarray, dtype=np.float64) -> np.ndarray:
        ...

    def _mm3(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, dtype=np.float64)-> np.ndarray:
        ...

    def _bmm3(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, dtype=np.float64)-> np.ndarray:
        ...

class KalmanProtocol(Protocol):

    def kalman_step(*args) -> List[np.ndarray]:
        ...

    def kalman_step_missing(*args) -> List[np.ndarray]:
        ...
    
    def kalman_filter(*args) -> List[np.ndarray]:
        ...
    
    def kalman_smoother(*args) -> List[np.ndarray]:
        ...

    def kalman_forecast(*args, time=10, dtype=np.float64) -> List[np.ndarray]:
        ...

    def log_likelihood(params, *args) -> float:
        ...

