from typing import List, Protocol
from StateSpacingProtocols import KalmanProtocol, MathProtocol
from utils import _initiate_variables, _get_nan_positions

import numpy as np

class KalmanV1(KalmanProtocol):

    # get an object of type math protocol to handle the math
    ssmath: MathProtocol

    def kalman_step(self, *args) -> List[np.ndarray]:
        """Computes the next step in the Kalman recursion using numpy math methods

        Returns:
            List[np.ndarray]: a_next, P_next, att, Ptt, M, K, F, v
        """
        
        T, Z, R, Q, H, a, P, y = args

        # prediction error: v
        v = y - self.ssmath._mm(Z, a)
        
        # prediction error variance: F
        F = self.ssmath._mm3(Z, P, self.ssmath._T(Z)) + H

        # incast-kalman gain: M
        M = self.ssmath._mm3(P, self.ssmath._T(Z), self.ssmath._inv(F)) 
        
        # kalman gain: K
        K = self.ssmath._mm(T, M)
        
        # incasted updates
        att = a + self.ssmath._mm(M, v)
        Ptt = P - self.ssmath._mm3(M, F, self.ssmath._T(M))
        
        a_next = self.ssmath._mm(T, att)
        P_next = self.ssmath._mm3(T, Ptt, self.ssmath._T(T)) + self.ssmath._mm3(R, Q, self.ssmath._T(R))
        
        return a_next, P_next, att, Ptt, M, K, F, v

    def kalman_step_missing(self, *args, dtype=np.float64) -> List[np.ndarray]:
        """Computes the next step in the Kalman recursion when y is missing using numpy math methods

        Returns:
            List[np.ndarray]: a_next, P_next, att, Ptt, M, K, F, v
        """

        T, Z, R, Q, H, a, P, y = args

        # dimension of observation vector
        s = int(y.shape[0])
        
        # prediction error: v
        v = np.nan
        
        # prediction error variance: F
        F = np.eye(s).astype(dtype) * np.inf
        
        # incast-kalman gain: M
        M = self.ssmath._mm3(P, self.ssmath._T(Z), self.ssmath._inv(F).astype(dtype)) 
        
        # kalman gain: K
        K = np.zeros_like(self.ssmath._mm(T, M))
        
        # incasted updates
        att = a
        Ptt = P
        
        a_next = self.ssmath._mm(T, att)
        P_next = self.ssmath._mm3(T, Ptt, self.ssmath._T(T)) + self.ssmath._mm3(R, Q, self.ssmath._T(R))
        
        return a_next, P_next, att, Ptt, M, K, F, v
    
    def kalman_filter(self, *args, dtype=np.float64) -> List[np.ndarray]:
        """
        perform all the steps of the Kalman filter
        """
        T, Z, R, Q, H, y, diffuse, shapes = args
        
        # number of observations
        n = shapes['n']
        
        # dimension of state vector
        p = shapes['p']

        # dimension of observation vector
        s = shapes['s']
        
        # initiate filters (a, att), filter variances (P, Ptt), errors (v), error variances (F), and Kalman gains (K, M)
        a, att, _, P, Ptt, _, v, F, K, M = _initiate_variables(p, s, n)
        
        # get positions of missing observations
        nan_pos_list = _get_nan_positions(y)
        
        # do a diffuse initialization
        if diffuse:
            a[:, :, 0] = 0
            P[:, :, 0] = P[:, :, 0] + 1e5 * np.eye(p, p)

        # iterate through time
        for t in range(1, n):
            
            if t-1 not in nan_pos_list:
                # a[0] contains a1, y[0] contains y1
                y_t = y[:, :, t-1]
                a[:, :, t], P[:, :, t], att[:, :, t-1], Ptt[:, :, t-1], M[:, :, t-1], K[:, :, t-1], F[:, :, t-1], v[:, :, t-1] = self._kalman_step(T[:, :, t-1], Z[:, :, t-1], R[:, :, t-1], Q[:, :, t-1], H[:, :, t-1], a[:, :, t-1], P[:, :, t-1], y_t)
            
            else:
                y_t = y[:, :, t-1]
                a[:, :, t], P[:, :, t], att[:, :, t-1], Ptt[:, :, t-1], M[:, :, t-1], K[:, :, t-1], F[:, :, t-1], v[:, :, t-1] = self._kalman_step_missing(T[:, :, t-1], Z[:, :, t-1], R[:, :, t-1], Q[:, :, t-1], H[:, :, t-1], a[:, :, t-1], P[:, :, t-1], y_t)
        
        if n-1 not in nan_pos_list:
            # do final incasting update
            y_t = y[:, :, n-1]
            _, _, att[:, :, n-1], Ptt[:, :, n-1], M[:, :, n-1], K[:, :, n-1], F[:, :, n-1], v[:, :, n-1] = self._kalman_step(T[:, :, t-1], Z[:, :, t-1], R[:, :, t-1], Q[:, :, t-1], H[:, :, t-1], a[:, :, n-1], P[:, :, n-1], y_t)
        
        else:
            # do final incasting update
            y_t = y[:, :, n-1]
            _, _, att[:, :, n-1], Ptt[:, :, n-1], M[:, :, n-1], K[:, :, n-1], F[:, :, n-1], v[:, :, n-1] = self._kalman_step_missing(T[:, :, t-1], Z[:, :, t-1], R[:, :, t-1], Q[:, :, t-1], H[:, :, t-1], a[:, :, n-1], P[:, :, n-1], y_t)
        
        return a, att, P, Ptt, F, v, K, M 
    
    def kalman_smoother(self, *args, dtype=np.float64) -> List[np.ndarray]:
        """perform kalman smoothing

        Returns:
            List[np.ndarray]: filters and states
        """
        T, Z, R, Q, H, a, P, v, F, K, shapes = args
        
        # number of observations
        n = shapes['n']
        
        # dimension of state vector
        p = shapes['p']

        # dimension of observation vector
        s = shapes['s']
        
        # instantiate a_hat
        a_hat = np.zeros_like(a)
        P_hat = np.zeros_like(P)
        
        # L' = (T - KZ)'
        L = T - np.einsum('ijn,jkn->ikn', K, Z)
        
        # r[n+1] = r_n = 0
        r = np.zeros((p, 1, n + 1)).astype(dtype)
        N = np.zeros((p, p, n + 1)).astype(dtype)
        
        # a[0] contains a_1, y[0] contains y_1
        
        # from T...0
        for t in range(n-1, -1, -1):
            
            # get r: depends on v -> adjust for missing values since v could be nan
            if np.isnan(v[:, :, t]):
                r[:, :, t] =  r[:, :, t+1]
            else:
                r[:, :, t] = self.ssmath._mm3(Z[:, :, t].T, self.ssmath._inv(F[:, :, t]), v[:, :, t]) + self.ssmath._mm(L[:, :, t].T, r[:, :, t+1])
        
            # get N
            N[:, :, t] = self.ssmath._mm3(Z[:, :, t].T, self.ssmath._inv(F[:, :, t]), Z[:, :, t]) + self.ssmath._mm3(L[:, :, t].T, N[:, :, t+1] , L[:, :, t]) 
            
            # enter smoothed filter and smoothed filter variance
            a_hat[:, :, t] = a[:, :, t] + self.ssmath._mm(P[:, :, t], r[:, :, t])
            P_hat[:, :, t] = P[:, :, t] - self.ssmath._mm3(P[:, :, t], N[:, :, t] , P[:, :, t])
            
        return a_hat, P_hat, r, N, L

    def kalman_forecast(self, *args, time=10, dtype=np.float64) -> List[np.ndarray]:
        """_summary_

        Args:
            time (int, optional): prediction window . Defaults to 10.
            dtype (_type_, optional): type of output. Defaults to np.float64.

        Returns:
            List[np.ndarray]: prediction of state, state variance and observable
        """
        T, Z, R, Q, H, att, Ptt, shapes = args
        
        # number of observations
        n = shapes['n']
        
        # dimension of state vector
        p = shapes['p']

        # dimension of observation vector
        s = shapes['s']
        
        # initiate filters (a, att), filter variances (P, Ptt), errors (v), error variances (F), and Kalman gains (K, M)
        a_forecast, P_forecast = np.zeros((p, 1, time + 1)).astype(dtype), np.zeros((p, p, n)).astype(dtype)
        a_forecast[:,:,0], P_forecast[:,:,0] = att[:, :, -1], Ptt[:, :, -1]
        
        for t in range(1, time + 1):
            a_forecast[:,:,t] = self.ssmath._mm(T, a_forecast[:,:,t-1])
            P_forecast[:,:,t] = self.ssmath._mm3(T, P_forecast[:,:,t-1], T.T) + self.ssmath._mm3(R, Q, R.T)
        
        return a_forecast[:,:,1:], P_forecast[:,:,1:]
