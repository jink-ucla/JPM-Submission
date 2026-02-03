"""
Kalman Filter Implementation for Linear-Gaussian State Space Models
Part 1 Warm-up (I): Linear-Gaussian SSM with Kalman Filter

Implements:
- Standard Kalman Filter
- Joseph stabilized covariance updates
- Numerical stability analysis (condition numbers)

Reference: Doucet(09) - Example 2
"""

import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


class KalmanFilter:
    """
    Kalman Filter for Linear-Gaussian State Space Models

    State Space Model:
        x_t = F @ x_{t-1} + B @ u_t + w_t,  w_t ~ N(0, Q)
        y_t = H @ x_t + v_t,                v_t ~ N(0, R)

    where:
        x_t: state vector (n_dim,)
        y_t: observation vector (obs_dim,)
        u_t: control input (optional)
        F: state transition matrix
        H: observation matrix
        Q: process noise covariance
        R: observation noise covariance
        B: control input matrix (optional)
    """

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 B: np.ndarray = None, use_joseph: bool = True):
        """
        Initialize Kalman Filter

        Args:
            F: State transition matrix (n_dim, n_dim)
            H: Observation matrix (obs_dim, n_dim)
            Q: Process noise covariance (n_dim, n_dim)
            R: Observation noise covariance (obs_dim, obs_dim)
            B: Control input matrix (n_dim, control_dim), optional
            use_joseph: Whether to use Joseph stabilized covariance update
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B
        self.use_joseph = use_joseph

        self.n_dim = F.shape[0]
        self.obs_dim = H.shape[0]

        # Initialize state
        self.x = None
        self.P = None

        # Store history
        self.history = {
            'x_pred': [],
            'P_pred': [],
            'x_filt': [],
            'P_filt': [],
            'K': [],
            'innovation': [],
            'S': [],
            'cond_P': [],
            'cond_S': []
        }

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """Initialize filter with initial state and covariance"""
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self, u: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step

        Args:
            u: Control input (optional)

        Returns:
            x_pred: Predicted state
            P_pred: Predicted covariance
        """
        # State prediction
        x_pred = self.F @ self.x
        if u is not None and self.B is not None:
            x_pred += self.B @ u

        # Covariance prediction
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Ensure symmetry
        P_pred = 0.5 * (P_pred + P_pred.T)

        return x_pred, P_pred

    def update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step

        Args:
            y: Observation
            x_pred: Predicted state
            P_pred: Predicted covariance

        Returns:
            x_filt: Filtered state
            P_filt: Filtered covariance
        """
        # Innovation
        innovation = y - self.H @ x_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        S = 0.5 * (S + S.T)  # Ensure symmetry

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # State update
        x_filt = x_pred + K @ innovation

        # Covariance update
        if self.use_joseph:
            # Joseph stabilized form
            I_KH = np.eye(self.n_dim) - K @ self.H
            P_filt = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            # Standard form
            P_filt = (np.eye(self.n_dim) - K @ self.H) @ P_pred

        # Ensure symmetry and positive definiteness
        P_filt = 0.5 * (P_filt + P_filt.T)

        # Store diagnostics
        cond_P = np.linalg.cond(P_pred)
        cond_S = np.linalg.cond(S)

        return x_filt, P_filt, K, innovation, S, cond_P, cond_S

    def filter_step(self, y: np.ndarray, u: np.ndarray = None) -> Dict:
        """
        Single filter step (predict + update)

        Args:
            y: Observation
            u: Control input (optional)

        Returns:
            Dictionary with filtered state and diagnostics
        """
        # Predict
        x_pred, P_pred = self.predict(u)

        # Update
        x_filt, P_filt, K, innovation, S, cond_P, cond_S = self.update(y, x_pred, P_pred)

        # Store in history
        self.history['x_pred'].append(x_pred.copy())
        self.history['P_pred'].append(P_pred.copy())
        self.history['x_filt'].append(x_filt.copy())
        self.history['P_filt'].append(P_filt.copy())
        self.history['K'].append(K.copy())
        self.history['innovation'].append(innovation.copy())
        self.history['S'].append(S.copy())
        self.history['cond_P'].append(cond_P)
        self.history['cond_S'].append(cond_S)

        # Update current state
        self.x = x_filt
        self.P = P_filt

        return {
            'x_pred': x_pred,
            'P_pred': P_pred,
            'x_filt': x_filt,
            'P_filt': P_filt,
            'K': K,
            'innovation': innovation,
            'S': S,
            'cond_P': cond_P,
            'cond_S': cond_S
        }

    def filter(self, observations: np.ndarray, controls: np.ndarray = None) -> Dict:
        """
        Run Kalman filter on sequence of observations

        Args:
            observations: Array of observations (T, obs_dim)
            controls: Array of control inputs (T, control_dim), optional

        Returns:
            Dictionary with filtered states and diagnostics
        """
        T = len(observations)

        for t in range(T):
            y = observations[t]
            u = controls[t] if controls is not None else None
            self.filter_step(y, u)

        return self.get_history()

    def get_history(self) -> Dict:
        """Get filtering history"""
        return {
            'x_pred': np.array(self.history['x_pred']),
            'P_pred': np.array(self.history['P_pred']),
            'x_filt': np.array(self.history['x_filt']),
            'P_filt': np.array(self.history['P_filt']),
            'K': np.array(self.history['K']),
            'innovation': np.array(self.history['innovation']),
            'S': np.array(self.history['S']),
            'cond_P': np.array(self.history['cond_P']),
            'cond_S': np.array(self.history['cond_S'])
        }

    def reset_history(self):
        """Reset history"""
        for key in self.history:
            self.history[key] = []


def generate_lgssm_data(F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                        x0: np.ndarray, T: int, B: np.ndarray = None,
                        controls: np.ndarray = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from Linear-Gaussian State Space Model

    Args:
        F: State transition matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Observation noise covariance
        x0: Initial state
        T: Number of time steps
        B: Control input matrix (optional)
        controls: Control inputs (optional)
        seed: Random seed

    Returns:
        states: True states (T, n_dim)
        observations: Observations (T, obs_dim)
    """
    if seed is not None:
        np.random.seed(seed)

    n_dim = F.shape[0]
    obs_dim = H.shape[0]

    states = np.zeros((T, n_dim))
    observations = np.zeros((T, obs_dim))

    # Cholesky decompositions for sampling
    L_Q = np.linalg.cholesky(Q)
    L_R = np.linalg.cholesky(R)

    x = x0.copy()

    for t in range(T):
        # Generate state
        w = L_Q @ np.random.randn(n_dim)
        x = F @ x + w
        if controls is not None and B is not None:
            x += B @ controls[t]

        # Generate observation
        v = L_R @ np.random.randn(obs_dim)
        y = H @ x + v

        states[t] = x
        observations[t] = y

    return states, observations


def compute_nees(true_states: np.ndarray, filtered_means: np.ndarray,
                 filtered_covs: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Estimation Error Squared (NEES)

    NEES_t = (x_t - x̂_t)^T P_t^{-1} (x_t - x̂_t)

    For a consistent filter, NEES should be chi-squared distributed with n_dim degrees of freedom
    """
    T = len(true_states)
    nees = np.zeros(T)

    for t in range(T):
        error = true_states[t] - filtered_means[t]
        P_inv = np.linalg.inv(filtered_covs[t])
        nees[t] = error.T @ P_inv @ error

    return nees


def compute_nis(innovations: np.ndarray, innovation_covs: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Innovation Squared (NIS)

    NIS_t = ν_t^T S_t^{-1} ν_t

    For a consistent filter, NIS should be chi-squared distributed with obs_dim degrees of freedom
    """
    T = len(innovations)
    nis = np.zeros(T)

    for t in range(T):
        nu = innovations[t]
        S_inv = np.linalg.inv(innovation_covs[t])
        nis[t] = nu.T @ S_inv @ nu

    return nis
