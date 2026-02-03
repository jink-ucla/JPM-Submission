"""
Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
Part 1 (II): Nonlinear/Non-Gaussian SSM

Implements:
- Extended Kalman Filter (EKF) with linearization
- Unscented Kalman Filter (UKF) with sigma points
- Analysis of linearization accuracy and sigma point failures
"""

import numpy as np
from typing import Tuple, Dict, Callable
from nonlinear_models import NonlinearSSM


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF)

    Linearizes nonlinear functions using first-order Taylor expansion:
        f(x) ≈ f(x̂) + F(x̂) @ (x - x̂)
        h(x) ≈ h(x̂) + H(x̂) @ (x - x̂)

    where F and H are Jacobians
    """

    def __init__(self, model: NonlinearSSM, f_jacobian: Callable, h_jacobian: Callable):
        """
        Initialize EKF

        Args:
            model: Nonlinear SSM
            f_jacobian: Function to compute Jacobian of f
            h_jacobian: Function to compute Jacobian of h
        """
        self.model = model
        self.f = model.f
        self.h = model.h
        self.Q = model.Q
        self.R = model.R
        self.f_jacobian = f_jacobian
        self.h_jacobian = h_jacobian

        self.n_dim = model.n_dim
        self.obs_dim = model.obs_dim

        self.x = None
        self.P = None

        self.history = {
            'x_pred': [],
            'P_pred': [],
            'x_filt': [],
            'P_filt': [],
            'K': [],
            'innovation': [],
            'F_matrices': [],
            'H_matrices': [],
            'linearization_error': []
        }

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """Initialize filter"""
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step with linearization"""
        # Compute Jacobian at current state
        F = self.f_jacobian(self.x)

        # Predicted state (use nonlinear function)
        x_pred = self.f(self.x)

        # Predicted covariance (use linearized dynamics)
        P_pred = F @ self.P @ F.T + self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        return x_pred, P_pred, F

    def update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> Tuple:
        """Update step with linearization"""
        # Compute Jacobian at predicted state
        H = self.h_jacobian(x_pred)

        # Predicted observation (use nonlinear function)
        y_pred = self.h(x_pred)

        # Innovation
        innovation = y - y_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R
        S = 0.5 * (S + S.T)

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # State update
        x_filt = x_pred + K @ innovation

        # Covariance update
        I_KH = np.eye(self.n_dim) - K @ H
        P_filt = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        P_filt = 0.5 * (P_filt + P_filt.T)

        return x_filt, P_filt, K, innovation, H

    def compute_linearization_error(self, x_nominal: np.ndarray, P: np.ndarray,
                                      func: Callable, jacobian: np.ndarray,
                                      n_samples: int = 100) -> float:
        """
        Estimate linearization error using Monte Carlo sampling

        Computes E[||f(x) - (f(x̂) + J(x̂)(x - x̂))||²] where x ~ N(x̂, P)

        Args:
            x_nominal: Linearization point
            P: Covariance around nominal point
            func: Nonlinear function
            jacobian: Jacobian at nominal point
            n_samples: Number of MC samples

        Returns:
            RMS linearization error
        """
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            return 0.0

        f_nominal = func(x_nominal)
        errors = []

        for _ in range(n_samples):
            # Sample from distribution
            x_sample = x_nominal + L @ np.random.randn(self.n_dim)

            # True function value
            f_true = func(x_sample)

            # Linearized approximation
            f_linear = f_nominal + jacobian @ (x_sample - x_nominal)

            # Error
            error = np.linalg.norm(f_true - f_linear)
            errors.append(error)

        return np.sqrt(np.mean(np.array(errors) ** 2))

    def filter_step(self, y: np.ndarray) -> Dict:
        """Single filter step"""
        # Predict
        x_pred, P_pred, F = self.predict()

        # Update
        x_filt, P_filt, K, innovation, H = self.update(y, x_pred, P_pred)

        # Compute linearization error for observation function
        linearization_error = self.compute_linearization_error(
            x_pred, P_pred, self.h, H, n_samples=50
        )

        # Store history
        self.history['x_pred'].append(x_pred.copy())
        self.history['P_pred'].append(P_pred.copy())
        self.history['x_filt'].append(x_filt.copy())
        self.history['P_filt'].append(P_filt.copy())
        self.history['K'].append(K.copy())
        self.history['innovation'].append(innovation.copy())
        self.history['F_matrices'].append(F.copy())
        self.history['H_matrices'].append(H.copy())
        self.history['linearization_error'].append(linearization_error)

        # Update state
        self.x = x_filt
        self.P = P_filt

        return {
            'x_pred': x_pred,
            'P_pred': P_pred,
            'x_filt': x_filt,
            'P_filt': P_filt,
            'K': K,
            'innovation': innovation
        }

    def filter(self, observations: np.ndarray) -> Dict:
        """Run EKF on sequence"""
        for y in observations:
            self.filter_step(y)
        return self.get_history()

    def get_history(self) -> Dict:
        """Get history"""
        return {
            'x_pred': np.array(self.history['x_pred']),
            'P_pred': np.array(self.history['P_pred']),
            'x_filt': np.array(self.history['x_filt']),
            'P_filt': np.array(self.history['P_filt']),
            'K': np.array(self.history['K']),
            'innovation': np.array(self.history['innovation']),
            'F_matrices': np.array(self.history['F_matrices']),
            'H_matrices': np.array(self.history['H_matrices']),
            'linearization_error': np.array(self.history['linearization_error'])
        }


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF)

    Uses sigma points to capture nonlinearity without computing Jacobians.
    The unscented transform propagates mean and covariance through nonlinear functions.
    """

    def __init__(self, model: NonlinearSSM, alpha: float = 1e-3, beta: float = 2.0, kappa: float = None):
        """
        Initialize UKF

        Args:
            model: Nonlinear SSM
            alpha: Spread of sigma points (typically 1e-4 to 1)
            beta: Prior knowledge of distribution (2 is optimal for Gaussian)
            kappa: Secondary scaling parameter (typically 3 - n_dim or 0)
        """
        self.model = model
        self.f = model.f
        self.h = model.h
        self.Q = model.Q
        self.R = model.R

        self.n_dim = model.n_dim
        self.obs_dim = model.obs_dim

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - self.n_dim

        # Compute lambda
        self.lambda_ = alpha ** 2 * (self.n_dim + self.kappa) - self.n_dim

        # Compute weights
        self.compute_weights()

        self.x = None
        self.P = None

        self.history = {
            'x_pred': [],
            'P_pred': [],
            'x_filt': [],
            'P_filt': [],
            'K': [],
            'innovation': [],
            'sigma_points': [],
            'sigma_failures': []
        }

    def compute_weights(self):
        """Compute sigma point weights"""
        n = self.n_dim
        lambda_ = self.lambda_

        # Mean weights
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1 / (2 * (n + lambda_))

        # Covariance weights
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lambda_))

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points

        Returns:
            sigma_points: (2n+1, n) array of sigma points
        """
        n = self.n_dim
        lambda_ = self.lambda_

        # Compute matrix square root
        try:
            L = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            L = eigvecs @ np.diag(np.sqrt(eigvals * (n + lambda_)))

        # Generate sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x

        for i in range(n):
            sigma_points[i + 1] = x + L[:, i]
            sigma_points[n + i + 1] = x - L[:, i]

        return sigma_points

    def unscented_transform(self, sigma_points: np.ndarray, func: Callable,
                          noise_cov: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply unscented transform

        Args:
            sigma_points: Input sigma points
            func: Nonlinear function
            noise_cov: Additive noise covariance (optional)

        Returns:
            mean: Transformed mean
            cov: Transformed covariance
        """
        # Propagate sigma points through function
        n_points = len(sigma_points)
        transformed = []

        for i in range(n_points):
            try:
                y = func(sigma_points[i])
                transformed.append(y)
            except:
                # If function fails, use mean as fallback
                if i == 0:
                    raise
                transformed.append(transformed[0])

        transformed = np.array(transformed)

        # Compute mean
        mean = np.sum(self.Wm[:, np.newaxis] * transformed, axis=0)

        # Compute covariance
        deviations = transformed - mean
        cov = np.sum(self.Wc[:, np.newaxis, np.newaxis] *
                     (deviations[:, :, np.newaxis] @ deviations[:, np.newaxis, :]), axis=0)

        # Add noise covariance
        if noise_cov is not None:
            cov += noise_cov

        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)

        return mean, cov

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """Initialize filter"""
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step using unscented transform"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)

        # Propagate through state transition
        x_pred, P_pred = self.unscented_transform(sigma_points, self.f, self.Q)

        return x_pred, P_pred, sigma_points

    def update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray,
               sigma_points_pred: np.ndarray) -> Tuple:
        """Update step using unscented transform"""
        # Generate sigma points from predicted state
        sigma_points = self.generate_sigma_points(x_pred, P_pred)

        # Propagate through observation function
        y_pred, S = self.unscented_transform(sigma_points, self.h, self.R)

        # Compute cross-covariance
        deviations_x = sigma_points - x_pred
        deviations_y = []
        for sp in sigma_points:
            deviations_y.append(self.h(sp) - y_pred)
        deviations_y = np.array(deviations_y)

        Pxy = np.sum(self.Wc[:, np.newaxis, np.newaxis] *
                     (deviations_x[:, :, np.newaxis] @ deviations_y[:, np.newaxis, :]), axis=0)

        # Kalman gain
        K = Pxy @ np.linalg.inv(S)

        # Innovation
        innovation = y - y_pred

        # State update
        x_filt = x_pred + K @ innovation

        # Covariance update
        P_filt = P_pred - K @ S @ K.T
        P_filt = 0.5 * (P_filt + P_filt.T)

        return x_filt, P_filt, K, innovation, sigma_points

    def filter_step(self, y: np.ndarray) -> Dict:
        """Single filter step"""
        # Predict
        x_pred, P_pred, sigma_points_pred = self.predict()

        # Update
        x_filt, P_filt, K, innovation, sigma_points = self.update(y, x_pred, P_pred, sigma_points_pred)

        # Check for sigma point failures (nan or inf)
        sigma_failure = np.any(np.isnan(sigma_points)) or np.any(np.isinf(sigma_points))

        # Store history
        self.history['x_pred'].append(x_pred.copy())
        self.history['P_pred'].append(P_pred.copy())
        self.history['x_filt'].append(x_filt.copy())
        self.history['P_filt'].append(P_filt.copy())
        self.history['K'].append(K.copy())
        self.history['innovation'].append(innovation.copy())
        self.history['sigma_points'].append(sigma_points.copy())
        self.history['sigma_failures'].append(sigma_failure)

        # Update state
        self.x = x_filt
        self.P = P_filt

        return {
            'x_pred': x_pred,
            'P_pred': P_pred,
            'x_filt': x_filt,
            'P_filt': P_filt,
            'K': K,
            'innovation': innovation
        }

    def filter(self, observations: np.ndarray) -> Dict:
        """Run UKF on sequence"""
        for y in observations:
            self.filter_step(y)
        return self.get_history()

    def get_history(self) -> Dict:
        """Get history"""
        return {
            'x_pred': np.array(self.history['x_pred']),
            'P_pred': np.array(self.history['P_pred']),
            'x_filt': np.array(self.history['x_filt']),
            'P_filt': np.array(self.history['P_filt']),
            'K': np.array(self.history['K']),
            'innovation': np.array(self.history['innovation']),
            'sigma_failures': np.array(self.history['sigma_failures'])
        }
