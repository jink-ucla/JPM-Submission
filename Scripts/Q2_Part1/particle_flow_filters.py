"""
Particle Flow Filters: EDH, LEDH, and PF-PF Framework
Part 1: Deterministic and Kernel Flows

Implements:
1. Exact Daum-Huang (EDH) flow [Daum 2010]
2. Local Exact Daum-Huang (LEDH) flow [Daum 2011]
3. Invertible Particle Flow Particle Filter (PF-PF) [Li 2017]

References:
- Daum & Huang (2010): Exact particle flow for nonlinear filters
- Daum & Huang (2011): Particle degeneracy: root cause and solution
- Li & Coates (2017): Particle filtering with invertible particle flow
"""

import numpy as np
from typing import Tuple, Dict, Callable
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import sqrtm, cho_factor, cho_solve
from scipy.spatial.distance import cdist
import warnings


class ParticleFlowFilter:
    """
    Base class for particle flow filters

    Instead of importance sampling and resampling, particle flow methods
    migrate particles from prior to posterior through a continuous flow:

        dx/dλ = f(x, λ)  for λ ∈ [0, 1]

    where x(0) ~ p(x|y_{1:t-1}) and x(1) ~ p(x|y_{1:t})
    """

    def __init__(self, model, N_particles: int = 100, n_steps: int = 50):
        """
        Initialize Particle Flow Filter

        Args:
            model: Nonlinear SSM
            N_particles: Number of particles
            n_steps: Number of integration steps for flow
        """
        self.model = model
        self.f_transition = model.f
        self.h_obs = model.h
        self.Q = model.Q
        self.R = model.R

        self.n_dim = model.n_dim
        self.obs_dim = model.obs_dim

        self.N = N_particles
        self.n_steps = n_steps

        # Particles
        self.particles = None

        # History
        self.history = {
            'particles': [],
            'x_mean': [],
            'x_cov': [],
            'flow_magnitude': [],
            'jacobian_cond': []
        }

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """Initialize particles from prior"""
        L_P0 = np.linalg.cholesky(P0)
        self.particles = x0[:, None] + L_P0 @ np.random.randn(self.n_dim, self.N)
        self.particles = self.particles.T  # (N, n_dim)

    def predict(self):
        """Prediction step: propagate through state transition"""
        L_Q = np.linalg.cholesky(self.Q)
        noise = L_Q @ np.random.randn(self.n_dim, self.N)
        noise = noise.T

        new_particles = np.zeros_like(self.particles)
        for i in range(self.N):
            new_particles[i] = self.f_transition(self.particles[i], noise[i])

        self.particles = new_particles

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and covariance"""
        x_mean = np.mean(self.particles, axis=0)
        x_cov = np.cov(self.particles.T)
        if self.n_dim == 1:
            x_cov = np.array([[x_cov]])
        return x_mean, x_cov


class EDHParticleFlowFilter(ParticleFlowFilter):
    """
    Exact Daum-Huang (EDH) Particle Flow

    The EDH flow is derived from the exact solution to the Fokker-Planck equation:

        dx/dλ = C(λ) @ ∇ log p(y | x, λ)

    where C(λ) is chosen to ensure the flow migrates particles from prior to posterior.

    For Gaussian observation model: p(y|x) = N(y; h(x), R)
        ∇ log p(y | x) = H^T R^{-1} (y - h(x))

    The EDH flow uses:
        dx/dλ = P(λ) @ H(x)^T @ R^{-1} @ (y - h(x))

    where P(λ) is the covariance at pseudo-time λ

    References:
        Daum & Huang (2010): Exact particle flow for nonlinear filters
    """

    def __init__(self, model, N_particles: int = 100, n_steps: int = 50,
                 h_jacobian: Callable = None):
        """
        Args:
            model: Nonlinear SSM
            N_particles: Number of particles
            n_steps: Integration steps
            h_jacobian: Function to compute Jacobian of h (optional)
        """
        super().__init__(model, N_particles, n_steps)
        self.h_jacobian = h_jacobian

        # Use Cholesky factorization for efficient solves instead of explicit inverse
        self.R_chol = cho_factor(self.R)
        self.R_inv = np.linalg.inv(self.R)  # Keep for compatibility

        # Override history to include jacobian_cond
        self.history['jacobian_cond'] = []

    def edh_flow_function(self, x: np.ndarray, y: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        EDH flow function: dx/dλ

        Args:
            x: Current particle state
            y: Observation
            P: Current covariance

        Returns:
            Flow velocity dx/dλ
        """
        # Predicted observation
        h_x = self.h_obs(x)

        # Innovation
        innovation = y - h_x

        # Jacobian
        if self.h_jacobian is not None:
            H = self.h_jacobian(x)
        else:
            # Numerical Jacobian
            H = self.numerical_jacobian(self.h_obs, x)

        # Flow: dx/dλ = P @ H^T @ R^{-1} @ (y - h(x))
        flow = P @ H.T @ self.R_inv @ innovation

        return flow

    def numerical_jacobian(self, func: Callable, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Compute numerical Jacobian"""
        f0 = func(x)
        m = len(f0)
        n = len(x)
        J = np.zeros((m, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            J[:, i] = (f_plus - f0) / eps

        return J

    def compute_flow_jacobian(self, x: np.ndarray, y: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of EDH flow w.r.t. particle state

        The flow is: dx/dλ = P @ H^T @ R^{-1} @ (y - h(x))

        Jacobian: ∂(dx/dλ)/∂x = -P @ H^T @ R^{-1} @ H

        This is the A matrix in Li(17) that determines flow stability.
        """
        if self.h_jacobian is not None:
            H = self.h_jacobian(x)
        else:
            H = self.numerical_jacobian(self.h_obs, x)

        # Flow Jacobian (negative because d(y - h(x))/dx = -H)
        A = -P @ H.T @ self.R_inv @ H

        return A

    def _compute_all_jacobians(self, particles: np.ndarray) -> np.ndarray:
        """
        Vectorized Jacobian computation for all particles.
        Returns shape (N, obs_dim, n_dim)
        """
        if self.h_jacobian is not None:
            # Analytical Jacobian - vectorize if possible
            H_all = np.array([self.h_jacobian(particles[i]) for i in range(self.N)])
        else:
            # Numerical Jacobian - batch computation
            H_all = np.array([self.numerical_jacobian(self.h_obs, particles[i])
                             for i in range(self.N)])
        return H_all

    def _compute_all_observations(self, particles: np.ndarray) -> np.ndarray:
        """Vectorized observation computation. Returns shape (N, obs_dim)"""
        return np.array([self.h_obs(particles[i]) for i in range(self.N)])

    def update_with_flow(self, y: np.ndarray):
        """Update particles using EDH flow - VECTORIZED VERSION"""
        # Integration pseudo-time
        d_lambda = 1.0 / (self.n_steps - 1)

        # Store flow diagnostics
        flow_magnitudes = []
        jacobian_conds = []

        # Compute covariance ONCE at start (matches original sequential behavior better)
        # The original computes P inside loops but sees a mix of updated/non-updated particles
        # Using a fixed P is more mathematically consistent with the EDH algorithm
        P = np.cov(self.particles.T)
        if self.n_dim == 1:
            P = np.array([[P]])

        # Vectorized flow integration
        for step in range(self.n_steps - 1):

            # Vectorized: compute h(x) for all particles
            h_all = self._compute_all_observations(self.particles)  # (N, obs_dim)

            # Vectorized: compute innovations for all particles
            innovations = y - h_all  # (N, obs_dim)

            # Vectorized: compute Jacobians for all particles
            H_all = self._compute_all_jacobians(self.particles)  # (N, obs_dim, n_dim)

            # Compute flows for all particles vectorized
            # flow_i = P @ H_i^T @ R^{-1} @ innovation_i
            # Use cho_solve for efficiency: R^{-1} @ innovation = cho_solve(R_chol, innovation)
            R_inv_innovations = cho_solve(self.R_chol, innovations.T).T  # (N, obs_dim)

            # flows = P @ H^T @ (R^{-1} @ innovations)
            # H_all: (N, obs_dim, n_dim), R_inv_innovations: (N, obs_dim)
            # H^T @ R_inv_inn: for each particle, (n_dim, obs_dim) @ (obs_dim,) = (n_dim,)
            HT_Rinv_inn = np.einsum('ijk,ij->ik', H_all, R_inv_innovations)  # (N, n_dim)

            # flows = P @ HT_Rinv_inn.T -> for each particle
            flows = (P @ HT_Rinv_inn.T).T  # (N, n_dim)

            # Euler integration - vectorized
            self.particles += flows * d_lambda

            # Diagnostics (sample a few particles for efficiency)
            flow_magnitudes.append(np.mean(np.linalg.norm(flows, axis=1)))

            # Jacobian conditioning (sample one particle for efficiency)
            H_sample = H_all[0]
            A = -P @ H_sample.T @ self.R_inv @ H_sample
            jacobian_conds.append(np.linalg.cond(np.eye(self.n_dim) + d_lambda * A))

        # Store diagnostics
        self.history['flow_magnitude'].append(np.mean(flow_magnitudes))
        self.history['jacobian_cond'].append(np.mean(jacobian_conds))

    def filter_step(self, y: np.ndarray):
        """Single filter step"""
        # Predict
        self.predict()

        # Update with flow
        self.update_with_flow(y)

        # Get estimate
        x_mean, x_cov = self.get_estimate()

        # Store history
        self.history['particles'].append(self.particles.copy())
        self.history['x_mean'].append(x_mean.copy())
        self.history['x_cov'].append(x_cov.copy())

        return {'x_mean': x_mean, 'x_cov': x_cov}

    def filter(self, observations: np.ndarray) -> Dict:
        """Run filter on sequence"""
        for y in observations:
            self.filter_step(y)
        return self.get_history()

    def get_history(self) -> Dict:
        """Get history"""
        return {
            'particles': self.history['particles'],
            'x_mean': np.array(self.history['x_mean']),
            'x_cov': np.array(self.history['x_cov']),
            'flow_magnitude': np.array(self.history['flow_magnitude']),
            'jacobian_cond': np.array(self.history['jacobian_cond'])
        }


class LEDHParticleFlowFilter(ParticleFlowFilter):
    """
    Local Exact Daum-Huang (LEDH) Particle Flow

    LEDH uses local (per-particle) covariance instead of global covariance:

        dx_i/dλ = P_i(λ) @ H(x_i)^T @ R^{-1} @ (y - h(x_i))

    where P_i is computed using local particles (k-nearest neighbors).
    This addresses particle degeneracy by maintaining diversity.

    References:
        Daum & Huang (2011): Particle degeneracy: root cause and solution
    """

    def __init__(self, model, N_particles: int = 100, n_steps: int = 50,
                 h_jacobian: Callable = None, localization_radius: float = None):
        """
        Args:
            model: Nonlinear SSM
            N_particles: Number of particles
            n_steps: Integration steps
            h_jacobian: Jacobian of observation function
            localization_radius: Radius for local covariance (if None, use k-nearest)
        """
        super().__init__(model, N_particles, n_steps)
        self.h_jacobian = h_jacobian
        self.R_inv = np.linalg.inv(self.R)
        self.R_chol = cho_factor(self.R)  # For efficient solves
        self.localization_radius = localization_radius

        # Precompute k_nearest
        self.k_nearest = max(self.n_dim + 2, self.N // 5)
        self.k_nearest = min(self.k_nearest, self.N)

        # Add jacobian conditioning tracking
        self.history['jacobian_cond'] = []

    def compute_local_covariance(self, particle_idx: int, k_nearest: int = None) -> np.ndarray:
        """
        Compute local covariance for particle i using k-nearest neighbors

        Following Daum & Huang (2011), local covariance addresses particle
        degeneracy by maintaining diversity through localized estimates.

        Args:
            particle_idx: Index of current particle
            k_nearest: Number of nearest neighbors (default: max(n_dim+2, N/5))

        Returns:
            Local covariance matrix
        """
        if k_nearest is None:
            k_nearest = max(self.n_dim + 2, self.N // 5)
        k_nearest = min(k_nearest, self.N)

        x_i = self.particles[particle_idx]

        if self.localization_radius is not None:
            # Use particles within radius
            distances = np.linalg.norm(self.particles - x_i, axis=1)
            local_mask = distances < self.localization_radius

            if np.sum(local_mask) > self.n_dim + 1:
                local_particles = self.particles[local_mask]
            else:
                # Fall back to k-nearest
                nearest_indices = np.argsort(distances)[:k_nearest]
                local_particles = self.particles[nearest_indices]
        else:
            # Use k-nearest neighbors
            distances = np.linalg.norm(self.particles - x_i, axis=1)
            nearest_indices = np.argsort(distances)[:k_nearest]
            local_particles = self.particles[nearest_indices]

        # Compute local covariance
        if len(local_particles) > self.n_dim:
            P_local = np.cov(local_particles.T)
            if self.n_dim == 1:
                P_local = np.array([[P_local]])
        else:
            P_global = np.cov(self.particles.T)
            if self.n_dim == 1:
                P_global = np.array([[P_global]])
            P_local = P_global

        # Apply shrinkage for regularization
        shrinkage = 0.1
        P_local = (1 - shrinkage) * P_local + shrinkage * np.eye(self.n_dim) * np.trace(P_local) / self.n_dim

        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(P_local))
        if min_eig < 1e-6:
            P_local += (1e-6 - min_eig) * np.eye(self.n_dim)

        return P_local

    def ledh_flow_function(self, x: np.ndarray, y: np.ndarray, P: np.ndarray) -> np.ndarray:
        """LEDH flow function (same as EDH but with local P)"""
        h_x = self.h_obs(x)
        innovation = y - h_x

        if self.h_jacobian is not None:
            H = self.h_jacobian(x)
        else:
            H = self.numerical_jacobian(self.h_obs, x)

        flow = P @ H.T @ self.R_inv @ innovation
        return flow

    def numerical_jacobian(self, func: Callable, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Numerical Jacobian"""
        f0 = func(x)
        m = len(f0)
        n = len(x)
        J = np.zeros((m, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            J[:, i] = (f_plus - f0) / eps

        return J

    def compute_flow_jacobian(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Compute Jacobian of LEDH flow: A = -P @ H^T @ R^{-1} @ H"""
        if self.h_jacobian is not None:
            H = self.h_jacobian(x)
        else:
            H = self.numerical_jacobian(self.h_obs, x)

        A = -P @ H.T @ self.R_inv @ H
        return A

    def _compute_all_local_covariances(self) -> np.ndarray:
        """
        Vectorized computation of local covariances for all particles.
        Uses efficient pairwise distance computation.
        Returns shape (N, n_dim, n_dim)
        """
        # Compute all pairwise distances at once using cdist
        distances = cdist(self.particles, self.particles, metric='euclidean')

        # For each particle, find k-nearest neighbors
        # Use argpartition for O(N) instead of O(N log N) sorting
        nearest_indices = np.argpartition(distances, self.k_nearest, axis=1)[:, :self.k_nearest]

        P_locals = np.zeros((self.N, self.n_dim, self.n_dim))
        shrinkage = 0.1

        for i in range(self.N):
            local_particles = self.particles[nearest_indices[i]]

            if len(local_particles) > self.n_dim:
                P_local = np.cov(local_particles.T)
                if self.n_dim == 1:
                    P_local = np.array([[P_local]])
            else:
                P_local = np.cov(self.particles.T)
                if self.n_dim == 1:
                    P_local = np.array([[P_local]])

            # Shrinkage regularization
            P_local = (1 - shrinkage) * P_local + shrinkage * np.eye(self.n_dim) * np.trace(P_local) / self.n_dim

            # Ensure positive definiteness
            min_eig = np.min(np.linalg.eigvalsh(P_local))
            if min_eig < 1e-6:
                P_local += (1e-6 - min_eig) * np.eye(self.n_dim)

            P_locals[i] = P_local

        return P_locals

    def update_with_flow(self, y: np.ndarray):
        """Update particles using LEDH flow - VECTORIZED VERSION"""
        d_lambda = 1.0 / (self.n_steps - 1)

        flow_magnitudes = []
        jacobian_conds = []

        for step in range(self.n_steps - 1):
            # Compute all local covariances at once (major speedup)
            P_locals = self._compute_all_local_covariances()  # (N, n_dim, n_dim)

            # Vectorized: compute h(x) for all particles
            h_all = np.array([self.h_obs(self.particles[i]) for i in range(self.N)])

            # Vectorized: compute innovations
            innovations = y - h_all  # (N, obs_dim)

            # Vectorized: compute Jacobians
            if self.h_jacobian is not None:
                H_all = np.array([self.h_jacobian(self.particles[i]) for i in range(self.N)])
            else:
                H_all = np.array([self.numerical_jacobian(self.h_obs, self.particles[i])
                                 for i in range(self.N)])

            # Compute R^{-1} @ innovations using Cholesky
            R_inv_innovations = cho_solve(self.R_chol, innovations.T).T  # (N, obs_dim)

            # Compute flows: flow_i = P_i @ H_i^T @ R^{-1} @ innovation_i
            flows = np.zeros((self.N, self.n_dim))
            for i in range(self.N):
                HT_Rinv_inn = H_all[i].T @ R_inv_innovations[i]  # (n_dim,)
                flows[i] = P_locals[i] @ HT_Rinv_inn

            # Euler integration - vectorized
            self.particles += flows * d_lambda

            # Diagnostics
            flow_magnitudes.append(np.mean(np.linalg.norm(flows, axis=1)))

            # Jacobian conditioning (sample one particle)
            A = -P_locals[0] @ H_all[0].T @ self.R_inv @ H_all[0]
            jacobian_conds.append(np.linalg.cond(np.eye(self.n_dim) + d_lambda * A))

        self.history['flow_magnitude'].append(np.mean(flow_magnitudes))
        self.history['jacobian_cond'].append(np.mean(jacobian_conds))

    def filter_step(self, y: np.ndarray):
        """Single filter step"""
        self.predict()
        self.update_with_flow(y)
        x_mean, x_cov = self.get_estimate()

        self.history['particles'].append(self.particles.copy())
        self.history['x_mean'].append(x_mean.copy())
        self.history['x_cov'].append(x_cov.copy())

        return {'x_mean': x_mean, 'x_cov': x_cov}

    def filter(self, observations: np.ndarray) -> Dict:
        """Run filter"""
        for y in observations:
            self.filter_step(y)
        return self.get_history()

    def get_history(self) -> Dict:
        """Get history"""
        return {
            'particles': self.history['particles'],
            'x_mean': np.array(self.history['x_mean']),
            'x_cov': np.array(self.history['x_cov']),
            'flow_magnitude': np.array(self.history['flow_magnitude']),
            'jacobian_cond': np.array(self.history['jacobian_cond'])
        }


class InvertiblePFPF(ParticleFlowFilter):
    """
    Invertible Particle Flow Particle Filter (PF-PF) [Li 2017]

    Combines particle flow with importance sampling:
    1. Use particle flow to migrate particles (proposal)
    2. Compute importance weights
    3. Resample if needed

    The flow is designed to be invertible, enabling proper importance weight computation.
    """

    def __init__(self, model, N_particles: int = 100, n_steps: int = 50,
                 h_jacobian: Callable = None, flow_type: str = 'ledh'):
        """
        Args:
            model: Nonlinear SSM
            N_particles: Number of particles
            n_steps: Integration steps
            h_jacobian: Jacobian of observation
            flow_type: Type of flow ('edh' or 'ledh')
        """
        super().__init__(model, N_particles, n_steps)
        self.h_jacobian = h_jacobian
        self.R_inv = np.linalg.inv(self.R)
        self.flow_type = flow_type

        # Weights
        self.weights = np.ones(N_particles) / N_particles
        self.log_weights = np.log(self.weights)

    def compute_importance_weight(self, x_after: np.ndarray, y: np.ndarray,
                                  cumulative_log_det: float) -> float:
        """
        Compute importance weight for particle flow

        For importance sampling with deterministic flow:
            w ∝ p(y|x_T) / |det(∂x_T/∂x_0)|
            log(w) = log p(y|x_T) - log|det(∂x_T/∂x_0)|

        Args:
            x_after: Particle after flow
            y: Observation
            cumulative_log_det: Log determinant of the flow map Jacobian

        Returns:
            Log importance weight
        """
        # Likelihood at final position
        h_x = self.h_obs(x_after)
        innovation = y - h_x
        log_likelihood = -0.5 * (
            innovation.T @ self.R_inv @ innovation +
            np.log(np.linalg.det(2 * np.pi * self.R))
        )

        # Importance weight: w ∝ p(y|x) / |det(Jacobian)|
        # log(w) = log_likelihood - cumulative_log_det
        log_weight = log_likelihood - cumulative_log_det

        return log_weight

    def update_with_flow(self, y: np.ndarray):
        """
        Update with flow and compute importance weights

        Following Li(17) Algorithm 1:
        - Compute flow at each λ step
        - Track Jacobian determinant: ∏_{j=1}^{N_λ} |det(I + ε_j A_j(λ))|
        - Update weights with likelihood and Jacobian term
        """
        lambda_vals = np.linspace(0, 1, self.n_steps)
        d_lambda = lambda_vals[1] - lambda_vals[0]

        log_weights_new = []

        for i in range(self.N):
            x_before = self.particles[i].copy()
            x = x_before.copy()

            # Initialize cumulative log determinant (Li 17, Eq 20)
            cumulative_log_det = 0.0

            for lam in lambda_vals[:-1]:
                # Compute covariance (global or local based on flow_type)
                if self.flow_type == 'ledh':
                    P = self.compute_local_covariance(i)
                else:
                    P = np.cov(self.particles.T)
                    if self.n_dim == 1:
                        P = np.array([[P]])

                # Compute flow
                h_x = self.h_obs(x)
                innovation = y - h_x

                if self.h_jacobian is not None:
                    H = self.h_jacobian(x)
                else:
                    H = self.numerical_jacobian(self.h_obs, x)

                flow = P @ H.T @ self.R_inv @ innovation

                # Update particle
                x = x + flow * d_lambda

                # Compute flow Jacobian matrix A = -P @ H^T @ R^{-1} @ H
                # The flow is dx/dλ = P @ H^T @ R^{-1} @ (y - h(x))
                # So ∂(flow)/∂x = -P @ H^T @ R^{-1} @ H (negative because d/dx of -h(x) = -H)
                # (Li 17, Algorithm 1, line 19)
                A = -P @ H.T @ self.R_inv @ H

                # Compute Jacobian determinant: |det(I + ε_j * A_j)|
                # where ε_j = d_lambda (Li 17, Theorem IV.3)
                I_plus_eps_A = np.eye(self.n_dim) + d_lambda * A

                # Accumulate log determinant
                sign, log_det_step = np.linalg.slogdet(I_plus_eps_A)
                if sign > 0:
                    cumulative_log_det += log_det_step
                else:
                    # Defensive: if determinant is non-positive, use small positive value
                    cumulative_log_det += -10.0  # Log of small probability

            x_after = x
            self.particles[i] = x_after

            # Compute importance weight directly using cumulative log determinant
            log_weight = self.compute_importance_weight(x_after, y, cumulative_log_det)
            log_weights_new.append(log_weight)

        # Update weights
        self.log_weights = np.array(log_weights_new)
        self.weights = self.normalize_weights(self.log_weights)

    def compute_local_covariance(self, particle_idx: int, k_nearest: int = None) -> np.ndarray:
        """
        Compute local covariance for LEDH using k-nearest neighbors

        Following Li(17), local covariance maintains particle diversity by
        using nearby particles rather than the global ensemble.

        Args:
            particle_idx: Index of current particle
            k_nearest: Number of nearest neighbors (default: max(n_dim+2, N/5))

        Returns:
            Local covariance matrix
        """
        if k_nearest is None:
            k_nearest = max(self.n_dim + 2, self.N // 5)
        k_nearest = min(k_nearest, self.N)

        x_i = self.particles[particle_idx]

        # Compute distances to all particles
        distances = np.linalg.norm(self.particles - x_i, axis=1)

        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:k_nearest]
        local_particles = self.particles[nearest_indices]

        # Compute local covariance
        if len(local_particles) > self.n_dim:
            P_local = np.cov(local_particles.T)
            if self.n_dim == 1:
                P_local = np.array([[P_local]])
        else:
            # Fall back to global with shrinkage
            P_global = np.cov(self.particles.T)
            if self.n_dim == 1:
                P_global = np.array([[P_global]])
            P_local = P_global

        # Regularize to ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(P_local))
        if min_eig < 1e-6:
            P_local += (1e-6 - min_eig) * np.eye(self.n_dim)

        return P_local

    def numerical_jacobian(self, func: Callable, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Numerical Jacobian"""
        f0 = func(x)
        m = len(f0)
        n = len(x)
        J = np.zeros((m, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = func(x_plus)
            J[:, i] = (f_plus - f0) / eps

        return J

    def normalize_weights(self, log_weights: np.ndarray) -> np.ndarray:
        """Normalize log weights"""
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weights /= np.sum(weights)
        return weights

    def resample(self):
        """Systematic resampling"""
        ess = 1.0 / np.sum(self.weights ** 2)

        if ess < self.N / 2:
            # Resample
            cumsum = np.cumsum(self.weights)
            u0 = np.random.uniform(0, 1.0 / self.N)
            u = u0 + np.arange(self.N) / self.N
            indices = np.searchsorted(cumsum, u)

            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N
            self.log_weights = np.log(self.weights)

            return True

        return False

    def filter_step(self, y: np.ndarray):
        """Single filter step"""
        self.predict()
        self.update_with_flow(y)

        # Get weighted estimate
        x_mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        deviations = self.particles - x_mean
        x_cov = np.sum(self.weights[:, np.newaxis, np.newaxis] *
                      (deviations[:, :, np.newaxis] @ deviations[:, np.newaxis, :]), axis=0)

        resampled = self.resample()

        self.history['particles'].append(self.particles.copy())
        self.history['x_mean'].append(x_mean.copy())
        self.history['x_cov'].append(x_cov.copy())

        return {'x_mean': x_mean, 'x_cov': x_cov, 'resampled': resampled}

    def filter(self, observations: np.ndarray) -> Dict:
        """Run filter"""
        for y in observations:
            self.filter_step(y)
        return self.get_history()

    def get_history(self) -> Dict:
        """Get history"""
        return {
            'particles': self.history['particles'],
            'x_mean': np.array(self.history['x_mean']),
            'x_cov': np.array(self.history['x_cov'])
        }
