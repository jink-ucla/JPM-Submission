"""
Particle Filter (Bootstrap Filter / Sequential Importance Resampling)
Part 1 (II): Nonlinear/Non-Gaussian SSM

Implements:
- Standard Bootstrap Particle Filter
- Multiple resampling schemes (multinomial, systematic, stratified)
- Particle degeneracy diagnostics (ESS)
- Comparison with EKF/UKF
"""

import numpy as np
from typing import Tuple, Dict, Callable
from nonlinear_models import NonlinearSSM


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute Effective Sample Size (ESS)

    ESS = 1 / sum(w_i^2)

    ESS measures particle diversity. ESS close to N indicates good diversity,
    while ESS close to 1 indicates degeneracy (all weight on one particle)
    """
    return 1.0 / np.sum(weights ** 2)


def normalize_weights(log_weights: np.ndarray) -> np.ndarray:
    """
    Normalize log weights using log-sum-exp trick for numerical stability

    Args:
        log_weights: Log weights

    Returns:
        weights: Normalized weights
    """
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights /= np.sum(weights)
    return weights


def multinomial_resample(weights: np.ndarray, N: int = None) -> np.ndarray:
    """
    Multinomial resampling

    Sample N particles from categorical distribution defined by weights

    Args:
        weights: Particle weights (N,)
        N: Number of particles to resample (default: len(weights))

    Returns:
        indices: Resampled particle indices
    """
    if N is None:
        N = len(weights)
    return np.random.choice(N, size=N, replace=True, p=weights)


def systematic_resample(weights: np.ndarray, N: int = None) -> np.ndarray:
    """
    Systematic resampling (lower variance than multinomial)

    Args:
        weights: Particle weights (N,)
        N: Number of particles to resample (default: len(weights))

    Returns:
        indices: Resampled particle indices
    """
    if N is None:
        N = len(weights)

    # Compute cumulative sum
    cumsum = np.cumsum(weights)

    # Generate systematic samples
    u0 = np.random.uniform(0, 1.0 / N)
    u = u0 + np.arange(N) / N

    indices = np.searchsorted(cumsum, u)

    return indices


def stratified_resample(weights: np.ndarray, N: int = None) -> np.ndarray:
    """
    Stratified resampling

    Args:
        weights: Particle weights (N,)
        N: Number of particles to resample (default: len(weights))

    Returns:
        indices: Resampled particle indices
    """
    if N is None:
        N = len(weights)

    # Compute cumulative sum
    cumsum = np.cumsum(weights)

    # Generate stratified samples
    u = (np.arange(N) + np.random.uniform(0, 1, N)) / N

    indices = np.searchsorted(cumsum, u)

    return indices


class ParticleFilter:
    """
    Bootstrap Particle Filter

    Uses state transition as proposal: q(x_t | x_{t-1}) = p(x_t | x_{t-1})
    Weights proportional to likelihood: w_t ∝ p(y_t | x_t)
    """

    def __init__(self, model: NonlinearSSM, N_particles: int = 100,
                 resample_scheme: str = 'systematic', ess_threshold: float = 0.5):
        """
        Initialize Particle Filter

        Args:
            model: Nonlinear SSM
            N_particles: Number of particles
            resample_scheme: Resampling scheme ('multinomial', 'systematic', 'stratified')
            ess_threshold: ESS threshold for resampling (as fraction of N)
        """
        self.model = model
        self.f = model.f
        self.h = model.h
        self.Q = model.Q
        self.R = model.R

        self.n_dim = model.n_dim
        self.obs_dim = model.obs_dim

        self.N = N_particles
        self.ess_threshold = ess_threshold * N_particles

        # Resampling scheme
        if resample_scheme == 'multinomial':
            self.resample = multinomial_resample
        elif resample_scheme == 'systematic':
            self.resample = systematic_resample
        elif resample_scheme == 'stratified':
            self.resample = stratified_resample
        else:
            raise ValueError(f"Unknown resampling scheme: {resample_scheme}")

        # Cholesky decompositions for sampling
        self.L_Q = np.linalg.cholesky(self.Q)
        self.L_R = np.linalg.cholesky(self.R)
        self.R_inv = np.linalg.inv(self.R)

        # Particles and weights
        self.particles = None
        self.weights = None
        self.log_weights = None

        # History
        self.history = {
            'particles': [],
            'weights': [],
            'ess': [],
            'x_mean': [],
            'x_cov': [],
            'resampled': [],
            'log_likelihood': []
        }

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """
        Initialize particles

        Args:
            x0: Initial mean
            P0: Initial covariance
        """
        # Sample particles from initial distribution
        L_P0 = np.linalg.cholesky(P0)
        self.particles = x0[:, None] + L_P0 @ np.random.randn(self.n_dim, self.N)
        self.particles = self.particles.T  # (N, n_dim)

        # Uniform weights
        self.weights = np.ones(self.N) / self.N
        self.log_weights = np.log(self.weights)

    def predict(self):
        """
        Prediction step: propagate particles through state transition

        x_t^(i) ~ p(x_t | x_{t-1}^(i))
        """
        # Sample process noise
        noise = self.L_Q @ np.random.randn(self.n_dim, self.N)
        noise = noise.T  # (N, n_dim)

        # Propagate particles
        new_particles = np.zeros_like(self.particles)
        for i in range(self.N):
            new_particles[i] = self.f(self.particles[i], noise[i])

        self.particles = new_particles

    def update(self, y: np.ndarray):
        """
        Update step: compute weights based on observation likelihood

        w_t^(i) ∝ p(y_t | x_t^(i))
        """
        # Compute log-likelihood for each particle
        log_likelihoods = np.zeros(self.N)

        for i in range(self.N):
            # Predicted observation
            y_pred = self.h(self.particles[i])

            # Innovation
            innovation = y - y_pred

            # Log-likelihood (Gaussian)
            log_likelihoods[i] = -0.5 * (
                innovation.T @ self.R_inv @ innovation +
                np.log(np.linalg.det(2 * np.pi * self.R))
            )

        # Update log weights
        self.log_weights += log_likelihoods

        # Normalize weights
        self.weights = normalize_weights(self.log_weights)

        # Compute marginal log-likelihood (for model comparison)
        log_likelihood = np.log(np.mean(np.exp(log_likelihoods)))

        return log_likelihood

    def resample_particles(self):
        """Resample particles if ESS is below threshold"""
        ess = effective_sample_size(self.weights)

        if ess < self.ess_threshold:
            # Resample
            indices = self.resample(self.weights, self.N)
            self.particles = self.particles[indices]

            # Reset weights
            self.weights = np.ones(self.N) / self.N
            self.log_weights = np.log(self.weights)

            return True, ess

        return False, ess

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weighted mean and covariance estimate

        Returns:
            x_mean: Weighted mean
            x_cov: Weighted covariance
        """
        # Weighted mean
        x_mean = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)

        # Weighted covariance
        deviations = self.particles - x_mean
        x_cov = np.sum(self.weights[:, np.newaxis, np.newaxis] *
                      (deviations[:, :, np.newaxis] @ deviations[:, np.newaxis, :]), axis=0)

        return x_mean, x_cov

    def filter_step(self, y: np.ndarray) -> Dict:
        """Single filter step"""
        # Predict
        self.predict()

        # Update
        log_likelihood = self.update(y)

        # Get estimate before resampling
        x_mean, x_cov = self.get_estimate()

        # Resample
        resampled, ess = self.resample_particles()

        # Store history
        self.history['particles'].append(self.particles.copy())
        self.history['weights'].append(self.weights.copy())
        self.history['ess'].append(ess)
        self.history['x_mean'].append(x_mean.copy())
        self.history['x_cov'].append(x_cov.copy())
        self.history['resampled'].append(resampled)
        self.history['log_likelihood'].append(log_likelihood)

        return {
            'x_mean': x_mean,
            'x_cov': x_cov,
            'ess': ess,
            'resampled': resampled,
            'log_likelihood': log_likelihood
        }

    def filter(self, observations: np.ndarray) -> Dict:
        """Run particle filter on sequence"""
        for y in observations:
            self.filter_step(y)
        return self.get_history()

    def get_history(self) -> Dict:
        """Get history"""
        return {
            'particles': self.history['particles'],
            'weights': self.history['weights'],
            'ess': np.array(self.history['ess']),
            'x_mean': np.array(self.history['x_mean']),
            'x_cov': np.array(self.history['x_cov']),
            'resampled': np.array(self.history['resampled']),
            'log_likelihood': np.array(self.history['log_likelihood'])
        }


def compute_rmse(true_states: np.ndarray, estimates: np.ndarray) -> float:
    """Compute RMSE"""
    return np.sqrt(np.mean(np.sum((true_states - estimates) ** 2, axis=1)))


def compute_log_likelihood_total(log_likelihoods: np.ndarray) -> float:
    """Compute total log-likelihood"""
    return np.sum(log_likelihoods)
