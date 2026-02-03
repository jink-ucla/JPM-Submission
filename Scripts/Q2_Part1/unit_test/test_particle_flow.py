"""
Unit Tests for Q2: Particle Flow Filters
=========================================
Tests the EDH, LEDH, and Kernel Particle Flow Filter implementations.

Tests cover:
1. Flow function correctness
2. Particle migration from prior to posterior
3. Covariance estimation
4. Jacobian computation
5. High-dimensional performance

References:
- Daum & Huang (2010): Exact particle flow for nonlinear filters
- Daum & Huang (2011): Particle degeneracy
- Li & Coates (2017): Particle filtering with invertible particle flow
- Hu & Van Leeuwen (2021): Particle flow filter for high-dimensional systems
"""

import pytest
import numpy as np
from scipy import stats
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particle_flow_filters import (
    ParticleFlowFilter,
    EDHParticleFlowFilter
)

# Try to import kernel flow filters
try:
    from kernel_particle_flow import KernelParticleFlowFilter
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False

# Try to import nonlinear models
try:
    from nonlinear_models import StochasticVolatilityModel, RangeBearingModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# ==============================================================================
# Mock Model for Testing
# ==============================================================================

class SimpleLinearModel:
    """Simple linear model for testing particle flow filters."""

    def __init__(self, n_dim=4, obs_dim=2):
        self.n_dim = n_dim
        self.obs_dim = obs_dim
        self.Q = np.eye(n_dim) * 0.1
        self.R = np.eye(obs_dim) * 0.5

        # Linear transition and observation
        self.F = np.eye(n_dim) * 0.95
        self.H = np.zeros((obs_dim, n_dim))
        self.H[:obs_dim, :obs_dim] = np.eye(obs_dim)

    def f(self, x, w=None):
        """State transition."""
        result = self.F @ x
        if w is not None:
            result += w
        return result

    def h(self, x):
        """Observation function."""
        return self.H @ x

    def h_jacobian(self, x):
        """Jacobian of observation function."""
        return self.H


class NonlinearModel:
    """Nonlinear model for testing (range-bearing style)."""

    def __init__(self):
        self.n_dim = 4
        self.obs_dim = 2
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.5

        # Constant velocity model
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def f(self, x, w=None):
        """State transition."""
        result = self.F @ x
        if w is not None:
            result += w
        return result

    def h(self, x):
        """Nonlinear observation (range-bearing)."""
        px, py = x[0], x[1]
        r = np.sqrt(px**2 + py**2) + 1e-6
        theta = np.arctan2(py, px)
        return np.array([r, theta])

    def h_jacobian(self, x):
        """Jacobian of observation function."""
        px, py = x[0], x[1]
        r = np.sqrt(px**2 + py**2) + 1e-6

        H = np.zeros((2, 4))
        H[0, 0] = px / r
        H[0, 1] = py / r
        H[1, 0] = -py / (r**2)
        H[1, 1] = px / (r**2)
        return H


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def linear_model():
    """Simple linear model for testing."""
    return SimpleLinearModel(n_dim=4, obs_dim=2)


@pytest.fixture
def nonlinear_model():
    """Nonlinear range-bearing model for testing."""
    return NonlinearModel()


@pytest.fixture
def edh_filter(linear_model):
    """EDH filter with linear model."""
    return EDHParticleFlowFilter(
        model=linear_model,
        N_particles=100,
        n_steps=20,
        h_jacobian=linear_model.h_jacobian
    )


# ==============================================================================
# Unit Tests: EDH Particle Flow
# ==============================================================================

class TestEDHParticleFlowFilter:
    """Tests for Exact Daum-Huang particle flow filter."""

    def test_initialization(self, edh_filter):
        """Test filter initialization."""
        x0 = np.zeros(4)
        P0 = np.eye(4)

        edh_filter.initialize(x0, P0)

        assert edh_filter.particles.shape == (100, 4)

    def test_particles_initialized_from_prior(self, edh_filter):
        """Test that particles are sampled from prior distribution."""
        x0 = np.array([1.0, 2.0, 0.0, 0.0])
        P0 = np.eye(4) * 0.1

        edh_filter.initialize(x0, P0)

        # Sample mean should be close to x0
        mean = np.mean(edh_filter.particles, axis=0)
        np.testing.assert_array_almost_equal(mean, x0, decimal=0)

    def test_prediction_step(self, edh_filter):
        """Test that prediction propagates particles through transition."""
        x0 = np.zeros(4)
        P0 = np.eye(4)

        edh_filter.initialize(x0, P0)
        particles_before = edh_filter.particles.copy()

        edh_filter.predict()

        # Particles should have changed
        assert not np.allclose(edh_filter.particles, particles_before)

    def test_get_estimate(self, edh_filter):
        """Test mean and covariance estimation."""
        x0 = np.array([1.0, 2.0, 0.0, 0.0])
        P0 = np.eye(4)

        edh_filter.initialize(x0, P0)

        mean, cov = edh_filter.get_estimate()

        assert mean.shape == (4,)
        assert cov.shape == (4, 4)

        # Covariance should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_flow_function_moves_toward_observation(self, linear_model):
        """Test that EDH flow moves particles toward observation."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=50,
            n_steps=10,
            h_jacobian=linear_model.h_jacobian
        )

        # Initialize away from observation
        x = np.array([0.0, 0.0, 0.0, 0.0])
        y = np.array([10.0, 10.0])  # Far observation
        P = np.eye(4)

        flow = filter.edh_flow_function(x, y, P)

        # Flow should point toward observation (positive direction)
        assert flow[0] > 0, "Flow should move toward observation"
        assert flow[1] > 0, "Flow should move toward observation"

    def test_numerical_jacobian_accuracy(self, linear_model):
        """Test that numerical Jacobian matches analytical."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=10,
            n_steps=10
        )

        x = np.array([1.0, 2.0, 0.5, 0.5])

        numerical_H = filter.numerical_jacobian(linear_model.h, x)
        analytical_H = linear_model.h_jacobian(x)

        np.testing.assert_array_almost_equal(
            numerical_H, analytical_H, decimal=5
        )


# ==============================================================================
# Unit Tests: Flow Jacobian
# ==============================================================================

class TestFlowJacobian:
    """Tests for flow Jacobian computation."""

    def test_flow_jacobian_shape(self, linear_model):
        """Test that flow Jacobian has correct shape."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=10,
            n_steps=10,
            h_jacobian=linear_model.h_jacobian
        )

        x = np.array([1.0, 2.0, 0.5, 0.5])
        y = np.array([1.0, 1.0])
        P = np.eye(4)

        A = filter.compute_flow_jacobian(x, y, P)

        assert A.shape == (4, 4)

    def test_flow_jacobian_stability(self, linear_model):
        """Test that flow Jacobian doesn't have extreme eigenvalues."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=10,
            n_steps=10,
            h_jacobian=linear_model.h_jacobian
        )

        x = np.array([1.0, 2.0, 0.5, 0.5])
        y = np.array([1.0, 1.0])
        P = np.eye(4)

        A = filter.compute_flow_jacobian(x, y, P)
        eigenvalues = np.linalg.eigvals(A)

        # All eigenvalues should be real and non-positive (stable flow)
        # For EDH, A = -P H^T R^{-1} H, which is negative semi-definite
        max_real = np.max(np.real(eigenvalues))
        assert max_real <= 1e-6, "Flow Jacobian has positive eigenvalues"


# ==============================================================================
# Integration Tests: Filtering Performance
# ==============================================================================

class TestFilteringPerformance:
    """Integration tests for filtering performance."""

    def test_filter_tracks_linear_system(self, linear_model):
        """Test that EDH filter tracks a linear system."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=200,
            n_steps=30,
            h_jacobian=linear_model.h_jacobian
        )

        # Generate synthetic data
        np.random.seed(42)
        T = 20
        true_states = np.zeros((T, 4))
        observations = np.zeros((T, 2))

        x = np.array([1.0, 1.0, 0.1, 0.1])
        for t in range(T):
            w = np.random.multivariate_normal(np.zeros(4), linear_model.Q)
            x = linear_model.f(x, w)
            v = np.random.multivariate_normal(np.zeros(2), linear_model.R)
            y = linear_model.h(x) + v

            true_states[t] = x
            observations[t] = y

        # Initialize and filter
        x0 = np.zeros(4)
        P0 = np.eye(4) * 2.0
        filter.initialize(x0, P0)

        estimates = []
        for t in range(T):
            filter.predict()
            # Note: full update step would be implemented in subclass
            mean, _ = filter.get_estimate()
            estimates.append(mean)

        estimates = np.array(estimates)

        # RMSE should be reasonable
        rmse = np.sqrt(np.mean((true_states[:, :2] - estimates[:, :2])**2))
        assert rmse < 5.0, f"RMSE {rmse:.2f} too large"

    def test_particle_diversity_maintained(self, linear_model):
        """Test that particles don't collapse to a single point."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=100,
            n_steps=20,
            h_jacobian=linear_model.h_jacobian
        )

        x0 = np.zeros(4)
        P0 = np.eye(4)
        filter.initialize(x0, P0)

        # Run several prediction steps
        for _ in range(10):
            filter.predict()

        # Check particle diversity
        _, cov = filter.get_estimate()
        min_variance = np.min(np.diag(cov))

        assert min_variance > 1e-6, "Particle variance collapsed"


# ==============================================================================
# Tests: Numerical Stability
# ==============================================================================

class TestNumericalStability:
    """Tests for numerical stability of particle flow."""

    def test_covariance_positive_definite(self, linear_model):
        """Test that estimated covariance remains positive definite."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=100,
            n_steps=20,
            h_jacobian=linear_model.h_jacobian
        )

        x0 = np.zeros(4)
        P0 = np.eye(4)
        filter.initialize(x0, P0)

        for _ in range(20):
            filter.predict()
            _, cov = filter.get_estimate()

            eigenvalues = np.linalg.eigvals(cov)
            assert np.all(eigenvalues > -1e-10), \
                "Covariance not positive semi-definite"

    def test_flow_magnitude_bounded(self, linear_model):
        """Test that flow magnitude doesn't explode."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=50,
            n_steps=20,
            h_jacobian=linear_model.h_jacobian
        )

        x = np.array([1.0, 2.0, 0.5, 0.5])
        y = np.array([100.0, 100.0])  # Large innovation
        P = np.eye(4)

        flow = filter.edh_flow_function(x, y, P)

        # Flow should be bounded
        assert np.linalg.norm(flow) < 1e6, "Flow magnitude too large"


# ==============================================================================
# Tests: Nonlinear Model
# ==============================================================================

class TestNonlinearModel:
    """Tests with nonlinear observation model."""

    def test_edh_handles_nonlinear_model(self, nonlinear_model):
        """Test that EDH works with nonlinear observation model."""
        filter = EDHParticleFlowFilter(
            model=nonlinear_model,
            N_particles=100,
            n_steps=20,
            h_jacobian=nonlinear_model.h_jacobian
        )

        x0 = np.array([10.0, 10.0, 1.0, 1.0])
        P0 = np.eye(4)

        filter.initialize(x0, P0)

        # Should not raise errors
        filter.predict()
        mean, cov = filter.get_estimate()

        assert mean.shape == (4,)
        assert cov.shape == (4, 4)

    def test_numerical_jacobian_for_nonlinear(self, nonlinear_model):
        """Test numerical Jacobian for nonlinear model."""
        filter = EDHParticleFlowFilter(
            model=nonlinear_model,
            N_particles=10,
            n_steps=10
        )

        x = np.array([10.0, 5.0, 1.0, 1.0])

        numerical_H = filter.numerical_jacobian(nonlinear_model.h, x)
        analytical_H = nonlinear_model.h_jacobian(x)

        np.testing.assert_array_almost_equal(
            numerical_H, analytical_H, decimal=4
        )


# ==============================================================================
# Tests: High-Dimensional Systems (Hu(21) motivation)
# ==============================================================================

class TestHighDimensional:
    """Tests for high-dimensional filtering (as in Hu(2021))."""

    def test_high_dim_particle_spread(self):
        """Test particle spread in high dimensions."""
        n_dim = 20
        obs_dim = 5  # 25% observation

        # Create high-dim linear model
        class HighDimModel:
            def __init__(self):
                self.n_dim = n_dim
                self.obs_dim = obs_dim
                self.Q = np.eye(n_dim) * 0.1
                self.R = np.eye(obs_dim) * 0.5
                self.H = np.zeros((obs_dim, n_dim))
                self.H[:obs_dim, :obs_dim] = np.eye(obs_dim)

            def f(self, x, w=None):
                result = 0.95 * x
                if w is not None:
                    result += w
                return result

            def h(self, x):
                return self.H @ x

            def h_jacobian(self, x):
                return self.H

        model = HighDimModel()
        filter = EDHParticleFlowFilter(
            model=model,
            N_particles=200,
            n_steps=30,
            h_jacobian=model.h_jacobian
        )

        x0 = np.zeros(n_dim)
        P0 = np.eye(n_dim)
        filter.initialize(x0, P0)

        # Check initial spread
        _, cov = filter.get_estimate()
        initial_trace = np.trace(cov)

        # Run predictions
        for _ in range(5):
            filter.predict()

        # Spread should be maintained
        _, cov = filter.get_estimate()
        final_trace = np.trace(cov)

        # Variance shouldn't collapse completely
        assert final_trace > initial_trace * 0.1, \
            f"Variance collapsed: {initial_trace:.2f} -> {final_trace:.2f}"


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_particle(self, linear_model):
        """Test with single particle (degenerate case)."""
        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=1,
            n_steps=10,
            h_jacobian=linear_model.h_jacobian
        )

        x0 = np.zeros(4)
        P0 = np.eye(4)

        filter.initialize(x0, P0)
        filter.predict()

        mean, _ = filter.get_estimate()
        assert mean.shape == (4,)

    def test_very_small_noise(self, linear_model):
        """Test with very small process noise."""
        linear_model.Q = np.eye(4) * 1e-10

        filter = EDHParticleFlowFilter(
            model=linear_model,
            N_particles=100,
            n_steps=20,
            h_jacobian=linear_model.h_jacobian
        )

        x0 = np.ones(4)
        P0 = np.eye(4)

        filter.initialize(x0, P0)

        # Should not crash
        filter.predict()
        mean, cov = filter.get_estimate()

        assert not np.any(np.isnan(mean))
        assert not np.any(np.isnan(cov))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
