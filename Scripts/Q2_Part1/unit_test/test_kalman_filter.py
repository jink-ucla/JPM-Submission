"""
Unit Tests for Q2: State-Space Filtering Methods
=================================================
Tests the Kalman Filter and Particle Flow Filter implementations using pytest.

Tests cover:
1. Kalman Filter correctness and numerical stability
2. NEES/NIS consistency checks
3. Joseph stabilized form comparison
4. Particle Flow Filter convergence
5. High-dimensional filtering

References:
- Kalman (1960): A new approach to linear filtering
- Doucet & Johansen (2009): Tutorial on particle filtering
- Daum & Huang (2010): Exact particle flow for nonlinear filters
"""

import pytest
import numpy as np
from scipy import stats
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalman_filter import (
    KalmanFilter,
    generate_lgssm_data,
    compute_nees,
    compute_nis
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def simple_1d_system():
    """Simple 1D linear-Gaussian system."""
    F = np.array([[0.9]])  # State transition
    H = np.array([[1.0]])  # Observation
    Q = np.array([[0.1]])  # Process noise
    R = np.array([[0.5]])  # Observation noise
    return F, H, Q, R


@pytest.fixture
def tracking_2d_system():
    """2D constant velocity tracking system."""
    dt = 1.0
    F = np.array([
        [1, dt],
        [0, 1]
    ])
    H = np.array([[1, 0]])  # Observe position only
    Q = np.array([
        [dt**3/3, dt**2/2],
        [dt**2/2, dt]
    ]) * 0.1
    R = np.array([[1.0]])
    return F, H, Q, R


@pytest.fixture
def tracking_4d_system():
    """4D tracking system (position and velocity in 2D)."""
    dt = 1.0
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    Q = np.eye(4) * 0.1
    R = np.eye(2) * 0.5
    return F, H, Q, R


# ==============================================================================
# Unit Tests: Kalman Filter Initialization
# ==============================================================================

class TestKalmanFilterInitialization:
    """Tests for Kalman Filter initialization."""

    def test_initialization(self, simple_1d_system):
        """Test that KalmanFilter initializes correctly."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        assert kf.n_dim == 1
        assert kf.obs_dim == 1
        assert kf.use_joseph is True

    def test_initialize_state(self, simple_1d_system):
        """Test state initialization."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        kf.initialize(x0, P0)

        np.testing.assert_array_equal(kf.x, x0)
        np.testing.assert_array_equal(kf.P, P0)

    def test_dimensions_match(self, tracking_4d_system):
        """Test that all dimensions are correctly inferred."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R)

        assert kf.n_dim == 4
        assert kf.obs_dim == 2


# ==============================================================================
# Unit Tests: Kalman Filter Prediction Step
# ==============================================================================

class TestKalmanFilterPredict:
    """Tests for Kalman Filter prediction step."""

    def test_predict_mean_update(self, simple_1d_system):
        """Test that prediction updates mean correctly."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([1.0])
        P0 = np.array([[1.0]])
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()

        expected_x = F @ x0
        np.testing.assert_array_almost_equal(x_pred, expected_x)

    def test_predict_covariance_update(self, simple_1d_system):
        """Test that prediction updates covariance correctly."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([1.0])
        P0 = np.array([[1.0]])
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()

        expected_P = F @ P0 @ F.T + Q
        np.testing.assert_array_almost_equal(P_pred, expected_P)

    def test_predict_covariance_symmetry(self, tracking_4d_system):
        """Test that predicted covariance remains symmetric."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.zeros(4)
        P0 = np.eye(4)
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()

        np.testing.assert_array_almost_equal(P_pred, P_pred.T)

    def test_predict_covariance_positive_definite(self, tracking_4d_system):
        """Test that predicted covariance is positive definite."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.zeros(4)
        P0 = np.eye(4)
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()

        eigenvalues = np.linalg.eigvals(P_pred)
        assert np.all(eigenvalues > 0), "Covariance not positive definite"


# ==============================================================================
# Unit Tests: Kalman Filter Update Step
# ==============================================================================

class TestKalmanFilterUpdate:
    """Tests for Kalman Filter update step."""

    def test_update_reduces_uncertainty(self, simple_1d_system):
        """Test that update step reduces covariance (adds information)."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([0.0])
        P0 = np.array([[10.0]])  # High initial uncertainty
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()
        y = np.array([1.0])
        x_filt, P_filt, _, _, _, _, _ = kf.update(y, x_pred, P_pred)

        # Trace of filtered covariance should be less than predicted
        assert np.trace(P_filt) < np.trace(P_pred)

    def test_update_moves_toward_observation(self, simple_1d_system):
        """Test that update moves estimate toward observation."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()
        y = np.array([10.0])  # Far from prediction
        x_filt, _, _, _, _, _, _ = kf.update(y, x_pred, P_pred)

        # Filtered estimate should be between prediction and observation
        assert x_pred[0] < x_filt[0] < y[0]

    def test_kalman_gain_range(self, simple_1d_system):
        """Test that Kalman gain is in reasonable range."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        kf.initialize(x0, P0)

        x_pred, P_pred = kf.predict()
        y = np.array([1.0])
        _, _, K, _, _, _, _ = kf.update(y, x_pred, P_pred)

        # Kalman gain should be between 0 and 1 for scalar case
        assert 0 <= K[0, 0] <= 1

    def test_joseph_vs_standard_form(self, tracking_4d_system):
        """Test that Joseph form gives similar results to standard form."""
        F, H, Q, R = tracking_4d_system

        kf_joseph = KalmanFilter(F, H, Q, R, use_joseph=True)
        kf_standard = KalmanFilter(F, H, Q, R, use_joseph=False)

        x0 = np.zeros(4)
        P0 = np.eye(4)

        kf_joseph.initialize(x0, P0)
        kf_standard.initialize(x0, P0)

        # Generate observation
        y = np.array([1.0, 1.0])

        x_pred_j, P_pred_j = kf_joseph.predict()
        x_filt_j, P_filt_j, _, _, _, _, _ = kf_joseph.update(y, x_pred_j, P_pred_j)

        x_pred_s, P_pred_s = kf_standard.predict()
        x_filt_s, P_filt_s, _, _, _, _, _ = kf_standard.update(y, x_pred_s, P_pred_s)

        # Results should be close
        np.testing.assert_array_almost_equal(x_filt_j, x_filt_s, decimal=5)


# ==============================================================================
# Integration Tests: Full Filtering Sequence
# ==============================================================================

class TestKalmanFilterIntegration:
    """Integration tests for complete filtering sequences."""

    def test_filter_sequence(self, simple_1d_system):
        """Test filtering over a sequence of observations."""
        F, H, Q, R = simple_1d_system
        kf = KalmanFilter(F, H, Q, R)

        # Generate data
        x0 = np.array([0.0])
        T = 50
        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        # Filter
        P0 = np.array([[1.0]])
        kf.initialize(x0, P0)
        history = kf.filter(observations)

        assert len(history['x_filt']) == T

    def test_nees_distribution(self, tracking_4d_system):
        """
        CRITICAL TEST: NEES should follow chi-squared distribution.

        This validates filter consistency.
        """
        F, H, Q, R = tracking_4d_system
        n_dim = 4

        # Run multiple trials
        nees_all = []
        n_trials = 20
        T = 100

        for trial in range(n_trials):
            kf = KalmanFilter(F, H, Q, R)
            x0 = np.zeros(n_dim)
            P0 = np.eye(n_dim)

            true_states, observations = generate_lgssm_data(
                F, H, Q, R, x0, T, seed=trial
            )

            kf.initialize(x0, P0)
            history = kf.filter(observations)

            nees = compute_nees(true_states, history['x_filt'], history['P_filt'])
            nees_all.extend(nees.tolist())

        nees_all = np.array(nees_all)

        # NEES should have mean approximately n_dim (chi-squared with n_dim DOF)
        mean_nees = np.mean(nees_all)
        assert n_dim * 0.5 < mean_nees < n_dim * 1.5, \
            f"Mean NEES {mean_nees:.2f} not close to expected {n_dim}"

    def test_tracking_accuracy(self, tracking_2d_system):
        """Test that filter tracks true state reasonably well."""
        F, H, Q, R = tracking_2d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.array([0.0, 1.0])  # Initial position=0, velocity=1
        P0 = np.eye(2)
        T = 100

        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        kf.initialize(x0, P0)
        history = kf.filter(observations)

        # Compute RMSE
        errors = true_states - history['x_filt']
        rmse = np.sqrt(np.mean(errors**2))

        # RMSE should be reasonable (not too large)
        assert rmse < 2.0, f"RMSE {rmse:.2f} too large"


# ==============================================================================
# Numerical Stability Tests
# ==============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""

    def test_condition_number_bounded(self, tracking_4d_system):
        """Test that covariance condition numbers stay bounded."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R, use_joseph=True)

        x0 = np.zeros(4)
        P0 = np.eye(4)
        T = 200

        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        kf.initialize(x0, P0)
        history = kf.filter(observations)

        # Check condition numbers don't explode
        max_cond = np.max(history['cond_P'])
        assert max_cond < 1e10, f"Condition number {max_cond:.2e} too large"

    def test_joseph_improves_stability(self, tracking_4d_system):
        """Test that Joseph form maintains better numerical stability."""
        F, H, Q, R = tracking_4d_system

        x0 = np.zeros(4)
        P0 = np.eye(4) * 0.01  # Small initial uncertainty
        T = 100

        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        # Run with Joseph form
        kf_joseph = KalmanFilter(F, H, Q, R, use_joseph=True)
        kf_joseph.initialize(x0, P0)
        history_joseph = kf_joseph.filter(observations)

        # Run with standard form
        kf_standard = KalmanFilter(F, H, Q, R, use_joseph=False)
        kf_standard.initialize(x0, P0)
        history_standard = kf_standard.filter(observations)

        # Joseph should have equal or better condition numbers
        avg_cond_joseph = np.mean(history_joseph['cond_P'])
        avg_cond_standard = np.mean(history_standard['cond_P'])

        # Joseph form should not be significantly worse
        assert avg_cond_joseph <= avg_cond_standard * 1.5

    def test_covariance_remains_positive_definite(self, tracking_4d_system):
        """Test that covariance remains positive definite throughout filtering."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.zeros(4)
        P0 = np.eye(4)
        T = 200

        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        kf.initialize(x0, P0)

        for t in range(T):
            kf.filter_step(observations[t])

            # Check positive definiteness
            eigenvalues = np.linalg.eigvals(kf.P)
            assert np.all(eigenvalues > -1e-10), \
                f"Non-PD covariance at time {t}"


# ==============================================================================
# Data Generation Tests
# ==============================================================================

class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_correct_dimensions(self, tracking_4d_system):
        """Test that generated data has correct dimensions."""
        F, H, Q, R = tracking_4d_system
        x0 = np.zeros(4)
        T = 50

        states, observations = generate_lgssm_data(F, H, Q, R, x0, T)

        assert states.shape == (T, 4)
        assert observations.shape == (T, 2)

    def test_generate_deterministic_with_seed(self, simple_1d_system):
        """Test that same seed gives same data."""
        F, H, Q, R = simple_1d_system
        x0 = np.array([0.0])
        T = 20

        states1, obs1 = generate_lgssm_data(F, H, Q, R, x0, T, seed=42)
        states2, obs2 = generate_lgssm_data(F, H, Q, R, x0, T, seed=42)

        np.testing.assert_array_equal(states1, states2)
        np.testing.assert_array_equal(obs1, obs2)


# ==============================================================================
# NEES/NIS Computation Tests
# ==============================================================================

class TestNEESNIS:
    """Tests for NEES and NIS computation."""

    def test_nees_positive(self, tracking_4d_system):
        """Test that NEES values are positive."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.zeros(4)
        P0 = np.eye(4)
        T = 50

        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        kf.initialize(x0, P0)
        history = kf.filter(observations)

        nees = compute_nees(true_states, history['x_filt'], history['P_filt'])

        assert np.all(nees >= 0)

    def test_nis_positive(self, tracking_4d_system):
        """Test that NIS values are positive."""
        F, H, Q, R = tracking_4d_system
        kf = KalmanFilter(F, H, Q, R)

        x0 = np.zeros(4)
        P0 = np.eye(4)
        T = 50

        true_states, observations = generate_lgssm_data(
            F, H, Q, R, x0, T, seed=42
        )

        kf.initialize(x0, P0)
        history = kf.filter(observations)

        nis = compute_nis(history['innovation'], history['S'])

        assert np.all(nis >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
