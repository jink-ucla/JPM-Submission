"""
Nonlinear State Space Models for filtering
Part 1 (II): Nonlinear/Non-Gaussian SSM

Implements:
1. Stochastic Volatility Model (Example 4 from Doucet 09)
2. Range-Bearing Tracking Model

Both models are nonlinear and non-Gaussian, motivating the need
for nonlinear filters (EKF, UKF, PF, etc.)
"""

import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass


@dataclass
class NonlinearSSM:
    """Base class for nonlinear state space models"""
    f: Callable  # State transition function
    h: Callable  # Observation function
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Observation noise covariance
    n_dim: int  # State dimension
    obs_dim: int  # Observation dimension
    x0: np.ndarray  # Initial state
    P0: np.ndarray  # Initial covariance


class StochasticVolatilityModel:
    """
    Stochastic Volatility Model (Example 4 from Doucet 09)

    State equation:
        x_t = φ * x_{t-1} + σ_w * w_t,  w_t ~ N(0, 1)

    Observation equation:
        y_t = β * exp(x_t / 2) * v_t,  v_t ~ N(0, 1)

    where:
        x_t: log-volatility
        y_t: asset return
        φ: persistence parameter (typically close to 1)
        σ_w: volatility of log-volatility
        β: scale parameter for observations
    """

    def __init__(self, phi: float = 0.91, sigma_w: float = 1.0, beta: float = 0.5):
        """
        Initialize Stochastic Volatility Model

        Default parameters from Doucet(09) Example 4:
        - phi = 0.91 (persistence parameter)
        - sigma_w = 1.0 (volatility of log-volatility)
        - beta = 0.5 (observation scale parameter)

        Args:
            phi: Persistence parameter
            sigma_w: Process noise std
            beta: Observation scale parameter
        """
        self.phi = phi
        self.sigma_w = sigma_w
        self.beta = beta

        self.n_dim = 1
        self.obs_dim = 1

    def f(self, x: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        """
        State transition function

        Args:
            x: Current state
            w: Process noise (if None, returns mean)

        Returns:
            Next state
        """
        if w is None:
            return self.phi * x
        else:
            return self.phi * x + self.sigma_w * w

    def h(self, x: np.ndarray, v: np.ndarray = None) -> np.ndarray:
        """
        Observation function

        Args:
            x: Current state
            v: Observation noise (if None, returns mean)

        Returns:
            Observation
        """
        if v is None:
            # Return mean (which is 0 for this model)
            return np.zeros_like(x)
        else:
            return self.beta * np.exp(x / 2) * v

    def f_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition (for EKF)"""
        return np.array([[self.phi]])

    def h_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function (for EKF)

        Note: The SV model has y = β * exp(x/2) * v where v ~ N(0,1).
        The expected observation E[y|x] = 0, so standard EKF linearization fails.

        For EKF to work, we linearize around the observation equation rewritten as:
        y² = β² * exp(x) * v²  =>  log(y²) = log(β²) + x + log(v²)

        This gives a linearized observation: z = log(y²) ≈ x + const
        with Jacobian H = 1.

        However, for the standard form, we return the derivative of
        h(x) = β * exp(x/2) w.r.t. x evaluated at expected v=0, which is 0.

        For practical EKF usage with SV models, use the log-squared transform
        or use UKF/PF instead.
        """
        # Return derivative of β * exp(x/2) * E[v] = 0, which is 0
        # This makes EKF ineffective for SV - use h_jacobian_logsquared instead
        return np.array([[0.0]])

    def h_jacobian_logsquared(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian for log-squared observation transform (alternative for EKF)

        If we use z = log(y²) as observation, then:
        E[z|x] = log(β²) + x + E[log(v²)]

        The Jacobian is simply 1.
        """
        return np.array([[1.0]])

    def get_Q(self) -> np.ndarray:
        """Process noise covariance"""
        return np.array([[self.sigma_w ** 2]])

    def get_R(self, x: np.ndarray) -> np.ndarray:
        """
        Observation noise covariance (state-dependent!)

        For SV model: Var(y_t | x_t) = β² * exp(x_t)
        """
        x = np.asarray(x)
        return np.array([[self.beta ** 2 * np.exp(x.item())]])

    @property
    def Q(self) -> np.ndarray:
        """Process noise covariance as property"""
        return self.get_Q()

    @property
    def R(self) -> np.ndarray:
        """
        Observation noise covariance as property (at x=0)

        Note: For SV model, R is state-dependent. This returns R at x=0
        for compatibility with filters that expect constant R.
        Use get_R(x) for state-dependent R.
        """
        return self.get_R(np.array([0.0]))

    def get_ssm(self) -> NonlinearSSM:
        """Get NonlinearSSM representation"""
        x0 = np.array([0.0])
        P0 = np.array([[self.sigma_w ** 2 / (1 - self.phi ** 2)]])

        return NonlinearSSM(
            f=self.f,
            h=self.h,
            Q=self.get_Q(),
            R=self.get_R(x0),  # Initial R (state-dependent)
            n_dim=self.n_dim,
            obs_dim=self.obs_dim,
            x0=x0,
            P0=P0
        )

    def simulate(self, T: int, x0: np.ndarray = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate data from Stochastic Volatility Model

        Args:
            T: Number of time steps
            x0: Initial state
            seed: Random seed

        Returns:
            states: Log-volatility states (T,)
            observations: Asset returns (T,)
        """
        if seed is not None:
            np.random.seed(seed)

        if x0 is None:
            # Sample from stationary distribution
            x0 = np.random.randn() * self.sigma_w / np.sqrt(1 - self.phi ** 2)

        states = np.zeros(T)
        observations = np.zeros(T)

        x = x0

        for t in range(T):
            # State transition
            w = np.random.randn()
            x = self.f(np.array([x]), np.array([w]))[0]

            # Observation
            v = np.random.randn()
            y = self.h(np.array([x]), np.array([v]))[0]

            states[t] = x
            observations[t] = y

        return states.reshape(-1, 1), observations.reshape(-1, 1)


class RangeBearingModel:
    """
    Range-Bearing Tracking Model

    A target moves in 2D space with constant velocity.
    We observe range and bearing from a fixed sensor.

    State: x_t = [p_x, p_y, v_x, v_y]^T
        - (p_x, p_y): position
        - (v_x, v_y): velocity

    Observation: y_t = [r, θ]^T
        - r: range (distance from sensor)
        - θ: bearing (angle from sensor)

    State transition (linear):
        x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)

    Observation (nonlinear):
        r_t = sqrt((p_x - s_x)² + (p_y - s_y)²) + v_r
        θ_t = atan2(p_y - s_y, p_x - s_x) + v_θ
        where (s_x, s_y) is sensor position
    """

    def __init__(self, dt: float = 1.0, sigma_w: float = 0.1,
                 sigma_r: float = 1.0, sigma_theta: float = 0.1,
                 sensor_pos: np.ndarray = None):
        """
        Initialize Range-Bearing Model

        Args:
            dt: Time step
            sigma_w: Process noise std
            sigma_r: Range measurement noise std
            sigma_theta: Bearing measurement noise std
            sensor_pos: Sensor position [s_x, s_y]
        """
        self.dt = dt
        self.sigma_w = sigma_w
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta

        if sensor_pos is None:
            sensor_pos = np.array([0.0, 0.0])
        self.sensor_pos = sensor_pos

        self.n_dim = 4
        self.obs_dim = 2

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def f(self, x: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        """State transition function"""
        if w is None:
            return self.F @ x
        else:
            return self.F @ x + w

    def h(self, x: np.ndarray, v: np.ndarray = None) -> np.ndarray:
        """
        Observation function (range and bearing)

        Args:
            x: State [p_x, p_y, v_x, v_y]
            v: Observation noise

        Returns:
            [range, bearing]
        """
        # Relative position
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]

        # Range and bearing
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        y = np.array([r, theta])

        if v is not None:
            y += v

        return y

    def f_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of state transition"""
        return self.F

    def h_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation function

        H = [∂r/∂p_x,  ∂r/∂p_y,  0, 0]
            [∂θ/∂p_x,  ∂θ/∂p_y,  0, 0]

        where:
            ∂r/∂p_x = (p_x - s_x) / r
            ∂r/∂p_y = (p_y - s_y) / r
            ∂θ/∂p_x = -(p_y - s_y) / r²
            ∂θ/∂p_y = (p_x - s_x) / r²
        """
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        r2 = r ** 2

        # Avoid division by zero
        if r < 1e-6:
            r = 1e-6
            r2 = r ** 2

        H = np.array([
            [dx / r, dy / r, 0, 0],
            [-dy / r2, dx / r2, 0, 0]
        ])

        return H

    def get_Q(self) -> np.ndarray:
        """Process noise covariance"""
        q = self.sigma_w ** 2
        dt = self.dt
        Q = q * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])
        return Q

    def get_R(self) -> np.ndarray:
        """Observation noise covariance"""
        return np.diag([self.sigma_r ** 2, self.sigma_theta ** 2])

    @property
    def Q(self) -> np.ndarray:
        """Process noise covariance as property"""
        return self.get_Q()

    @property
    def R(self) -> np.ndarray:
        """Observation noise covariance as property"""
        return self.get_R()

    def get_ssm(self) -> NonlinearSSM:
        """Get NonlinearSSM representation"""
        x0 = np.array([10.0, 10.0, 1.0, 0.5])
        P0 = np.diag([10.0, 10.0, 5.0, 5.0])

        return NonlinearSSM(
            f=self.f,
            h=self.h,
            Q=self.get_Q(),
            R=self.get_R(),
            n_dim=self.n_dim,
            obs_dim=self.obs_dim,
            x0=x0,
            P0=P0
        )

    def simulate(self, T: int, x0: np.ndarray = None, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate data from Range-Bearing Model

        Args:
            T: Number of time steps
            x0: Initial state
            seed: Random seed

        Returns:
            states: States (T, 4)
            observations: Observations (T, 2)
        """
        if seed is not None:
            np.random.seed(seed)

        if x0 is None:
            x0 = np.array([10.0, 10.0, 1.0, 0.5])

        states = np.zeros((T, self.n_dim))
        observations = np.zeros((T, self.obs_dim))

        # Cholesky decompositions
        L_Q = np.linalg.cholesky(self.get_Q())
        L_R = np.linalg.cholesky(self.get_R())

        x = x0.copy()

        for t in range(T):
            # State transition
            w = L_Q @ np.random.randn(self.n_dim)
            x = self.f(x, w)

            # Observation
            v = L_R @ np.random.randn(self.obs_dim)
            y = self.h(x, v)

            states[t] = x
            observations[t] = y

        return states, observations


def test_models():
    """Test nonlinear models"""
    print("=" * 70)
    print("Testing Nonlinear State Space Models")
    print("=" * 70)

    # Test Stochastic Volatility Model
    print("\n1. Stochastic Volatility Model")
    print("-" * 70)
    sv_model = StochasticVolatilityModel()
    states_sv, obs_sv = sv_model.simulate(T=100, seed=42)

    print(f"  Generated {len(states_sv)} time steps")
    print(f"  State (log-volatility) range: [{states_sv.min():.3f}, {states_sv.max():.3f}]")
    print(f"  Observation (returns) range: [{obs_sv.min():.3f}, {obs_sv.max():.3f}]")
    print(f"  Observation std: {obs_sv.std():.3f}")

    # Test Range-Bearing Model
    print("\n2. Range-Bearing Tracking Model")
    print("-" * 70)
    rb_model = RangeBearingModel()
    states_rb, obs_rb = rb_model.simulate(T=100, seed=42)

    print(f"  Generated {len(states_rb)} time steps")
    print(f"  Position range: X [{states_rb[:, 0].min():.2f}, {states_rb[:, 0].max():.2f}], "
          f"Y [{states_rb[:, 1].min():.2f}, {states_rb[:, 1].max():.2f}]")
    print(f"  Range observations: [{obs_rb[:, 0].min():.2f}, {obs_rb[:, 0].max():.2f}]")
    print(f"  Bearing observations: [{obs_rb[:, 1].min():.2f}, {obs_rb[:, 1].max():.2f}]")

    return sv_model, rb_model, states_sv, obs_sv, states_rb, obs_rb


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sv_model, rb_model, states_sv, obs_sv, states_rb, obs_rb = test_models()

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # SV model: log-volatility
    ax = axes[0, 0]
    ax.plot(states_sv, 'b-', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Log-Volatility')
    ax.set_title('Stochastic Volatility: Log-Volatility')
    ax.grid(True)

    # SV model: observations
    ax = axes[0, 1]
    ax.plot(obs_sv, 'r-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Returns')
    ax.set_title('Stochastic Volatility: Returns')
    ax.grid(True)

    # Range-Bearing: trajectory
    ax = axes[1, 0]
    ax.plot(states_rb[:, 0], states_rb[:, 1], 'b-', linewidth=2)
    ax.plot(states_rb[0, 0], states_rb[0, 1], 'go', markersize=10, label='Start')
    ax.plot(states_rb[-1, 0], states_rb[-1, 1], 'ro', markersize=10, label='End')
    ax.plot(rb_model.sensor_pos[0], rb_model.sensor_pos[1], 'ks', markersize=12, label='Sensor')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Range-Bearing: True Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Range-Bearing: observations
    ax = axes[1, 1]
    ax.plot(obs_rb[:, 0], obs_rb[:, 1], 'r.', alpha=0.5)
    ax.set_xlabel('Range')
    ax.set_ylabel('Bearing (radians)')
    ax.set_title('Range-Bearing: Observations')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('nonlinear_models.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("Model testing complete!")
    print("=" * 70)
