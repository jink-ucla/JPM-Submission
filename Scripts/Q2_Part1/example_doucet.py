"""
Example 2 from Doucet(09): Nearly Constant Velocity Model
Part 1 Warm-up (I): Linear-Gaussian SSM with Kalman Filter

This script implements Example 2 from Doucet & Johansen (2009):
A 2D tracking problem with nearly constant velocity.

State vector: x_t = [p_x, p_y, v_x, v_y]^T
- (p_x, p_y): position
- (v_x, v_y): velocity

Observations: y_t = [p_x, p_y]^T + noise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from kalman_filter import KalmanFilter, generate_lgssm_data, compute_nees, compute_nis


def setup_nearly_constant_velocity_model(dt: float = 1.0, sigma_w: float = 1.0,
                                        sigma_v: float = 1.0) -> dict:
    """
    Setup Nearly Constant Velocity Model (Example 2 from Doucet 09)

    State: x_t = [p_x, p_y, v_x, v_y]^T
    Observation: y_t = [p_x, p_y]^T

    Args:
        dt: Time step
        sigma_w: Process noise standard deviation
        sigma_v: Observation noise standard deviation

    Returns:
        Dictionary with model parameters
    """
    # State dimension: 4 (position_x, position_y, velocity_x, velocity_y)
    n_dim = 4
    # Observation dimension: 2 (position_x, position_y)
    obs_dim = 2

    # State transition matrix (nearly constant velocity)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Observation matrix (observe positions only)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # Process noise covariance
    # Discretized continuous white noise acceleration model
    q = sigma_w ** 2
    Q = q * np.array([
        [dt**3/3, 0, dt**2/2, 0],
        [0, dt**3/3, 0, dt**2/2],
        [dt**2/2, 0, dt, 0],
        [0, dt**2/2, 0, dt]
    ])

    # Observation noise covariance
    R = (sigma_v ** 2) * np.eye(obs_dim)

    # Initial state and covariance
    x0 = np.array([0.0, 0.0, 1.0, 1.0])  # Start at origin with unit velocity
    P0 = np.diag([10.0, 10.0, 5.0, 5.0])  # Initial uncertainty

    return {
        'F': F,
        'H': H,
        'Q': Q,
        'R': R,
        'x0': x0,
        'P0': P0,
        'n_dim': n_dim,
        'obs_dim': obs_dim,
        'dt': dt,
        'sigma_w': sigma_w,
        'sigma_v': sigma_v
    }


def run_kalman_filter_example(T: int = 100, dt: float = 1.0, sigma_w: float = 0.5,
                              sigma_v: float = 1.0, use_joseph: bool = True,
                              seed: int = 42) -> dict:
    """
    Run Kalman Filter on Nearly Constant Velocity Model

    Args:
        T: Number of time steps
        dt: Time step
        sigma_w: Process noise std
        sigma_v: Observation noise std
        use_joseph: Whether to use Joseph stabilized update
        seed: Random seed

    Returns:
        Dictionary with results
    """
    # Setup model
    model = setup_nearly_constant_velocity_model(dt, sigma_w, sigma_v)

    # Generate data
    print("Generating synthetic data...")
    true_states, observations = generate_lgssm_data(
        model['F'], model['H'], model['Q'], model['R'],
        model['x0'], T, seed=seed
    )

    # Initialize Kalman Filter
    print(f"Running Kalman Filter (Joseph update: {use_joseph})...")
    kf = KalmanFilter(
        model['F'], model['H'], model['Q'], model['R'],
        use_joseph=use_joseph
    )
    kf.initialize(model['x0'], model['P0'])

    # Run filter
    results = kf.filter(observations)

    # Compute metrics
    nees = compute_nees(true_states, results['x_filt'], results['P_filt'])
    nis = compute_nis(results['innovation'], results['S'])

    # Compute errors
    position_errors = np.linalg.norm(
        true_states[:, :2] - results['x_filt'][:, :2], axis=1
    )
    velocity_errors = np.linalg.norm(
        true_states[:, 2:] - results['x_filt'][:, 2:], axis=1
    )

    # Compute RMSE
    rmse_position = np.sqrt(np.mean(position_errors ** 2))
    rmse_velocity = np.sqrt(np.mean(velocity_errors ** 2))

    print(f"\nResults:")
    print(f"  RMSE Position: {rmse_position:.4f}")
    print(f"  RMSE Velocity: {rmse_velocity:.4f}")
    print(f"  Mean NEES: {np.mean(nees):.4f} (expected: {model['n_dim']:.1f})")
    print(f"  Mean NIS: {np.mean(nis):.4f} (expected: {model['obs_dim']:.1f})")
    print(f"  Mean Condition Number (P): {np.mean(results['cond_P']):.2e}")
    print(f"  Max Condition Number (P): {np.max(results['cond_P']):.2e}")

    return {
        'model': model,
        'true_states': true_states,
        'observations': observations,
        'results': results,
        'nees': nees,
        'nis': nis,
        'position_errors': position_errors,
        'velocity_errors': velocity_errors,
        'rmse_position': rmse_position,
        'rmse_velocity': rmse_velocity
    }


def plot_results(output: dict, save_path: str = None):
    """
    Plot Kalman Filter results

    Args:
        output: Output from run_kalman_filter_example
        save_path: Path to save figure (optional)
    """
    true_states = output['true_states']
    observations = output['observations']
    results = output['results']
    nees = output['nees']
    nis = output['nis']
    model = output['model']

    T = len(true_states)
    time = np.arange(T)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: 2D Trajectory
    ax = axes[0, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True', linewidth=2)
    ax.plot(observations[:, 0], observations[:, 1], 'r.', label='Observations', alpha=0.5)
    ax.plot(results['x_filt'][:, 0], results['x_filt'][:, 1], 'g--', label='Filtered', linewidth=2)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.grid(True)

    # Plot 2: Position X
    ax = axes[0, 1]
    ax.plot(time, true_states[:, 0], 'b-', label='True', linewidth=2)
    ax.plot(time, observations[:, 0], 'r.', label='Observations', alpha=0.5)
    ax.plot(time, results['x_filt'][:, 0], 'g--', label='Filtered', linewidth=2)
    # Plot uncertainty
    std = np.sqrt(results['P_filt'][:, 0, 0])
    ax.fill_between(time, results['x_filt'][:, 0] - 2*std, results['x_filt'][:, 0] + 2*std,
                     alpha=0.3, color='g', label='95% CI')
    ax.set_xlabel('Time')
    ax.set_ylabel('X Position')
    ax.set_title('X Position vs Time')
    ax.legend()
    ax.grid(True)

    # Plot 3: Position Y
    ax = axes[1, 0]
    ax.plot(time, true_states[:, 1], 'b-', label='True', linewidth=2)
    ax.plot(time, observations[:, 1], 'r.', label='Observations', alpha=0.5)
    ax.plot(time, results['x_filt'][:, 1], 'g--', label='Filtered', linewidth=2)
    # Plot uncertainty
    std = np.sqrt(results['P_filt'][:, 1, 1])
    ax.fill_between(time, results['x_filt'][:, 1] - 2*std, results['x_filt'][:, 1] + 2*std,
                     alpha=0.3, color='g', label='95% CI')
    ax.set_xlabel('Time')
    ax.set_ylabel('Y Position')
    ax.set_title('Y Position vs Time')
    ax.legend()
    ax.grid(True)

    # Plot 4: NEES
    ax = axes[1, 1]
    ax.plot(time, nees, 'b-', linewidth=1.5)
    # Chi-squared confidence bounds
    alpha = 0.05
    n_dim = model['n_dim']
    upper_bound = stats.chi2.ppf(1 - alpha/2, n_dim)
    lower_bound = stats.chi2.ppf(alpha/2, n_dim)
    ax.axhline(upper_bound, color='r', linestyle='--', label=f'{int((1-alpha)*100)}% bounds')
    ax.axhline(lower_bound, color='r', linestyle='--')
    ax.axhline(n_dim, color='g', linestyle=':', label=f'Expected ({n_dim})')
    ax.set_xlabel('Time')
    ax.set_ylabel('NEES')
    ax.set_title('Normalized Estimation Error Squared (NEES)')
    ax.legend()
    ax.grid(True)

    # Plot 5: NIS
    ax = axes[2, 0]
    ax.plot(time, nis, 'b-', linewidth=1.5)
    # Chi-squared confidence bounds
    obs_dim = model['obs_dim']
    upper_bound = stats.chi2.ppf(1 - alpha/2, obs_dim)
    lower_bound = stats.chi2.ppf(alpha/2, obs_dim)
    ax.axhline(upper_bound, color='r', linestyle='--', label=f'{int((1-alpha)*100)}% bounds')
    ax.axhline(lower_bound, color='r', linestyle='--')
    ax.axhline(obs_dim, color='g', linestyle=':', label=f'Expected ({obs_dim})')
    ax.set_xlabel('Time')
    ax.set_ylabel('NIS')
    ax.set_title('Normalized Innovation Squared (NIS)')
    ax.legend()
    ax.grid(True)

    # Plot 6: Condition Numbers
    ax = axes[2, 1]
    ax.semilogy(time, results['cond_P'], 'b-', label='Cond(P)', linewidth=1.5)
    ax.semilogy(time, results['cond_S'], 'r-', label='Cond(S)', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Condition Number')
    ax.set_title('Covariance Conditioning')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def compare_joseph_update(T: int = 100, seed: int = 42):
    """
    Compare standard and Joseph-stabilized covariance updates

    Args:
        T: Number of time steps
        seed: Random seed
    """
    print("=" * 70)
    print("Comparing Standard vs Joseph-Stabilized Covariance Update")
    print("=" * 70)

    # Run with standard update
    print("\n--- Standard Update ---")
    output_standard = run_kalman_filter_example(T=T, use_joseph=False, seed=seed)

    # Run with Joseph update
    print("\n--- Joseph-Stabilized Update ---")
    output_joseph = run_kalman_filter_example(T=T, use_joseph=True, seed=seed)

    # Compare condition numbers
    print("\n" + "=" * 70)
    print("Condition Number Comparison:")
    print("=" * 70)
    print(f"Standard Update:")
    print(f"  Mean Cond(P): {np.mean(output_standard['results']['cond_P']):.2e}")
    print(f"  Max Cond(P): {np.max(output_standard['results']['cond_P']):.2e}")
    print(f"\nJoseph Update:")
    print(f"  Mean Cond(P): {np.mean(output_joseph['results']['cond_P']):.2e}")
    print(f"  Max Cond(P): {np.max(output_joseph['results']['cond_P']):.2e}")

    # Check positive definiteness
    def check_positive_definite(covs):
        min_eigs = []
        for P in covs:
            eigvals = np.linalg.eigvalsh(P)
            min_eigs.append(np.min(eigvals))
        return np.array(min_eigs)

    min_eigs_standard = check_positive_definite(output_standard['results']['P_filt'])
    min_eigs_joseph = check_positive_definite(output_joseph['results']['P_filt'])

    print(f"\nMinimum Eigenvalue (should be > 0):")
    print(f"  Standard: {np.min(min_eigs_standard):.2e}")
    print(f"  Joseph: {np.min(min_eigs_joseph):.2e}")

    return output_standard, output_joseph


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run basic example
    print("\n" + "=" * 70)
    print("Running Kalman Filter on Nearly Constant Velocity Model")
    print("=" * 70)

    output = run_kalman_filter_example(T=100, dt=1.0, sigma_w=0.5, sigma_v=1.0)

    # Plot results
    fig = plot_results(output, save_path='kalman_filter_results.png')
    plt.show()

    # Compare Joseph update
    output_standard, output_joseph = compare_joseph_update(T=100)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    time = np.arange(len(output_standard['results']['cond_P']))

    # Condition numbers
    ax = axes[0, 0]
    ax.semilogy(time, output_standard['results']['cond_P'], 'b-', label='Standard', linewidth=1.5)
    ax.semilogy(time, output_joseph['results']['cond_P'], 'r-', label='Joseph', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Condition Number')
    ax.set_title('Condition Number of P')
    ax.legend()
    ax.grid(True)

    # Position errors
    ax = axes[0, 1]
    ax.plot(time, output_standard['position_errors'], 'b-', label='Standard', linewidth=1.5)
    ax.plot(time, output_joseph['position_errors'], 'r-', label='Joseph', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position Error')
    ax.set_title('Position Estimation Error')
    ax.legend()
    ax.grid(True)

    # NEES comparison
    ax = axes[1, 0]
    ax.plot(time, output_standard['nees'], 'b-', label='Standard', linewidth=1.5, alpha=0.7)
    ax.plot(time, output_joseph['nees'], 'r-', label='Joseph', linewidth=1.5, alpha=0.7)
    ax.axhline(4, color='g', linestyle=':', label='Expected (4)')
    ax.set_xlabel('Time')
    ax.set_ylabel('NEES')
    ax.set_title('NEES Comparison')
    ax.legend()
    ax.grid(True)

    # Minimum eigenvalue
    min_eigs_standard = [np.min(np.linalg.eigvalsh(P)) for P in output_standard['results']['P_filt']]
    min_eigs_joseph = [np.min(np.linalg.eigvalsh(P)) for P in output_joseph['results']['P_filt']]

    ax = axes[1, 1]
    ax.semilogy(time, min_eigs_standard, 'b-', label='Standard', linewidth=1.5)
    ax.semilogy(time, min_eigs_joseph, 'r-', label='Joseph', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Minimum Eigenvalue')
    ax.set_title('Minimum Eigenvalue of P (Positive Definiteness)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('joseph_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
