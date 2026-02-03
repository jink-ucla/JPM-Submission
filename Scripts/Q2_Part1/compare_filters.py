"""
Compare EKF, UKF, and Particle Filter
Part 1 (II-d): Performance comparison

Metrics:
- RMSE (Root Mean Square Error)
- Log-likelihood
- Runtime
- Memory usage
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from typing import Dict
from nonlinear_models import StochasticVolatilityModel, RangeBearingModel
from ekf_ukf import ExtendedKalmanFilter, UnscentedKalmanFilter
from particle_filter import ParticleFilter, compute_rmse


def benchmark_filter(filter_obj, observations, filter_name: str) -> Dict:
    """
    Benchmark a filter: runtime and memory

    Args:
        filter_obj: Filter object with .filter() method
        observations: Observations
        filter_name: Name for printing

    Returns:
        Dictionary with results and benchmarks
    """
    print(f"\nRunning {filter_name}...")

    # Start memory tracking
    tracemalloc.start()

    # Start timer
    start_time = time.time()

    # Run filter
    results = filter_obj.filter(observations)

    # End timer
    end_time = time.time()
    runtime = end_time - start_time

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024

    print(f"  Runtime: {runtime:.4f} seconds")
    print(f"  Peak Memory: {peak_memory_mb:.2f} MB")

    return {
        'results': results,
        'runtime': runtime,
        'peak_memory_mb': peak_memory_mb
    }


def compare_on_stochastic_volatility(T: int = 200, N_particles: int = 100, seed: int = 42):
    """Compare filters on Stochastic Volatility Model"""
    print("=" * 80)
    print("COMPARISON ON STOCHASTIC VOLATILITY MODEL")
    print("=" * 80)

    # Generate data
    np.random.seed(seed)
    sv_model = StochasticVolatilityModel(phi=0.98, sigma_w=0.16, beta=0.6)
    true_states, observations = sv_model.simulate(T=T, seed=seed)

    print(f"\nGenerated {T} time steps")
    print(f"State (log-vol) range: [{true_states.min():.3f}, {true_states.max():.3f}]")

    # Setup filters
    ssm = sv_model.get_ssm()

    # EKF
    ekf = ExtendedKalmanFilter(ssm, sv_model.f_jacobian, sv_model.h_jacobian)
    ekf.initialize(ssm.x0, ssm.P0)

    # UKF
    ukf = UnscentedKalmanFilter(ssm, alpha=1e-3, beta=2.0)
    ukf.initialize(ssm.x0, ssm.P0)

    # PF
    pf = ParticleFilter(ssm, N_particles=N_particles, resample_scheme='systematic')
    pf.initialize(ssm.x0, ssm.P0)

    # Run filters
    ekf_bench = benchmark_filter(ekf, observations, "EKF")
    ukf_bench = benchmark_filter(ukf, observations, "UKF")
    pf_bench = benchmark_filter(pf, observations, f"PF (N={N_particles})")

    # Compute metrics
    ekf_rmse = compute_rmse(true_states, ekf_bench['results']['x_filt'])
    ukf_rmse = compute_rmse(true_states, ukf_bench['results']['x_filt'])
    pf_rmse = compute_rmse(true_states, pf_bench['results']['x_mean'])

    # Compute log-likelihood (for PF only)
    pf_log_lik = np.sum(pf_bench['results']['log_likelihood'])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nRMSE:")
    print(f"  EKF: {ekf_rmse:.4f}")
    print(f"  UKF: {ukf_rmse:.4f}")
    print(f"  PF:  {pf_rmse:.4f}")

    print(f"\nLog-Likelihood:")
    print(f"  PF:  {pf_log_lik:.2f}")

    print(f"\nRuntime:")
    print(f"  EKF: {ekf_bench['runtime']:.4f} s")
    print(f"  UKF: {ukf_bench['runtime']:.4f} s")
    print(f"  PF:  {pf_bench['runtime']:.4f} s")

    print(f"\nPeak Memory:")
    print(f"  EKF: {ekf_bench['peak_memory_mb']:.2f} MB")
    print(f"  UKF: {ukf_bench['peak_memory_mb']:.2f} MB")
    print(f"  PF:  {pf_bench['peak_memory_mb']:.2f} MB")

    # Plot results
    plot_comparison_sv(true_states, observations, ekf_bench, ukf_bench, pf_bench)

    return {
        'ekf': ekf_bench,
        'ukf': ukf_bench,
        'pf': pf_bench,
        'true_states': true_states,
        'observations': observations
    }


def compare_on_range_bearing(T: int = 100, N_particles: int = 100, seed: int = 42):
    """Compare filters on Range-Bearing Model"""
    print("\n" + "=" * 80)
    print("COMPARISON ON RANGE-BEARING MODEL")
    print("=" * 80)

    # Generate data
    np.random.seed(seed)
    rb_model = RangeBearingModel(dt=1.0, sigma_w=0.1, sigma_r=1.0, sigma_theta=0.1)
    true_states, observations = rb_model.simulate(T=T, seed=seed)

    print(f"\nGenerated {T} time steps")
    print(f"Position range: X [{true_states[:, 0].min():.2f}, {true_states[:, 0].max():.2f}], "
          f"Y [{true_states[:, 1].min():.2f}, {true_states[:, 1].max():.2f}]")

    # Setup filters
    ssm = rb_model.get_ssm()

    # EKF
    ekf = ExtendedKalmanFilter(ssm, rb_model.f_jacobian, rb_model.h_jacobian)
    ekf.initialize(ssm.x0, ssm.P0)

    # UKF
    ukf = UnscentedKalmanFilter(ssm, alpha=1e-3, beta=2.0)
    ukf.initialize(ssm.x0, ssm.P0)

    # PF
    pf = ParticleFilter(ssm, N_particles=N_particles, resample_scheme='systematic')
    pf.initialize(ssm.x0, ssm.P0)

    # Run filters
    ekf_bench = benchmark_filter(ekf, observations, "EKF")
    ukf_bench = benchmark_filter(ukf, observations, "UKF")
    pf_bench = benchmark_filter(pf, observations, f"PF (N={N_particles})")

    # Compute metrics (position only)
    ekf_rmse = compute_rmse(true_states[:, :2], ekf_bench['results']['x_filt'][:, :2])
    ukf_rmse = compute_rmse(true_states[:, :2], ukf_bench['results']['x_filt'][:, :2])
    pf_rmse = compute_rmse(true_states[:, :2], pf_bench['results']['x_mean'][:, :2])

    # Compute log-likelihood (for PF only)
    pf_log_lik = np.sum(pf_bench['results']['log_likelihood'])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nRMSE (Position):")
    print(f"  EKF: {ekf_rmse:.4f}")
    print(f"  UKF: {ukf_rmse:.4f}")
    print(f"  PF:  {pf_rmse:.4f}")

    print(f"\nLog-Likelihood:")
    print(f"  PF:  {pf_log_lik:.2f}")

    print(f"\nRuntime:")
    print(f"  EKF: {ekf_bench['runtime']:.4f} s")
    print(f"  UKF: {ukf_bench['runtime']:.4f} s")
    print(f"  PF:  {pf_bench['runtime']:.4f} s")

    print(f"\nPeak Memory:")
    print(f"  EKF: {ekf_bench['peak_memory_mb']:.2f} MB")
    print(f"  UKF: {ukf_bench['peak_memory_mb']:.2f} MB")
    print(f"  PF:  {pf_bench['peak_memory_mb']:.2f} MB")

    # Plot results
    plot_comparison_rb(true_states, observations, ekf_bench, ukf_bench, pf_bench)

    return {
        'ekf': ekf_bench,
        'ukf': ukf_bench,
        'pf': pf_bench,
        'true_states': true_states,
        'observations': observations
    }


def plot_comparison_sv(true_states, observations, ekf_bench, ukf_bench, pf_bench):
    """Plot comparison for SV model"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    T = len(true_states)
    time = np.arange(T)

    # Plot 1: States
    ax = axes[0, 0]
    ax.plot(time, true_states, 'k-', label='True', linewidth=2)
    ax.plot(time, ekf_bench['results']['x_filt'], 'b--', label='EKF', linewidth=1.5)
    ax.plot(time, ukf_bench['results']['x_filt'], 'r--', label='UKF', linewidth=1.5)
    ax.plot(time, pf_bench['results']['x_mean'], 'g--', label='PF', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Log-Volatility')
    ax.set_title('State Estimates')
    ax.legend()
    ax.grid(True)

    # Plot 2: Observations
    ax = axes[0, 1]
    ax.plot(time, observations, 'k.', alpha=0.5, label='Observations')
    ax.set_xlabel('Time')
    ax.set_ylabel('Returns')
    ax.set_title('Observations')
    ax.legend()
    ax.grid(True)

    # Plot 3: Errors
    ax = axes[1, 0]
    ekf_errors = np.abs(true_states - ekf_bench['results']['x_filt'])
    ukf_errors = np.abs(true_states - ukf_bench['results']['x_filt'])
    pf_errors = np.abs(true_states - pf_bench['results']['x_mean'])

    ax.plot(time, ekf_errors, 'b-', label='EKF', alpha=0.7)
    ax.plot(time, ukf_errors, 'r-', label='UKF', alpha=0.7)
    ax.plot(time, pf_errors, 'g-', label='PF', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Estimation Errors')
    ax.legend()
    ax.grid(True)

    # Plot 4: ESS (for PF only)
    ax = axes[1, 1]
    ax.plot(time, pf_bench['results']['ess'], 'g-', linewidth=1.5)
    ax.axhline(len(pf_bench['results']['ess']) / 2, color='r', linestyle='--',
               label='ESS Threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('Particle Filter ESS')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_sv.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: comparison_sv.png")


def plot_comparison_rb(true_states, observations, ekf_bench, ukf_bench, pf_bench):
    """Plot comparison for Range-Bearing model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    T = len(true_states)
    time = np.arange(T)

    # Plot 1: 2D Trajectory
    ax = axes[0, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True', linewidth=2)
    ax.plot(ekf_bench['results']['x_filt'][:, 0], ekf_bench['results']['x_filt'][:, 1],
            'b--', label='EKF', linewidth=1.5)
    ax.plot(ukf_bench['results']['x_filt'][:, 0], ukf_bench['results']['x_filt'][:, 1],
            'r--', label='UKF', linewidth=1.5)
    ax.plot(pf_bench['results']['x_mean'][:, 0], pf_bench['results']['x_mean'][:, 1],
            'g--', label='PF', linewidth=1.5)
    ax.plot(0, 0, 'ks', markersize=12, label='Sensor')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Plot 2: Position X
    ax = axes[0, 1]
    ax.plot(time, true_states[:, 0], 'k-', label='True', linewidth=2)
    ax.plot(time, ekf_bench['results']['x_filt'][:, 0], 'b--', label='EKF', linewidth=1.5)
    ax.plot(time, ukf_bench['results']['x_filt'][:, 0], 'r--', label='UKF', linewidth=1.5)
    ax.plot(time, pf_bench['results']['x_mean'][:, 0], 'g--', label='PF', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('X Position')
    ax.set_title('X Position vs Time')
    ax.legend()
    ax.grid(True)

    # Plot 3: Linearization Error (EKF diagnostic)
    ax = axes[0, 2]
    if 'linearization_error' in ekf_bench['results']:
        ax.plot(time, ekf_bench['results']['linearization_error'], 'b-', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('RMS Linearization Error')
        ax.set_title('EKF Linearization Error\n(Higher = stronger nonlinearity)')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No linearization error data', ha='center', va='center')

    # Plot 4: Position Errors
    ax = axes[1, 0]
    ekf_errors = np.linalg.norm(true_states[:, :2] - ekf_bench['results']['x_filt'][:, :2], axis=1)
    ukf_errors = np.linalg.norm(true_states[:, :2] - ukf_bench['results']['x_filt'][:, :2], axis=1)
    pf_errors = np.linalg.norm(true_states[:, :2] - pf_bench['results']['x_mean'][:, :2], axis=1)

    ax.plot(time, ekf_errors, 'b-', label='EKF', alpha=0.7)
    ax.plot(time, ukf_errors, 'r-', label='UKF', alpha=0.7)
    ax.plot(time, pf_errors, 'g-', label='PF', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position Error')
    ax.set_title('Position Estimation Errors')
    ax.legend()
    ax.grid(True)

    # Plot 5: ESS (PF diagnostic)
    ax = axes[1, 1]
    ax.plot(time, pf_bench['results']['ess'], 'g-', linewidth=1.5)
    ax.axhline(len(pf_bench['results']['ess']) / 2, color='r', linestyle='--',
               label='ESS Threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('Particle Filter ESS\n(Particle Degeneracy Diagnostic)')
    ax.legend()
    ax.grid(True)

    # Plot 6: UKF Sigma Point Failures
    ax = axes[1, 2]
    if 'sigma_failures' in ukf_bench['results']:
        failures = ukf_bench['results']['sigma_failures'].astype(int)
        ax.bar(time, failures, color='red', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Sigma Point Failure (1=yes)')
        ax.set_title('UKF Sigma Point Failures\n(Strong nonlinearity indicator)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No sigma failure data', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('comparison_rb.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: comparison_rb.png")


if __name__ == "__main__":
    # Compare on both models
    sv_results = compare_on_stochastic_volatility(T=200, N_particles=100, seed=42)
    rb_results = compare_on_range_bearing(T=100, N_particles=100, seed=42)

    plt.show()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
