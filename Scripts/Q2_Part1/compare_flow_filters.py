"""
Comprehensive Comparison of Particle Flow Filters
Part 1: Deterministic and Kernel Flows

Compares:
1. EDH (Exact Daum-Huang)
2. LEDH (Local Exact Daum-Huang)
3. PF-PF (Invertible Particle Flow Particle Filter)
4. Kernel PFF (Scalar RBF)
5. Kernel PFF (Matrix-valued)
6. Standard Bootstrap Particle Filter (baseline)

Analyzes when each method excels or fails based on:
- Nonlinearity strength
- Observation sparsity
- Dimension
- Conditioning

References:
- Daum & Huang (2010, 2011)
- Li & Coates (2017)
- Hu & Van Leeuwen (2021)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "part1_nonlinear"))

from nonlinear_models import StochasticVolatilityModel, RangeBearingModel
from particle_filter import ParticleFilter, compute_rmse
from particle_flow_filters import EDHParticleFlowFilter, LEDHParticleFlowFilter, InvertiblePFPF
from kernel_particle_flow import KernelParticleFlowFilter


def benchmark_filter(filter_obj, observations: np.ndarray, filter_name: str,
                    verbose: bool = True) -> Dict:
    """Benchmark a filter for runtime and memory"""
    if verbose:
        print(f"  Running {filter_name}...", end=" ", flush=True)

    tracemalloc.start()
    start_time = time.time()

    results = filter_obj.filter(observations)

    runtime = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024

    if verbose:
        print(f"Done ({runtime:.2f}s, {peak_memory_mb:.1f}MB)")

    return {
        'results': results,
        'runtime': runtime,
        'peak_memory_mb': peak_memory_mb,
        'name': filter_name
    }


def compare_all_flow_filters(model, true_states: np.ndarray, observations: np.ndarray,
                             N_particles: int = 100, n_steps: int = 30,
                             h_jacobian=None) -> Dict:
    """
    Compare all particle flow methods on a given model

    Args:
        model: NonlinearSSM object
        true_states: Ground truth states
        observations: Observations
        N_particles: Number of particles
        n_steps: Integration steps for flows
        h_jacobian: Jacobian of observation function

    Returns:
        Dictionary with all results
    """
    ssm = model.get_ssm() if hasattr(model, 'get_ssm') else model

    results = {}

    # 1. Standard Particle Filter (baseline)
    pf = ParticleFilter(ssm, N_particles=N_particles, resample_scheme='systematic')
    pf.initialize(ssm.x0, ssm.P0)
    results['PF'] = benchmark_filter(pf, observations, "Standard PF")

    # 2. EDH Particle Flow
    edh = EDHParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                 h_jacobian=h_jacobian)
    edh.initialize(ssm.x0, ssm.P0)
    results['EDH'] = benchmark_filter(edh, observations, "EDH Flow")

    # 3. LEDH Particle Flow
    ledh = LEDHParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                   h_jacobian=h_jacobian)
    ledh.initialize(ssm.x0, ssm.P0)
    results['LEDH'] = benchmark_filter(ledh, observations, "LEDH Flow")

    # 4. Invertible PF-PF (Li 2017)
    pfpf = InvertiblePFPF(ssm, N_particles=N_particles, n_steps=n_steps,
                          h_jacobian=h_jacobian, flow_type='ledh')
    pfpf.initialize(ssm.x0, ssm.P0)
    results['PF-PF'] = benchmark_filter(pfpf, observations, "PF-PF (Li17)")

    # 5. Kernel PFF - Scalar RBF
    kpf_scalar = KernelParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                          matrix_valued=False, h_jacobian=h_jacobian)
    kpf_scalar.initialize(ssm.x0, ssm.P0)
    results['Kernel-Scalar'] = benchmark_filter(kpf_scalar, observations, "Kernel (Scalar)")

    # 6. Kernel PFF - Matrix-valued
    kpf_matrix = KernelParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                          matrix_valued=True, h_jacobian=h_jacobian)
    kpf_matrix.initialize(ssm.x0, ssm.P0)
    results['Kernel-Matrix'] = benchmark_filter(kpf_matrix, observations, "Kernel (Matrix)")

    # Compute RMSE for each method
    for name, res in results.items():
        if 'x_mean' in res['results']:
            estimates = res['results']['x_mean']
        else:
            estimates = res['results']['x_filt'] if 'x_filt' in res['results'] else None

        if estimates is not None:
            res['rmse'] = compute_rmse(true_states, estimates)
        else:
            res['rmse'] = np.nan

    return results


def analyze_nonlinearity_effect(seed: int = 42):
    """
    Analyze how different methods handle varying nonlinearity strengths

    Uses Range-Bearing model with varying observation noise
    (lower noise = stronger effective nonlinearity due to tighter constraints)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: Effect of Nonlinearity Strength")
    print("=" * 80)

    np.random.seed(seed)

    # Varying observation noise (lower = stronger nonlinearity effect)
    sigma_theta_values = [0.5, 0.2, 0.1, 0.05]
    T = 50
    N_particles = 100

    results_by_noise = {}

    for sigma_theta in sigma_theta_values:
        print(f"\n--- Bearing noise σ_θ = {sigma_theta} ---")

        rb_model = RangeBearingModel(dt=1.0, sigma_w=0.1, sigma_r=1.0,
                                     sigma_theta=sigma_theta)
        true_states, observations = rb_model.simulate(T=T, seed=seed)

        results = compare_all_flow_filters(
            rb_model, true_states, observations,
            N_particles=N_particles, n_steps=30,
            h_jacobian=rb_model.h_jacobian
        )

        results_by_noise[sigma_theta] = {
            name: res['rmse'] for name, res in results.items()
        }

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_by_noise[sigma_theta_values[0]].keys())
    x = np.arange(len(sigma_theta_values))
    width = 0.12

    for i, method in enumerate(methods):
        rmses = [results_by_noise[sigma][method] for sigma in sigma_theta_values]
        ax.bar(x + i * width, rmses, width, label=method)

    ax.set_xlabel('Bearing Noise σ_θ (lower = stronger nonlinearity effect)')
    ax.set_ylabel('RMSE')
    ax.set_title('Filter Performance vs Nonlinearity Strength')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(s) for s in sigma_theta_values])
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nonlinearity_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: nonlinearity_comparison.png")

    return results_by_noise


def analyze_dimension_effect(seed: int = 42):
    """
    Analyze how methods scale with state dimension

    Creates synthetic high-dimensional linear-Gaussian models to test
    kernel matrix-valued vs scalar performance (Hu 2021 motivation)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: Effect of State Dimension")
    print("=" * 80)

    from dataclasses import dataclass
    from typing import Callable

    @dataclass
    class HighDimSSM:
        """High-dimensional SSM for testing"""
        f: Callable
        h: Callable
        Q: np.ndarray
        R: np.ndarray
        n_dim: int
        obs_dim: int
        x0: np.ndarray
        P0: np.ndarray

    def create_high_dim_model(n_dim: int, obs_dim: int = None):
        """Create a high-dimensional tracking model"""
        if obs_dim is None:
            obs_dim = max(1, n_dim // 2)

        # State transition: random walk with drift
        F = 0.95 * np.eye(n_dim) + 0.02 * np.random.randn(n_dim, n_dim)
        Q = 0.1 * np.eye(n_dim)

        # Observation: partial observation with nonlinearity
        H_base = np.zeros((obs_dim, n_dim))
        for i in range(obs_dim):
            H_base[i, i] = 1.0
        R = 0.5 * np.eye(obs_dim)

        def f(x, w=None):
            result = F @ x
            if w is not None:
                result += w
            return result

        def h(x, v=None):
            # Nonlinear observation: squared terms
            linear_part = H_base @ x
            nonlinear_part = 0.1 * (H_base @ x) ** 2
            result = linear_part + nonlinear_part
            if v is not None:
                result += v
            return result

        def h_jacobian(x):
            linear_jac = H_base
            nonlinear_jac = 0.2 * np.diag(H_base @ x) @ H_base
            return linear_jac + nonlinear_jac

        x0 = np.zeros(n_dim)
        P0 = np.eye(n_dim)

        return HighDimSSM(f=f, h=h, Q=Q, R=R, n_dim=n_dim, obs_dim=obs_dim,
                         x0=x0, P0=P0), h_jacobian

    def simulate_high_dim(model, T, seed=None):
        """Simulate from high-dim model"""
        if seed is not None:
            np.random.seed(seed)

        n_dim = model.n_dim
        obs_dim = model.obs_dim
        L_Q = np.linalg.cholesky(model.Q)
        L_R = np.linalg.cholesky(model.R)

        states = np.zeros((T, n_dim))
        observations = np.zeros((T, obs_dim))

        x = model.x0.copy()
        for t in range(T):
            w = L_Q @ np.random.randn(n_dim)
            x = model.f(x, w)
            v = L_R @ np.random.randn(obs_dim)
            y = model.h(x, v)
            states[t] = x
            observations[t] = y

        return states, observations

    np.random.seed(seed)

    dimensions = [2, 4, 8, 16]
    T = 30
    N_particles = 50  # Fewer particles for high-dim

    results_by_dim = {}

    for n_dim in dimensions:
        print(f"\n--- State dimension = {n_dim} ---")

        model, h_jac = create_high_dim_model(n_dim)
        true_states, observations = simulate_high_dim(model, T, seed=seed)

        # Only compare kernel methods (focus of Hu 2021)
        results = {}

        # Kernel - Scalar
        kpf_scalar = KernelParticleFlowFilter(model, N_particles=N_particles, n_steps=20,
                                              matrix_valued=False, h_jacobian=h_jac)
        kpf_scalar.initialize(model.x0, model.P0)
        results['Kernel-Scalar'] = benchmark_filter(kpf_scalar, observations, "Kernel (Scalar)")

        # Kernel - Matrix
        kpf_matrix = KernelParticleFlowFilter(model, N_particles=N_particles, n_steps=20,
                                              matrix_valued=True, h_jacobian=h_jac)
        kpf_matrix.initialize(model.x0, model.P0)
        results['Kernel-Matrix'] = benchmark_filter(kpf_matrix, observations, "Kernel (Matrix)")

        # Standard PF
        pf = ParticleFilter(model, N_particles=N_particles, resample_scheme='systematic')
        pf.initialize(model.x0, model.P0)
        results['PF'] = benchmark_filter(pf, observations, "Standard PF")

        # Compute RMSE
        for name, res in results.items():
            if 'x_mean' in res['results']:
                estimates = res['results']['x_mean']
            else:
                estimates = res['results'].get('x_filt', None)

            if estimates is not None:
                res['rmse'] = compute_rmse(true_states, estimates)
            else:
                res['rmse'] = np.nan

        results_by_dim[n_dim] = {name: res['rmse'] for name, res in results.items()}

    # Plot results (Figure 2-3 style from Hu 2021)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE comparison
    ax = axes[0]
    methods = list(results_by_dim[dimensions[0]].keys())
    x = np.arange(len(dimensions))
    width = 0.25

    for i, method in enumerate(methods):
        rmses = [results_by_dim[dim][method] for dim in dimensions]
        ax.bar(x + i * width, rmses, width, label=method)

    ax.set_xlabel('State Dimension')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE vs State Dimension\n(Matrix kernel prevents marginal collapse)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(d) for d in dimensions])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratio plot: Matrix/Scalar RMSE
    ax = axes[1]
    ratios = [results_by_dim[dim]['Kernel-Matrix'] / results_by_dim[dim]['Kernel-Scalar']
              for dim in dimensions]
    ax.bar(x, ratios, color='steelblue')
    ax.axhline(1.0, color='r', linestyle='--', label='Equal performance')
    ax.set_xlabel('State Dimension')
    ax.set_ylabel('RMSE Ratio (Matrix / Scalar)')
    ax.set_title('Matrix vs Scalar Kernel Performance\n(< 1 means Matrix is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dimensions])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dimension_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: dimension_comparison.png")

    return results_by_dim


def analyze_stability_diagnostics(seed: int = 42):
    """
    Analyze stability diagnostics: flow magnitude and Jacobian conditioning
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: Stability Diagnostics")
    print("=" * 80)

    np.random.seed(seed)

    # Use Range-Bearing model
    rb_model = RangeBearingModel(dt=1.0, sigma_w=0.1, sigma_r=1.0, sigma_theta=0.1)
    true_states, observations = rb_model.simulate(T=100, seed=seed)
    ssm = rb_model.get_ssm()

    N_particles = 100
    n_steps = 30

    # Run filters and collect diagnostics
    print("\nRunning filters with diagnostics...")

    # EDH
    edh = EDHParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                 h_jacobian=rb_model.h_jacobian)
    edh.initialize(ssm.x0, ssm.P0)
    edh_results = edh.filter(observations)
    print("  EDH done")

    # LEDH
    ledh = LEDHParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                   h_jacobian=rb_model.h_jacobian)
    ledh.initialize(ssm.x0, ssm.P0)
    ledh_results = ledh.filter(observations)
    print("  LEDH done")

    # Kernel (Matrix)
    kpf = KernelParticleFlowFilter(ssm, N_particles=N_particles, n_steps=n_steps,
                                    matrix_valued=True, h_jacobian=rb_model.h_jacobian)
    kpf.initialize(ssm.x0, ssm.P0)
    kpf_results = kpf.filter(observations)
    print("  Kernel done")

    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    T = len(observations)
    time = np.arange(T)

    # Flow magnitude comparison
    ax = axes[0, 0]
    ax.plot(time, edh_results['flow_magnitude'], 'b-', label='EDH', alpha=0.8)
    ax.plot(time, ledh_results['flow_magnitude'], 'r-', label='LEDH', alpha=0.8)
    ax.plot(time, kpf_results['flow_magnitude'], 'g-', label='Kernel', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Flow Magnitude')
    ax.set_title('Flow Magnitude Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Jacobian conditioning
    ax = axes[0, 1]
    ax.semilogy(time, edh_results['jacobian_cond'], 'b-', label='EDH', alpha=0.8)
    ax.semilogy(time, ledh_results['jacobian_cond'], 'r-', label='LEDH', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Jacobian Condition Number')
    ax.set_title('Flow Jacobian Conditioning\n(Lower = more stable)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # State estimates comparison
    ax = axes[1, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True', linewidth=2)
    ax.plot(edh_results['x_mean'][:, 0], edh_results['x_mean'][:, 1], 'b--',
            label='EDH', alpha=0.8)
    ax.plot(ledh_results['x_mean'][:, 0], ledh_results['x_mean'][:, 1], 'r--',
            label='LEDH', alpha=0.8)
    ax.plot(kpf_results['x_mean'][:, 0], kpf_results['x_mean'][:, 1], 'g--',
            label='Kernel', alpha=0.8)
    ax.plot(0, 0, 'ks', markersize=10, label='Sensor')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('2D Trajectory Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # RMSE over time
    ax = axes[1, 1]
    edh_errors = np.linalg.norm(true_states[:, :2] - edh_results['x_mean'][:, :2], axis=1)
    ledh_errors = np.linalg.norm(true_states[:, :2] - ledh_results['x_mean'][:, :2], axis=1)
    kpf_errors = np.linalg.norm(true_states[:, :2] - kpf_results['x_mean'][:, :2], axis=1)

    ax.plot(time, edh_errors, 'b-', label='EDH', alpha=0.8)
    ax.plot(time, ledh_errors, 'r-', label='LEDH', alpha=0.8)
    ax.plot(time, kpf_errors, 'g-', label='Kernel', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position Error')
    ax.set_title('Position Estimation Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stability_diagnostics.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: stability_diagnostics.png")

    # Print summary statistics
    print("\n" + "-" * 60)
    print("Summary Statistics:")
    print("-" * 60)
    print(f"{'Method':<15} {'RMSE':>10} {'Mean Flow':>12} {'Mean Cond':>12}")
    print("-" * 60)
    print(f"{'EDH':<15} {compute_rmse(true_states, edh_results['x_mean']):>10.4f} "
          f"{np.mean(edh_results['flow_magnitude']):>12.4f} "
          f"{np.mean(edh_results['jacobian_cond']):>12.2f}")
    print(f"{'LEDH':<15} {compute_rmse(true_states, ledh_results['x_mean']):>10.4f} "
          f"{np.mean(ledh_results['flow_magnitude']):>12.4f} "
          f"{np.mean(ledh_results['jacobian_cond']):>12.2f}")
    print(f"{'Kernel':<15} {compute_rmse(true_states, kpf_results['x_mean']):>10.4f} "
          f"{np.mean(kpf_results['flow_magnitude']):>12.4f} "
          f"{'N/A':>12}")

    return {
        'edh': edh_results,
        'ledh': ledh_results,
        'kernel': kpf_results
    }


def main_comparison(seed: int = 42):
    """
    Main comparison of all methods on standard test problems
    """
    print("\n" + "=" * 80)
    print("MAIN COMPARISON: All Particle Flow Methods")
    print("=" * 80)

    np.random.seed(seed)

    # Test on Range-Bearing model
    print("\n--- Range-Bearing Tracking Model ---")
    rb_model = RangeBearingModel(dt=1.0, sigma_w=0.1, sigma_r=1.0, sigma_theta=0.1)
    true_states_rb, obs_rb = rb_model.simulate(T=100, seed=seed)

    results_rb = compare_all_flow_filters(
        rb_model, true_states_rb, obs_rb,
        N_particles=100, n_steps=30,
        h_jacobian=rb_model.h_jacobian
    )

    # Test on Stochastic Volatility model
    print("\n--- Stochastic Volatility Model ---")
    sv_model = StochasticVolatilityModel(phi=0.98, sigma_w=0.16, beta=0.6)
    true_states_sv, obs_sv = sv_model.simulate(T=200, seed=seed)

    results_sv = compare_all_flow_filters(
        sv_model, true_states_sv, obs_sv,
        N_particles=100, n_steps=30,
        h_jacobian=None  # SV model has problematic Jacobian
    )

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\nRange-Bearing Model:")
    print(f"{'Method':<20} {'RMSE':>10} {'Runtime (s)':>12} {'Memory (MB)':>12}")
    print("-" * 60)
    for name, res in results_rb.items():
        print(f"{name:<20} {res['rmse']:>10.4f} {res['runtime']:>12.3f} {res['peak_memory_mb']:>12.2f}")

    print("\nStochastic Volatility Model:")
    print(f"{'Method':<20} {'RMSE':>10} {'Runtime (s)':>12} {'Memory (MB)':>12}")
    print("-" * 60)
    for name, res in results_sv.items():
        print(f"{name:<20} {res['rmse']:>10.4f} {res['runtime']:>12.3f} {res['peak_memory_mb']:>12.2f}")

    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Range-Bearing RMSE
    ax = axes[0]
    methods = list(results_rb.keys())
    rmses = [results_rb[m]['rmse'] for m in methods]
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, rmses, color=colors)
    ax.set_ylabel('RMSE')
    ax.set_title('Range-Bearing Model: RMSE Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # Stochastic Volatility RMSE
    ax = axes[1]
    rmses = [results_sv[m]['rmse'] for m in methods]
    bars = ax.bar(methods, rmses, color=colors)
    ax.set_ylabel('RMSE')
    ax.set_title('Stochastic Volatility Model: RMSE Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('main_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved: main_comparison.png")

    return results_rb, results_sv


def analyze_method_strengths():
    """
    Summarize when each method excels or fails
    """
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY: When Each Method Excels or Fails")
    print("=" * 80)

    analysis = """
    Method              | Excels When                          | Fails When
    --------------------|--------------------------------------|----------------------------------
    Standard PF         | - Low dimensions                     | - High dimensions (curse of dim)
                        | - Sufficient particles               | - Strong nonlinearity
                        | - Bootstrap proposal works well      | - Weight degeneracy

    EDH Flow            | - Moderate nonlinearity              | - Strong nonlinearity (stiff flow)
                        | - Good observation coverage          | - Sparse observations
                        | - Low to moderate dimensions         | - High dimensions (global cov)

    LEDH Flow           | - Particle diversity needed          | - Very high dimensions
                        | - Moderate nonlinearity              | - Insufficient particles for
                        | - Localization benefits              |   local covariance estimation

    PF-PF (Li 2017)     | - Importance weighting needed        | - Flow non-invertible
                        | - Combining flow with resampling     | - Numerical instability
                        | - Moderate dimensions                | - Extreme Jacobians

    Kernel Scalar       | - Low dimensions                     | - High dimensions (collapse)
                        | - Smooth posteriors                  | - Sharp multimodal posteriors
                        | - Similar scale across dims          | - Anisotropic distributions

    Kernel Matrix       | - High dimensions                    | - Very low dimensions (overkill)
                        | - Prevent marginal collapse          | - Isotropic problems
                        | - Anisotropic distributions          | - Computational constraints

    Key Observations:
    1. LEDH typically outperforms EDH by maintaining particle diversity
    2. Matrix-valued kernel prevents marginal collapse in high dimensions
    3. PF-PF provides proper importance weights but adds complexity
    4. Standard PF remains competitive for low-dimensional problems
    5. Flow methods benefit from more integration steps but cost more
    """
    print(analysis)


if __name__ == "__main__":
    # Run all analyses
    print("\n" + "#" * 80)
    print("# COMPREHENSIVE PARTICLE FLOW FILTER COMPARISON")
    print("# Part 1: Deterministic and Kernel Flows")
    print("#" * 80)

    # Main comparison
    results_rb, results_sv = main_comparison(seed=42)

    # Stability diagnostics
    stability_results = analyze_stability_diagnostics(seed=42)

    # Nonlinearity analysis
    nonlinearity_results = analyze_nonlinearity_effect(seed=42)

    # Dimension analysis
    dimension_results = analyze_dimension_effect(seed=42)

    # Summary
    analyze_method_strengths()

    plt.show()

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("Generated figures:")
    print("  - main_comparison.png")
    print("  - stability_diagnostics.png")
    print("  - nonlinearity_comparison.png")
    print("  - dimension_comparison.png")
    print("=" * 80)
