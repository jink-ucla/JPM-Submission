"""
High-Dimensional Comparison of Particle Flow Filters

Tests EDH, LEDH, and Kernel methods on high-dimensional problems (20D-100D)
where standard covariance estimation breaks down.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "part1_nonlinear"))

from particle_flow_filters import EDHParticleFlowFilter, LEDHParticleFlowFilter
from kernel_particle_flow import KernelParticleFlowFilter
from particle_filter import ParticleFilter, compute_rmse


@dataclass
class HighDimSSM:
    """High-dimensional State Space Model"""
    f: Callable
    h: Callable
    Q: np.ndarray
    R: np.ndarray
    n_dim: int
    obs_dim: int
    x0: np.ndarray
    P0: np.ndarray


def create_high_dim_model(n_dim: int, obs_dim: int = None, obs_noise: float = 0.5):
    """
    Create a high-dimensional tracking model.

    State transition: x_{t+1} = F @ x_t + w_t (near-random-walk)
    Observation: y_t = H @ x_t + nonlinear_term + v_t

    Args:
        n_dim: State dimension
        obs_dim: Observation dimension (default: n_dim // 2)
        obs_noise: Observation noise level
    """
    if obs_dim is None:
        obs_dim = max(1, n_dim // 2)

    # State transition: slow mean-reversion
    F = 0.95 * np.eye(n_dim)
    Q = 0.1 * np.eye(n_dim)

    # Observation: partial observation of first obs_dim states
    H_base = np.zeros((obs_dim, n_dim))
    for i in range(obs_dim):
        H_base[i, i] = 1.0
    R = obs_noise * np.eye(obs_dim)

    def f(x, w=None):
        result = F @ x
        if w is not None:
            result = result + w
        return result

    def h(x, v=None):
        # Linear part + mild nonlinearity
        linear_part = H_base @ x
        nonlinear_part = 0.1 * np.tanh(linear_part)  # Bounded nonlinearity
        result = linear_part + nonlinear_part
        if v is not None:
            result = result + v
        return result

    def h_jacobian(x):
        # Jacobian of h
        linear_jac = H_base.copy()
        # d/dx tanh(H@x) = diag(1 - tanh^2(H@x)) @ H
        Hx = H_base @ x
        tanh_deriv = 1 - np.tanh(Hx) ** 2
        nonlinear_jac = 0.1 * np.diag(tanh_deriv) @ H_base
        return linear_jac + nonlinear_jac

    x0 = np.zeros(n_dim)
    P0 = np.eye(n_dim)

    model = HighDimSSM(f=f, h=h, Q=Q, R=R, n_dim=n_dim, obs_dim=obs_dim,
                       x0=x0, P0=P0)

    return model, h_jacobian


def simulate_high_dim(model, T, seed=None):
    """Simulate from high-dimensional model"""
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


def run_comparison(n_dim, N_particles=200, T=50, n_steps=20, seed=42):
    """Run comparison for a given dimension"""

    model, h_jacobian = create_high_dim_model(n_dim)
    true_states, observations = simulate_high_dim(model, T, seed=seed)

    results = {}

    # Standard PF
    np.random.seed(seed + 1)
    pf = ParticleFilter(model, N_particles=N_particles, resample_scheme='systematic')
    pf.initialize(model.x0, model.P0)
    pf_results = pf.filter(observations)
    pf_est = pf_results.get('x_filt', pf_results.get('x_mean'))
    results['PF'] = compute_rmse(true_states, pf_est)

    # EDH
    np.random.seed(seed + 1)
    edh = EDHParticleFlowFilter(model, N_particles=N_particles, n_steps=n_steps,
                                 h_jacobian=h_jacobian)
    edh.initialize(model.x0, model.P0)
    try:
        edh_results = edh.filter(observations)
        results['EDH'] = compute_rmse(true_states, edh_results['x_mean'])
    except Exception as e:
        results['EDH'] = np.nan
        print(f"    EDH failed: {e}")

    # LEDH
    np.random.seed(seed + 1)
    ledh = LEDHParticleFlowFilter(model, N_particles=N_particles, n_steps=n_steps,
                                   h_jacobian=h_jacobian)
    ledh.initialize(model.x0, model.P0)
    try:
        ledh_results = ledh.filter(observations)
        results['LEDH'] = compute_rmse(true_states, ledh_results['x_mean'])
    except Exception as e:
        results['LEDH'] = np.nan
        print(f"    LEDH failed: {e}")

    # Kernel Scalar
    np.random.seed(seed + 1)
    kpf_s = KernelParticleFlowFilter(model, N_particles=N_particles, n_steps=n_steps,
                                      matrix_valued=False, h_jacobian=h_jacobian)
    kpf_s.initialize(model.x0, model.P0)
    try:
        kpf_s_results = kpf_s.filter(observations)
        results['Kernel-Scalar'] = compute_rmse(true_states, kpf_s_results['x_mean'])
    except Exception as e:
        results['Kernel-Scalar'] = np.nan
        print(f"    Kernel-Scalar failed: {e}")

    # Kernel Matrix
    np.random.seed(seed + 1)
    kpf_m = KernelParticleFlowFilter(model, N_particles=N_particles, n_steps=n_steps,
                                      matrix_valued=True, h_jacobian=h_jacobian)
    kpf_m.initialize(model.x0, model.P0)
    try:
        kpf_m_results = kpf_m.filter(observations)
        results['Kernel-Matrix'] = compute_rmse(true_states, kpf_m_results['x_mean'])
    except Exception as e:
        results['Kernel-Matrix'] = np.nan
        print(f"    Kernel-Matrix failed: {e}")

    return results


def main():
    print("=" * 70)
    print("HIGH-DIMENSIONAL PARTICLE FLOW FILTER COMPARISON")
    print("=" * 70)
    print()
    print("Testing dimensions: 4, 10, 20, 50, 100")
    print("N_particles = 200, T = 50, n_steps = 20")
    print()

    dimensions = [4, 10, 20, 50, 100]
    all_results = {}

    for n_dim in dimensions:
        print(f"--- Dimension = {n_dim} ---")
        results = run_comparison(n_dim, N_particles=200, T=50, n_steps=20)
        all_results[n_dim] = results

        for method, rmse in results.items():
            print(f"    {method:<15}: RMSE = {rmse:.4f}")
        print()

    # Print summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    methods = ['PF', 'EDH', 'LEDH', 'Kernel-Scalar', 'Kernel-Matrix']

    print(f"{'Dim':>5}", end="")
    for method in methods:
        print(f"{method:>15}", end="")
    print()
    print("-" * 80)

    for n_dim in dimensions:
        print(f"{n_dim:>5}", end="")
        for method in methods:
            rmse = all_results[n_dim].get(method, np.nan)
            print(f"{rmse:>15.4f}", end="")
        print()

    # Find best method for each dimension
    print()
    print("Best method by dimension:")
    for n_dim in dimensions:
        results = all_results[n_dim]
        valid_results = {k: v for k, v in results.items() if not np.isnan(v)}
        if valid_results:
            best_method = min(valid_results, key=valid_results.get)
            print(f"  {n_dim:>3}D: {best_method} (RMSE = {valid_results[best_method]:.4f})")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: RMSE vs Dimension
    ax = axes[0]
    for method in methods:
        rmses = [all_results[d].get(method, np.nan) for d in dimensions]
        ax.plot(dimensions, rmses, 'o-', label=method, markersize=8)

    ax.set_xlabel('State Dimension')
    ax.set_ylabel('RMSE')
    ax.set_title('Filter Performance vs State Dimension\n(Lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Relative performance (normalized to PF)
    ax = axes[1]
    for method in methods:
        if method == 'PF':
            continue
        ratios = []
        for d in dimensions:
            pf_rmse = all_results[d].get('PF', np.nan)
            method_rmse = all_results[d].get(method, np.nan)
            if pf_rmse > 0 and not np.isnan(method_rmse):
                ratios.append(method_rmse / pf_rmse)
            else:
                ratios.append(np.nan)
        ax.plot(dimensions, ratios, 'o-', label=method, markersize=8)

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='PF baseline')
    ax.set_xlabel('State Dimension')
    ax.set_ylabel('RMSE Ratio (Method / PF)')
    ax.set_title('Relative Performance vs PF\n(< 1 means better than PF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('high_dim_comparison.png', dpi=300, bbox_inches='tight')
    print()
    print("Figure saved: high_dim_comparison.png")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
