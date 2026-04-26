#!/usr/bin/env python3
"""
=============================================================================
Multi-Direction Optimization & Algorithm Comparison
=============================================================================
Extends the inverse identification to:
  1. Multi-objective optimization across 0°, 45°, 90° directions simultaneously
  2. Comparison of optimization algorithms (Nelder-Mead, DE, PSO, Bayesian)
  3. Uses surrogate analytical model (no Abaqus needed for algorithm comparison)

The surrogate model uses the combined hardening stress equation directly,
which is exact for uniaxial monotonic tension. The actual FEA-based
optimization (optimize_hardening.py) runs on Abaqus/Windows separately.
The weighted surrogate study in this file prioritizes the 0-degree rolling
direction (2:1:1). That weighting is intentionally different from the Abaqus
three-direction script, which emphasizes 45 degrees because it is the most
anisotropy-sensitive direction in the validated FEA comparison.

Author: HOUR Sokaon
Date:   2026
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.signal import savgol_filter
import os
import time
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

E_YOUNG = 200000.0  # MPa
PLASTIC_STRAIN_THRESHOLD = 0.002
SURROGATE_WEIGHTED_WEIGHTS = {'00': 2.0, '45': 1.0, '90': 1.0}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'lines.linewidth': 1.5, 'font.family': 'serif',
})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_stress_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['Time', 'Load', 'Stress', 'Strain']
    return df

def load_dic_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    delimiter = ';' if ';' in lines[7] else ','
    data_lines = lines[7:]
    steps, eyy, exx, exy = [], [], [], []
    for line in data_lines:
        parts = line.strip().split(delimiter)
        if len(parts) >= 5:
            try:
                steps.append(int(parts[1]))
                eyy.append(float(parts[2]))
                exx.append(float(parts[3]))
                exy.append(float(parts[4]))
            except:
                continue
    return pd.DataFrame({'Step': steps, 'Eyy': eyy, 'Exx': exx, 'Exy': exy})


def load_all_experimental():
    """Load and process experimental data for all 3 directions."""
    exp_data = {}

    for direction in ['00', '45', '90']:
        all_ep = []
        all_ts = []
        for specimen in ['01', '02', '03']:
            stress_file = os.path.join(DATA_DIR, f'stress-{direction}-{specimen}.csv')
            if not os.path.exists(stress_file):
                continue
            df = load_stress_data(stress_file)
            eng_s = df['Stress'].values
            eng_e = df['Strain'].values

            # Truncate at UTS
            uts_idx = np.argmax(eng_s)
            eng_s = eng_s[1:uts_idx+1]
            eng_e = eng_e[1:uts_idx+1]

            # True conversion
            true_strain = np.log(1 + eng_e)
            true_stress = eng_s * (1 + eng_e)
            plastic_strain = true_strain - true_stress / E_YOUNG

            mask = plastic_strain > PLASTIC_STRAIN_THRESHOLD
            if np.sum(mask) > 10:
                all_ep.extend(plastic_strain[mask].tolist())
                all_ts.extend(true_stress[mask].tolist())

        if all_ep:
            ep = np.array(all_ep)
            ts = np.array(all_ts)
            sort_idx = np.argsort(ep)
            exp_data[direction] = {
                'plastic_strain': ep[sort_idx],
                'true_stress': ts[sort_idx]
            }

    return exp_data


# ============================================================================
# COMBINED HARDENING MODEL (Analytical surrogate for uniaxial tension)
# ============================================================================

def combined_hardening_stress(eps_p, sigma0, C1, gamma1, C2, gamma2, Q_inf, b_iso):
    """
    Total stress for monotonic uniaxial tension under combined hardening:
    sigma = sigma_y(ep) + alpha(ep)
          = [sigma0 + Q*(1-exp(-b*ep))] + [C1/g1*(1-exp(-g1*ep)) + C2/g2*(1-exp(-g2*ep))]
    """
    iso = sigma0 + Q_inf * (1.0 - np.exp(-b_iso * eps_p))
    kin1 = (C1 / gamma1) * (1.0 - np.exp(-gamma1 * eps_p))
    kin2 = (C2 / gamma2) * (1.0 - np.exp(-gamma2 * eps_p))
    return iso + kin1 + kin2


# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================

def single_direction_objective(params, exp_data_dir, fixed_iso):
    """NRMSE for a single direction."""
    sigma0, C1, gamma1, C2, gamma2 = params
    Q_inf, b_iso = fixed_iso

    if any(p <= 0 for p in params):
        return 1e10

    ep = exp_data_dir['plastic_strain']
    ts_exp = exp_data_dir['true_stress']

    ts_pred = combined_hardening_stress(ep, sigma0, C1, gamma1, C2, gamma2,
                                         Q_inf, b_iso)

    nrmse = np.sqrt(np.mean((ts_pred - ts_exp)**2)) / (ts_exp.max() - ts_exp.min())
    return nrmse


def multi_direction_objective(params, exp_data, fixed_iso, weights=None):
    """
    Multi-objective: weighted sum of NRMSE across all 3 directions.

    params: [sigma0, C1, gamma1, C2, gamma2]
    weights: dict {'00': w0, '45': w45, '90': w90}, default equal
    """
    if weights is None:
        weights = {'00': 1.0, '45': 1.0, '90': 1.0}

    total_error = 0.0
    total_weight = 0.0

    for direction, w in weights.items():
        if direction in exp_data:
            err = single_direction_objective(params, exp_data[direction], fixed_iso)
            if err < 1e8:
                total_error += w * err
                total_weight += w

    if total_weight == 0:
        return 1e10

    return total_error / total_weight


# ============================================================================
# VOCE FIT FOR ISOTROPIC PARAMETERS
# ============================================================================

def fit_voce_for_iso(exp_data, direction='00'):
    """Get Voce isotropic parameters from experimental data."""
    ep = exp_data[direction]['plastic_strain']
    ts = exp_data[direction]['true_stress']

    def voce(x, sy, Q, b):
        return sy + Q * (1 - np.exp(-b * x))

    try:
        popt, _ = curve_fit(voce, ep, ts, p0=[250, 200, 5],
                           bounds=([50, 10, 0.1], [500, 1000, 100]))
        return popt[1], popt[2]  # Q_inf, b_iso
    except:
        return 335.16, 3.95  # fallback


# ============================================================================
# OPTIMIZATION ALGORITHMS
# ============================================================================

def run_nelder_mead(objective, x0, args, bounds_low, bounds_high):
    """Nelder-Mead with penalty for bounds."""
    def bounded_obj(x, *a):
        penalty = sum(1000*(lo-v)**2 for v, lo in zip(x, bounds_low) if v < lo)
        penalty += sum(1000*(v-hi)**2 for v, hi in zip(x, bounds_high) if v > hi)
        return objective(x, *a) + penalty

    result = minimize(bounded_obj, x0, args=args, method='Nelder-Mead',
                     options={'maxiter': 500, 'maxfev': 2000, 'xatol': 1e-5,
                              'fatol': 1e-6, 'adaptive': True})
    return result


def run_differential_evolution(objective, bounds, args):
    """Differential Evolution (global optimizer)."""
    result = differential_evolution(objective, bounds, args=args,
                                    maxiter=200, seed=42, tol=1e-6,
                                    popsize=20, mutation=(0.5, 1.5),
                                    recombination=0.9, polish=True)
    return result


def run_basin_hopping(objective, x0, args, bounds_low, bounds_high):
    """Basin-Hopping (global + local)."""
    from scipy.optimize import basinhopping

    class BoundsCheck:
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi
        def __call__(self, **kwargs):
            x = kwargs['x_new']
            return all(lo <= v <= hi for v, lo, hi in zip(x, self.lo, self.hi))

    minimizer_kwargs = {'method': 'L-BFGS-B', 'args': args,
                        'bounds': list(zip(bounds_low, bounds_high))}
    result = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs,
                         niter=100, seed=42,
                         accept_test=BoundsCheck(bounds_low, bounds_high))
    return result


def run_pso(objective, bounds_low, bounds_high, args, n_particles=30, n_iters=150):
    """Particle Swarm Optimization (custom implementation)."""
    np.random.seed(42)
    dim = len(bounds_low)
    lo = np.array(bounds_low)
    hi = np.array(bounds_high)

    # Initialize particles
    positions = lo + np.random.rand(n_particles, dim) * (hi - lo)
    velocities = np.random.randn(n_particles, dim) * (hi - lo) * 0.1
    personal_best_pos = positions.copy()
    personal_best_cost = np.full(n_particles, np.inf)
    global_best_pos = positions[0].copy()
    global_best_cost = np.inf

    # PSO parameters
    w = 0.729   # inertia
    c1 = 1.494  # cognitive
    c2 = 1.494  # social

    convergence = []

    for it in range(n_iters):
        for i in range(n_particles):
            cost = objective(positions[i], *args)
            if cost < personal_best_cost[i]:
                personal_best_cost[i] = cost
                personal_best_pos[i] = positions[i].copy()
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_pos = positions[i].copy()

        convergence.append(global_best_cost)

        # Update
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        velocities = (w * velocities +
                     c1 * r1 * (personal_best_pos - positions) +
                     c2 * r2 * (global_best_pos - positions))
        positions = positions + velocities

        # Clip to bounds
        positions = np.clip(positions, lo, hi)

    # Wrap as result-like object
    class PSOResult:
        def __init__(self, x, fun, nfev, convergence):
            self.x = x
            self.fun = fun
            self.nfev = nfev
            self.nit = n_iters
            self.convergence = convergence
    return PSOResult(global_best_pos, global_best_cost,
                     n_particles * n_iters, convergence)


# ============================================================================
# MAIN ROUTINES
# ============================================================================

def run_multi_direction_optimization(exp_data):
    """
    Run multi-direction optimization using all 3 directions simultaneously.
    """
    print("=" * 70)
    print("MULTI-DIRECTION INVERSE IDENTIFICATION")
    print("=" * 70)

    Q_inf, b_iso = fit_voce_for_iso(exp_data, '00')
    fixed_iso = (Q_inf, b_iso)
    print(f"\n  Fixed isotropic: Q_inf = {Q_inf:.2f}, b = {b_iso:.2f}")

    x0 = np.array([270.0, 5000.0, 50.0, 1000.0, 10.0])
    bounds_low  = [150.0, 500.0,  5.0,  100.0,  1.0]
    bounds_high = [400.0, 50000.0, 500.0, 20000.0, 200.0]
    bounds_de = list(zip(bounds_low, bounds_high))

    results = {}

    # --- Single-direction (0° only, baseline) ---
    print("\n--- Single-Direction Optimization (0° only) ---")
    t0 = time.time()
    res_0 = run_nelder_mead(single_direction_objective, x0,
                            args=(exp_data['00'], fixed_iso),
                            bounds_low=bounds_low, bounds_high=bounds_high)
    t_0 = time.time() - t0
    results['single_0deg'] = {
        'params': res_0.x.tolist(), 'cost': float(res_0.fun),
        'time': t_0, 'nfev': res_0.nfev
    }
    print(f"  NRMSE = {res_0.fun:.6f}, time = {t_0:.1f}s")
    print(f"  σ0={res_0.x[0]:.1f}, C1={res_0.x[1]:.1f}, γ1={res_0.x[2]:.1f}, "
          f"C2={res_0.x[3]:.1f}, γ2={res_0.x[4]:.1f}")

    # --- Multi-direction (equal weights) ---
    print("\n--- Multi-Direction Optimization (equal weights) ---")
    t0 = time.time()
    res_eq = run_differential_evolution(
        multi_direction_objective, bounds_de,
        args=(exp_data, fixed_iso, {'00': 1.0, '45': 1.0, '90': 1.0}))
    t_eq = time.time() - t0
    results['multi_equal'] = {
        'params': res_eq.x.tolist(), 'cost': float(res_eq.fun),
        'time': t_eq, 'nfev': res_eq.nfev
    }
    print(f"  NRMSE = {res_eq.fun:.6f}, time = {t_eq:.1f}s")
    print(f"  σ0={res_eq.x[0]:.1f}, C1={res_eq.x[1]:.1f}, γ1={res_eq.x[2]:.1f}, "
          f"C2={res_eq.x[3]:.1f}, γ2={res_eq.x[4]:.1f}")

    # --- Multi-direction (weighted surrogate: more on 0°) ---
    print("\n--- Multi-Direction Optimization (weighted surrogate: 0°=2, 45°=1, 90°=1) ---")
    t0 = time.time()
    res_w = run_differential_evolution(
        multi_direction_objective, bounds_de,
        args=(exp_data, fixed_iso, SURROGATE_WEIGHTED_WEIGHTS))
    t_w = time.time() - t0
    results['multi_weighted'] = {
        'params': res_w.x.tolist(), 'cost': float(res_w.fun),
        'time': t_w, 'nfev': res_w.nfev
    }
    print(f"  NRMSE = {res_w.fun:.6f}, time = {t_w:.1f}s")

    # Evaluate all results across all 3 directions
    print("\n--- Cross-Direction Performance ---")
    print(f"  {'Strategy':>25s}  {'NRMSE 0°':>10s}  {'NRMSE 45°':>10s}  {'NRMSE 90°':>10s}  {'Average':>10s}")
    print(f"  {'-'*70}")

    for name, res_key in [('Single (0° only)', 'single_0deg'),
                           ('Multi (equal)', 'multi_equal'),
                           ('Multi (weighted)', 'multi_weighted')]:
        p = results[res_key]['params']
        e0 = single_direction_objective(p, exp_data['00'], fixed_iso)
        e45 = single_direction_objective(p, exp_data['45'], fixed_iso) if '45' in exp_data else float('nan')
        e90 = single_direction_objective(p, exp_data['90'], fixed_iso) if '90' in exp_data else float('nan')
        avg = np.nanmean([e0, e45, e90])
        results[res_key]['nrmse_per_dir'] = {'00': e0, '45': e45, '90': e90}
        print(f"  {name:>25s}  {e0:>10.6f}  {e45:>10.6f}  {e90:>10.6f}  {avg:>10.6f}")

    return results, fixed_iso


def run_algorithm_comparison(exp_data):
    """
    Compare 4 optimization algorithms on the same problem.
    """
    print("\n" + "=" * 70)
    print("OPTIMIZATION ALGORITHM COMPARISON")
    print("=" * 70)

    Q_inf, b_iso = fit_voce_for_iso(exp_data, '00')
    fixed_iso = (Q_inf, b_iso)

    x0 = np.array([270.0, 5000.0, 50.0, 1000.0, 10.0])
    bounds_low  = [150.0, 500.0,  5.0,  100.0,  1.0]
    bounds_high = [400.0, 50000.0, 500.0, 20000.0, 200.0]
    bounds_de = list(zip(bounds_low, bounds_high))

    # Use multi-direction objective with equal weights
    weights = {'00': 1.0, '45': 1.0, '90': 1.0}
    args = (exp_data, fixed_iso, weights)

    algorithms = {}

    # 1. Nelder-Mead
    print("\n[1] Nelder-Mead Simplex...")
    t0 = time.time()
    res_nm = run_nelder_mead(multi_direction_objective, x0, args=args,
                             bounds_low=bounds_low, bounds_high=bounds_high)
    t_nm = time.time() - t0
    algorithms['Nelder-Mead'] = {
        'params': res_nm.x.tolist(), 'cost': float(res_nm.fun),
        'time': t_nm, 'nfev': res_nm.nfev, 'nit': res_nm.nit
    }
    print(f"  NRMSE={res_nm.fun:.6f}, nfev={res_nm.nfev}, time={t_nm:.2f}s")

    # 2. Differential Evolution
    print("\n[2] Differential Evolution...")
    t0 = time.time()
    res_de = run_differential_evolution(multi_direction_objective, bounds_de, args=args)
    t_de = time.time() - t0
    algorithms['Diff. Evolution'] = {
        'params': res_de.x.tolist(), 'cost': float(res_de.fun),
        'time': t_de, 'nfev': res_de.nfev, 'nit': res_de.nit
    }
    print(f"  NRMSE={res_de.fun:.6f}, nfev={res_de.nfev}, time={t_de:.2f}s")

    # 3. Basin-Hopping
    print("\n[3] Basin-Hopping...")
    t0 = time.time()
    res_bh = run_basin_hopping(multi_direction_objective, x0, args=args,
                               bounds_low=bounds_low, bounds_high=bounds_high)
    t_bh = time.time() - t0
    algorithms['Basin-Hopping'] = {
        'params': res_bh.x.tolist(), 'cost': float(res_bh.fun),
        'time': t_bh, 'nfev': getattr(res_bh, 'nfev', 0),
        'nit': getattr(res_bh, 'nit', 0)
    }
    print(f"  NRMSE={res_bh.fun:.6f}, time={t_bh:.2f}s")

    # 4. PSO
    print("\n[4] Particle Swarm Optimization...")
    t0 = time.time()
    res_pso = run_pso(multi_direction_objective, bounds_low, bounds_high,
                      args=args, n_particles=30, n_iters=150)
    t_pso = time.time() - t0
    algorithms['PSO'] = {
        'params': res_pso.x.tolist(), 'cost': float(res_pso.fun),
        'time': t_pso, 'nfev': res_pso.nfev, 'nit': res_pso.nit,
        'convergence': [float(c) for c in res_pso.convergence]
    }
    print(f"  NRMSE={res_pso.fun:.6f}, nfev={res_pso.nfev}, time={t_pso:.2f}s")

    # Summary table
    print(f"\n{'Algorithm':>18s}  {'NRMSE':>10s}  {'Func Evals':>10s}  {'Time (s)':>10s}")
    print(f"{'-'*55}")
    for name, data in sorted(algorithms.items(), key=lambda x: x[1]['cost']):
        print(f"{name:>18s}  {data['cost']:>10.6f}  {data['nfev']:>10d}  {data['time']:>10.2f}")

    return algorithms


# ============================================================================
# PLOTTING
# ============================================================================

def plot_multi_direction_comparison(results, exp_data, fixed_iso):
    """Plot experimental vs model for each optimization strategy."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    dir_colors = {'00': '#1f77b4', '45': '#ff7f0e', '90': '#2ca02c'}
    strategies = [
        ('single_0deg', 'Single Dir (0° only)'),
        ('multi_equal', 'Multi-Dir (equal weights)'),
        ('multi_weighted', 'Multi-Dir (weighted)')
    ]

    for col, (key, title) in enumerate(strategies):
        params = results[key]['params']
        Q_inf, b_iso = fixed_iso

        for row, direction in enumerate(['00', '45', '90']):
            ax = axes[row][col]
            if direction in exp_data:
                ep = exp_data[direction]['plastic_strain']
                ts = exp_data[direction]['true_stress']
                ax.scatter(ep, ts, s=2, alpha=0.2, color='gray', label='Experimental')

                ep_fit = np.linspace(ep.min(), ep.max(), 300)
                ts_pred = combined_hardening_stress(
                    ep_fit, params[0], params[1], params[2],
                    params[3], params[4], Q_inf, b_iso)
                nrmse = results[key].get('nrmse_per_dir', {}).get(direction, 0)
                ax.plot(ep_fit, ts_pred, 'r-', linewidth=2,
                       label=f'Model (NRMSE={nrmse:.4f})')

            ax.set_xlabel('Plastic Strain')
            ax.set_ylabel('True Stress (MPa)')
            if row == 0:
                ax.set_title(title)
            ax.text(0.02, 0.95, f'{int(direction)}°', transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='top')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_multi_direction_optimization.png'))
    plt.close()
    print("  Saved: fig_multi_direction_optimization.png")


def plot_algorithm_comparison(algorithms):
    """Plot algorithm comparison bar chart and PSO convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = list(algorithms.keys())
    costs = [algorithms[n]['cost'] for n in names]
    times = [algorithms[n]['time'] for n in names]
    nfevs = [algorithms[n]['nfev'] for n in names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Cost comparison
    ax = axes[0]
    bars = ax.bar(names, costs, color=colors[:len(names)])
    ax.set_ylabel('NRMSE')
    ax.set_title('Final Cost Function')
    ax.tick_params(axis='x', rotation=30)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
               f'{c:.5f}', ha='center', fontsize=9)

    # Time comparison
    ax = axes[1]
    ax.bar(names, times, color=colors[:len(names)])
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Computation Time')
    ax.tick_params(axis='x', rotation=30)

    # Function evaluations
    ax = axes[2]
    ax.bar(names, nfevs, color=colors[:len(names)])
    ax.set_ylabel('Function Evaluations')
    ax.set_title('Total Function Evaluations')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_algorithm_comparison.png'))
    plt.close()
    print("  Saved: fig_algorithm_comparison.png")

    # PSO convergence
    if 'PSO' in algorithms and 'convergence' in algorithms['PSO']:
        fig, ax = plt.subplots(figsize=(10, 6))
        conv = algorithms['PSO']['convergence']
        ax.semilogy(range(1, len(conv)+1), conv, 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Cost (NRMSE)')
        ax.set_title('PSO Convergence History')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig_pso_convergence.png'))
        plt.close()
        print("  Saved: fig_pso_convergence.png")


def plot_parameter_sensitivity(exp_data, best_params, fixed_iso):
    """
    One-at-a-time parameter sensitivity analysis around the best solution.
    """
    param_names = ['σ₀', 'C₁', 'γ₁', 'C₂', 'γ₂']
    n_pts = 30

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for i, (name, ax) in enumerate(zip(param_names, axes)):
        base_val = best_params[i]
        scan_range = np.linspace(base_val * 0.5, base_val * 1.5, n_pts)
        costs = []

        for val in scan_range:
            test_params = list(best_params)
            test_params[i] = val
            cost = multi_direction_objective(
                test_params, exp_data, fixed_iso,
                {'00': 1.0, '45': 1.0, '90': 1.0})
            costs.append(cost)

        ax.plot(scan_range, costs, 'b-', linewidth=2)
        ax.axvline(base_val, color='r', linestyle='--', alpha=0.7, label='Optimal')
        ax.set_xlabel(name)
        ax.set_ylabel('NRMSE')
        ax.set_title(f'Sensitivity: {name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_parameter_sensitivity.png'))
    plt.close()
    print("  Saved: fig_parameter_sensitivity.png")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_all_results(multi_results, algo_results, fixed_iso):
    """Save all results to JSON and text files."""
    all_results = {
        'multi_direction': multi_results,
        'algorithm_comparison': algo_results,
        'fixed_isotropic': {'Q_inf': fixed_iso[0], 'b_iso': fixed_iso[1]}
    }

    with open(os.path.join(OUTPUT_DIR, 'optimization_comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  Saved: optimization_comparison_results.json")

    with open(os.path.join(OUTPUT_DIR, 'optimization_comparison_summary.txt'), 'w') as f:
        f.write("MULTI-DIRECTION OPTIMIZATION & ALGORITHM COMPARISON\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. MULTI-DIRECTION OPTIMIZATION\n")
        f.write("-" * 40 + "\n")
        f.write("Weighted surrogate study uses 0°:45°:90° = 2:1:1.\n")
        f.write("This is distinct from the Abaqus three-direction script, which uses 1:2:1.\n")
        for key, name in [('single_0deg', 'Single (0° only)'),
                          ('multi_equal', 'Multi (equal)'),
                  ('multi_weighted', 'Multi (weighted surrogate)')]:
            if key in multi_results:
                r = multi_results[key]
                p = r['params']
                f.write(f"\n  {name}:\n")
                f.write(f"    σ0={p[0]:.2f}, C1={p[1]:.2f}, γ1={p[2]:.2f}, "
                       f"C2={p[3]:.2f}, γ2={p[4]:.2f}\n")
                f.write(f"    Cost={r['cost']:.6f}, Time={r['time']:.1f}s\n")
                if 'nrmse_per_dir' in r:
                    nrmse = r['nrmse_per_dir']
                    f.write(f"    Per-dir NRMSE: 0°={nrmse['00']:.6f}, "
                           f"45°={nrmse['45']:.6f}, 90°={nrmse['90']:.6f}\n")

        f.write("\n\n2. ALGORITHM COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Algorithm':>18s}  {'NRMSE':>10s}  {'Func Evals':>10s}  {'Time (s)':>10s}\n")
        for name, data in sorted(algo_results.items(), key=lambda x: x[1]['cost']):
            f.write(f"  {name:>18s}  {data['cost']:>10.6f}  "
                   f"{data['nfev']:>10d}  {data['time']:>10.2f}\n")

    print("  Saved: optimization_comparison_summary.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading experimental data...")
    exp_data = load_all_experimental()
    print(f"  Loaded {len(exp_data)} directions: {list(exp_data.keys())}")
    for d, data in exp_data.items():
        print(f"    {d}°: {len(data['plastic_strain'])} points")

    # Multi-direction optimization
    multi_results, fixed_iso = run_multi_direction_optimization(exp_data)

    # Algorithm comparison
    algo_results = run_algorithm_comparison(exp_data)

    # Plots
    print("\nGenerating plots...")
    plot_multi_direction_comparison(multi_results, exp_data, fixed_iso)
    plot_algorithm_comparison(algo_results)

    # Get best params for sensitivity
    best_key = min(algo_results, key=lambda k: algo_results[k]['cost'])
    best_params = algo_results[best_key]['params']
    plot_parameter_sensitivity(exp_data, best_params, fixed_iso)

    # Save
    print("\nSaving results...")
    save_all_results(multi_results, algo_results, fixed_iso)

    print("\n" + "=" * 70)
    print("MULTI-DIRECTION OPTIMIZATION & ALGORITHM COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
