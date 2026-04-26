#!/usr/bin/env python3
"""
=============================================================================
Mesh Convergence Study — Plot Results (runs locally on Mac)
=============================================================================
Reads the CSV files produced by mesh_convergence_extract.py and generates:
  1. fig_mesh_convergence.png — 4-panel convergence study figure

Requires: mesh_convergence_results.csv, mesh_0p5_stress_strain.csv,
          mesh_1p0_stress_strain.csv, mesh_2p0_stress_strain.csv

Also loads experimental data for comparison (same as compare_sim_exp.py).
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Experimental data loading (from stress-00-*.csv files) ──────────────

def load_experimental():
    """Load stress-00-01/02/03.csv, return list of (eng_strain, eng_stress)."""
    curves = []
    for s in ['01', '02', '03']:
        fp = os.path.join(BASE_DIR, f'stress-00-{s}.csv')
        if not os.path.exists(fp):
            continue
        data = np.genfromtxt(fp, delimiter=',', skip_header=1)
        if data.ndim == 2 and data.shape[1] >= 4:
            eng_strain = data[:, 3]  # Strain column
            eng_stress = data[:, 2]  # Stress column
            curves.append((eng_strain, eng_stress))
    return curves


def eng_to_true(eng_strain, eng_stress):
    mask = (eng_strain > 0) & (eng_stress > 0)
    e = eng_strain[mask]
    s = eng_stress[mask]
    true_strain = np.log(1 + e)
    true_stress = s * (1 + e)
    return true_strain, true_stress


def get_exp_true_mean():
    """Get mean experimental true stress-strain for 0° direction."""
    curves = load_experimental()
    if not curves:
        return None, None
    true_curves = [eng_to_true(e, s) for e, s in curves]
    # Interpolate to common strain grid
    max_strain = min(tc[0][-1] for tc in true_curves if len(tc[0]) > 0)
    common_strain = np.linspace(0.002, min(max_strain, 0.22), 500)
    interp_stress = []
    for ts, tsig in true_curves:
        if len(ts) > 10:
            interp_stress.append(np.interp(common_strain, ts, tsig))
    if not interp_stress:
        return None, None
    mean_stress = np.mean(interp_stress, axis=0)
    return common_strain, mean_stress


# ── Load mesh convergence data ──────────────────────────────────────────

def load_summary():
    """Load mesh_convergence_results.csv (handles missing wall_time)."""
    fp = os.path.join(BASE_DIR, 'mesh_convergence_results.csv')
    # Use genfromtxt to handle missing/empty values gracefully
    data = np.genfromtxt(fp, delimiter=',', skip_header=1, filling_values=np.nan)
    return {
        'mesh_size': data[:, 0],
        'n_nodes': data[:, 1].astype(int),
        'n_elements': data[:, 2].astype(int),
        'n_frames': data[:, 3].astype(int),
        'final_S22': data[:, 4],
        'final_LE22': data[:, 5],
        'max_PEEQ': data[:, 6],
        'max_Mises': data[:, 7],
    }


def load_stress_strain(mesh_size):
    """Load per-mesh stress-strain CSV."""
    tag = f"{mesh_size:.1f}".replace('.', 'p')
    fp = os.path.join(BASE_DIR, f'mesh_{tag}_stress_strain.csv')
    if not os.path.exists(fp):
        return None
    data = np.loadtxt(fp, delimiter=',', skiprows=1)
    return {
        'time': data[:, 1],
        'S22': data[:, 2],
        'LE22': data[:, 3],
        'RF2': data[:, 4],
        'U2': data[:, 5],
    }


# ── Compute NRMSE against experiment ────────────────────────────────────

def compute_nrmse(exp_strain, exp_stress, sim_strain, sim_stress):
    """Interpolate sim onto exp grid and compute NRMSE."""
    valid = (exp_strain >= sim_strain.min()) & (exp_strain <= sim_strain.max())
    es = exp_strain[valid]
    ex = exp_stress[valid]
    sim_interp = np.interp(es, sim_strain, sim_stress)
    rmse = np.sqrt(np.mean((sim_interp - ex)**2))
    stress_range = ex.max() - ex.min()
    nrmse = rmse / stress_range if stress_range > 0 else rmse
    return rmse, nrmse


# ── Main plot ───────────────────────────────────────────────────────────

def main():
    print("Mesh Convergence Study — Plotting\n")

    summary = load_summary()
    mesh_sizes = summary['mesh_size']
    n_elements = summary['n_elements']

    # Load stress-strain curves
    ss_curves = {}
    for ms in mesh_sizes:
        ss_curves[ms] = load_stress_strain(ms)

    # Load experimental mean
    exp_strain, exp_stress = get_exp_true_mean()

    # Compute NRMSE for each mesh
    nrmse_vals = []
    rmse_vals = []
    for ms in mesh_sizes:
        ss = ss_curves[ms]
        if ss is not None and exp_strain is not None:
            rmse, nrmse = compute_nrmse(exp_strain, exp_stress,
                                        ss['LE22'], ss['S22'])
            rmse_vals.append(rmse)
            nrmse_vals.append(nrmse)
            print(f"  mesh={ms:.1f}mm: {int(summary['n_elements'][list(mesh_sizes).index(ms)])} elems, "
                  f"RMSE={rmse:.2f} MPa, NRMSE={nrmse:.4f}, "
                  f"final S22={summary['final_S22'][list(mesh_sizes).index(ms)]:.2f} MPa")
        else:
            rmse_vals.append(np.nan)
            nrmse_vals.append(np.nan)

    # ── Figure: 4-panel convergence study ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ['#2196F3', '#4CAF50', '#FF9800']  # blue, green, orange
    labels = [f'{ms:.1f} mm ({int(ne)} elem)' for ms, ne in zip(mesh_sizes, n_elements)]

    # Panel (a): Overlaid stress-strain curves
    ax = axes[0, 0]
    if exp_strain is not None:
        ax.plot(exp_strain, exp_stress, 'k-', lw=2, label='Experiment (0° mean)', zorder=5)
    for i, ms in enumerate(mesh_sizes):
        ss = ss_curves[ms]
        if ss is not None:
            ax.plot(ss['LE22'], ss['S22'], '-', color=colors[i], lw=1.5,
                    label=labels[i])
    ax.set_xlabel('True strain (LE22)')
    ax.set_ylabel('True stress S22 (MPa)')
    ax.set_title('(a) Stress–strain curves')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel (b): Force-displacement curves
    ax = axes[0, 1]
    for i, ms in enumerate(mesh_sizes):
        ss = ss_curves[ms]
        if ss is not None:
            # Force = RF2_sum (quarter model, so multiply by 2 for half-model symmetry)
            force = np.abs(ss['RF2']) * 2  # N (full width reaction)
            ax.plot(ss['U2'], force, '-', color=colors[i], lw=1.5, label=labels[i])
    ax.set_xlabel('Displacement U2 (mm)')
    ax.set_ylabel('Force (N)')
    ax.set_title('(b) Force–displacement curves')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel (c): Convergence of final stress with mesh refinement
    ax = axes[1, 0]
    ax.plot(n_elements, summary['final_S22'], 'o-', color='#D32F2F', lw=2, ms=8)
    for i, (ne, s22) in enumerate(zip(n_elements, summary['final_S22'])):
        ax.annotate(f'{s22:.1f} MPa', (ne, s22),
                    textcoords='offset points', xytext=(10, 5), fontsize=8)
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Final S22 (MPa)')
    ax.set_title('(c) Stress convergence')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Panel (d): NRMSE convergence
    ax = axes[1, 1]
    valid_nrmse = [n for n in nrmse_vals if not np.isnan(n)]
    if valid_nrmse:
        ax.plot(n_elements, nrmse_vals, 's-', color='#7B1FA2', lw=2, ms=8)
        for i, (ne, nr) in enumerate(zip(n_elements, nrmse_vals)):
            if not np.isnan(nr):
                ax.annotate(f'{nr:.4f}', (ne, nr),
                            textcoords='offset points', xytext=(10, 5), fontsize=8)
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('NRMSE vs. experiment')
    ax.set_title('(d) NRMSE convergence')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Mesh Convergence Study — CPS4R Quarter-Symmetry Model\n'
                 '(0.5, 1.0, 2.0 mm element sizes)', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(BASE_DIR, 'output', 'fig_mesh_convergence.png')
    os.makedirs(os.path.join(BASE_DIR, 'output'), exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSaved: {out_path}")
    plt.close()

    # ── Print summary table ─────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("MESH CONVERGENCE SUMMARY")
    print("=" * 75)
    print(f"{'Mesh (mm)':<10} {'Elements':<10} {'Nodes':<10} "
          f"{'Final S22':<12} {'NRMSE':<10} {'RMSE (MPa)':<10}")
    print("-" * 75)
    for i, ms in enumerate(mesh_sizes):
        idx = list(mesh_sizes).index(ms)
        print(f"{ms:<10.1f} {int(n_elements[idx]):<10} "
              f"{int(summary['n_nodes'][idx]):<10} "
              f"{summary['final_S22'][idx]:<12.2f} "
              f"{nrmse_vals[i]:<10.4f} {rmse_vals[i]:<10.2f}")

    # Richardson extrapolation
    if len(valid_nrmse) >= 3:
        s = summary['final_S22']
        h = mesh_sizes
        r = h[1] / h[0]  # refinement ratio
        if abs(s[1] - s[0]) > 1e-10:
            p_est = np.log(abs((s[2] - s[1]) / (s[1] - s[0]))) / np.log(r)
            s_exact = s[0] + (s[0] - s[1]) / (r**p_est - 1)
            print(f"\nRichardson extrapolation (p={p_est:.2f}): S22_exact ≈ {s_exact:.2f} MPa")
            for i, ms in enumerate(mesh_sizes):
                err = abs(s[i] - s_exact) / abs(s_exact) * 100
                print(f"  mesh={ms:.1f}mm: relative error = {err:.3f}%")
        print(f"\n1.0mm mesh used in study — verify convergence is adequate.")

    print("\nDone.")


if __name__ == '__main__':
    main()
