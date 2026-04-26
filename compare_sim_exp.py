#!/usr/bin/env python3
"""
Compare Abaqus simulation results with experimental data.
Generates publication-quality figures for thesis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
# Figures are referenced as `output/fig_*.png` from PhD_Thesis.md, so save
# them into the `output/` subdirectory rather than the workspace root.
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

E_YOUNG = 200000.0  # MPa
THICKNESS = 1.5     # mm
WIDTH = 20.0        # mm
AREA = WIDTH * THICKNESS  # 30 mm^2

# Hardening parameters used in the simulation (ORIGINAL)
SIGMA0 = 312.35
Q_INF = 335.16
B_ISO = 3.95
C1, GAMMA1 = 502.71, 499.72
C2, GAMMA2 = 100.37, 199.44

# Optimized parameters (from inverse identification)
OPT_SIGMA0 = 324.1330
OPT_C1, OPT_GAMMA1 = 109.4935, 691.6965
OPT_C2, OPT_GAMMA2 = 104.6664, 213.9800

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'font.family': 'serif',
})

COLORS = {'00': '#1f77b4', '45': '#ff7f0e', '90': '#2ca02c'}


def load_experimental():
    """Load all 0-degree experimental stress-strain data."""
    exp_data = {}
    for s in ['01', '02', '03']:
        stress_file = os.path.join(DATA_DIR, f'stress-00-{s}.csv')
        if os.path.exists(stress_file):
            df = pd.read_csv(stress_file)
            df.columns = ['Time', 'Load', 'Stress', 'Strain']
            exp_data[f'00-{s}'] = df
    return exp_data


def eng_to_true(eng_stress, eng_strain):
    """Convert engineering to true stress-strain."""
    true_strain = np.log(1 + eng_strain)
    true_stress = eng_stress * (1 + eng_strain)
    return true_stress, true_strain


def analytical_curve(eps_p, sigma0=SIGMA0, c1=C1, g1=GAMMA1, c2=C2, g2=GAMMA2):
    """Compute analytical monotonic combined hardening stress."""
    sigma = sigma0 + Q_INF * (1 - np.exp(-B_ISO * eps_p))
    sigma += (c1 / g1) * (1 - np.exp(-g1 * eps_p))
    sigma += (c2 / g2) * (1 - np.exp(-g2 * eps_p))
    return sigma


def get_exp_true_curves(exp_data):
    """Get individual + mean experimental true stress-strain."""
    curves = []
    for name, df in exp_data.items():
        eng_stress = df['Stress'].values
        eng_strain = df['Strain'].values
        mask = (eng_strain > 0) & (eng_stress > 0)
        ts, te = eng_to_true(eng_stress[mask], eng_strain[mask])
        curves.append((name, te, ts))

    # Interpolate to common grid for mean
    strain_common = np.linspace(0.002, 0.22, 500)
    stress_all = []
    for _, te, ts in curves:
        if len(te) > 10:
            interp = np.interp(strain_common, te, ts, left=np.nan, right=np.nan)
            stress_all.append(interp)

    stress_mean = np.nanmean(stress_all, axis=0) if stress_all else None
    stress_std = np.nanstd(stress_all, axis=0) if len(stress_all) > 1 else None

    return curves, strain_common, stress_mean, stress_std


def compute_metrics(strain_common, stress_mean, model_stress):
    """Compute RMSE, NRMSE, R² for a model vs experiment."""
    valid = ~np.isnan(stress_mean) & ~np.isnan(model_stress)
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    res = model_stress[valid] - stress_mean[valid]
    rmse = np.sqrt(np.mean(res ** 2))
    srange = stress_mean[valid].max() - stress_mean[valid].min()
    nrmse = rmse / srange if srange > 0 else np.nan
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((stress_mean[valid] - np.mean(stress_mean[valid])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, nrmse, r2


def main():
    # Load simulation data
    sim_ss = pd.read_csv(os.path.join(DATA_DIR, 'sim_stress_strain.csv'))
    sim_fd = pd.read_csv(os.path.join(DATA_DIR, 'sim_force_disp.csv'))

    # Load experimental data
    exp_data = load_experimental()

    # Precompute experimental curves
    curves, strain_common, stress_mean, stress_std = get_exp_true_curves(exp_data)

    # Analytical curves (original + optimized)
    eps_p = np.linspace(0, 0.45, 1000)
    sigma_orig = analytical_curve(eps_p)
    sigma_opt = analytical_curve(eps_p, OPT_SIGMA0, OPT_C1, OPT_GAMMA1, OPT_C2, OPT_GAMMA2)
    eps_total_orig = eps_p + sigma_orig / E_YOUNG
    eps_total_opt = eps_p + sigma_opt / E_YOUNG

    # Interpolate both models to common strain for metrics
    model_orig_interp = np.interp(strain_common, eps_total_orig, sigma_orig,
                                   left=np.nan, right=np.nan)
    model_opt_interp = np.interp(strain_common, eps_total_opt, sigma_opt,
                                  left=np.nan, right=np.nan)
    sim_interp = np.interp(strain_common, sim_ss['E22'], sim_ss['S22'],
                            left=np.nan, right=np.nan)

    rmse_orig, nrmse_orig, r2_orig = compute_metrics(strain_common, stress_mean, model_orig_interp)
    rmse_opt, nrmse_opt, r2_opt = compute_metrics(strain_common, stress_mean, model_opt_interp)
    rmse_fea, nrmse_fea, r2_fea = compute_metrics(strain_common, stress_mean, sim_interp)

    print("Metrics vs experiment (0-deg mean):")
    print(f"  Original analytical : RMSE={rmse_orig:.2f} MPa, NRMSE={nrmse_orig:.4f}, R²={r2_orig:.6f}")
    print(f"  Optimized analytical: RMSE={rmse_opt:.2f} MPa, NRMSE={nrmse_opt:.4f}, R²={r2_opt:.6f}")
    print(f"  FEA (original .inp) : RMSE={rmse_fea:.2f} MPa, NRMSE={nrmse_fea:.4f}, R²={r2_fea:.6f}")

    # ================================================================
    # Figure 1: Final comparison — True stress-strain
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Experimental individual curves
    for name, te, ts in curves:
        ax.plot(te, ts, color='#1f77b4', alpha=0.25, linewidth=0.8)

    # Experimental mean ± std
    valid = ~np.isnan(stress_mean)
    ax.plot(strain_common[valid], stress_mean[valid], 'b-', linewidth=2.5,
            label='Experiment (0° mean, n=3)')
    if stress_std is not None:
        ax.fill_between(strain_common[valid],
                        (stress_mean - stress_std)[valid],
                        (stress_mean + stress_std)[valid],
                        color='blue', alpha=0.1, label='Exp. ±1 std')

    # FEA result (from Abaqus .odb)
    sim_mask = sim_ss['S22'] > 0
    ax.plot(sim_ss['E22'][sim_mask], sim_ss['S22'][sim_mask], 'r--',
            linewidth=2.5, label=f'FEA original (NRMSE={nrmse_fea:.4f})')

    # Analytical original
    ax.plot(eps_total_orig, sigma_orig, color='darkred', linestyle=':',
            linewidth=1.8, alpha=0.7,
            label=f'Analytical original (NRMSE={nrmse_orig:.4f})')

    # Analytical optimized
    ax.plot(eps_total_opt, sigma_opt, color='green', linestyle='-.',
            linewidth=2.0,
            label=f'Analytical optimized (NRMSE={nrmse_opt:.4f})')

    ax.set_xlabel('True Strain')
    ax.set_ylabel('True Stress (MPa)')
    ax.set_title('Simulation vs Experiment: True Stress-Strain (0° RD)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 650)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_sim_vs_exp_true_stress.png'))
    plt.close()
    print("Saved: fig_sim_vs_exp_true_stress.png")

    # ================================================================
    # Figure 2: Engineering stress-strain
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, df in exp_data.items():
        mask = df['Stress'].values > 0
        ax.plot(df['Strain'].values[mask], df['Stress'].values[mask],
                color='#1f77b4', alpha=0.25, linewidth=0.8)

    # Exp mean (convert back to engineering)
    eng_strain_mean = np.exp(strain_common) - 1
    eng_stress_mean = stress_mean / (1 + eng_strain_mean)
    valid = ~np.isnan(eng_stress_mean)
    ax.plot(eng_strain_mean[valid], eng_stress_mean[valid], 'b-', linewidth=2.5,
            label='Experiment (0° mean)')

    # FEA engineering
    sim_mask = sim_fd['EngStress_MPa'] > 0
    ax.plot(sim_fd['EngStrain'][sim_mask], sim_fd['EngStress_MPa'][sim_mask],
            'r--', linewidth=2.5, label='FEA original')

    ax.set_xlabel('Engineering Strain')
    ax.set_ylabel('Engineering Stress (MPa)')
    ax.set_title('Simulation vs Experiment: Engineering Stress-Strain (0° RD)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 500)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_sim_vs_exp_eng_stress.png'))
    plt.close()
    print("Saved: fig_sim_vs_exp_eng_stress.png")

    # ================================================================
    # Figure 3: Residual — original vs optimized
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax_i, (label, model_interp, color) in enumerate([
        ('Original', model_orig_interp, 'red'),
        ('Optimized', model_opt_interp, 'green'),
    ]):
        ax = axes[ax_i]
        residual = model_interp - stress_mean
        valid = ~np.isnan(residual)
        ax.plot(strain_common[valid], residual[valid], color=color, linewidth=1.5)
        ax.fill_between(strain_common[valid], residual[valid], 0,
                        color=color, alpha=0.15)
        ax.axhline(y=0, color='k', linewidth=0.5)
        rmse_val = np.sqrt(np.nanmean(residual[valid] ** 2))
        ax.set_ylabel('Residual (MPa)')
        ax.set_title(f'{label} Parameters — RMSE = {rmse_val:.2f} MPa')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-40, 40)

    axes[1].set_xlabel('True Strain')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_sim_vs_exp_residual.png'))
    plt.close()
    print("Saved: fig_sim_vs_exp_residual.png")

    # ================================================================
    # Figure 4: Hardening decomposition — original vs optimized
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    eps_p_plot = np.linspace(0, 0.35, 500)

    for ax, title, s0, c1, g1, c2, g2 in [
        (ax1, 'Original Parameters', SIGMA0, C1, GAMMA1, C2, GAMMA2),
        (ax2, 'Optimized Parameters', OPT_SIGMA0, OPT_C1, OPT_GAMMA1, OPT_C2, OPT_GAMMA2),
    ]:
        iso = s0 + Q_INF * (1 - np.exp(-B_ISO * eps_p_plot))
        k1 = (c1 / g1) * (1 - np.exp(-g1 * eps_p_plot))
        k2 = (c2 / g2) * (1 - np.exp(-g2 * eps_p_plot))
        total = iso + k1 + k2

        ax.plot(eps_p_plot, total, 'k-', linewidth=2.5, label='Total')
        ax.plot(eps_p_plot, iso, 'b--', linewidth=1.5,
                label=f'Isotropic ($\\sigma_0$={s0:.1f})')
        ax.plot(eps_p_plot, k1, 'r-.', linewidth=1.5,
                label=f'KH1: C/g={c1/g1:.2f} MPa')
        ax.plot(eps_p_plot, k2, 'g:', linewidth=1.5,
                label=f'KH2: C/g={c2/g2:.2f} MPa')
        ax.set_xlabel('Plastic Strain $\\varepsilon^p$')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='center right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.35)

    ax1.set_ylabel('Stress (MPa)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_hardening_decomposition.png'))
    plt.close()
    print("Saved: fig_hardening_decomposition.png")

    # ================================================================
    # Figure 5: Parameter comparison table as figure
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    table_data = [
        ['Parameter', 'Original', 'Optimized', 'Change'],
        ['$\\sigma_0$ (MPa)', f'{SIGMA0:.2f}', f'{OPT_SIGMA0:.2f}',
         f'{(OPT_SIGMA0 - SIGMA0)/SIGMA0*100:+.1f}%'],
        ['$C_1$ (MPa)', f'{C1:.2f}', f'{OPT_C1:.2f}',
         f'{(OPT_C1 - C1)/C1*100:+.1f}%'],
        ['$\\gamma_1$', f'{GAMMA1:.2f}', f'{OPT_GAMMA1:.2f}',
         f'{(OPT_GAMMA1 - GAMMA1)/GAMMA1*100:+.1f}%'],
        ['$C_2$ (MPa)', f'{C2:.2f}', f'{OPT_C2:.2f}',
         f'{(OPT_C2 - C2)/C2*100:+.1f}%'],
        ['$\\gamma_2$', f'{GAMMA2:.2f}', f'{OPT_GAMMA2:.2f}',
         f'{(OPT_GAMMA2 - GAMMA2)/GAMMA2*100:+.1f}%'],
        ['$C_1/\\gamma_1$ (MPa)', f'{C1/GAMMA1:.3f}', f'{OPT_C1/OPT_GAMMA1:.3f}', ''],
        ['$C_2/\\gamma_2$ (MPa)', f'{C2/GAMMA2:.3f}', f'{OPT_C2/OPT_GAMMA2:.3f}', ''],
        ['$Q_\\infty$ (MPa)', f'{Q_INF:.2f}', f'{Q_INF:.2f}', 'fixed'],
        ['b', f'{B_ISO:.2f}', f'{B_ISO:.2f}', 'fixed'],
        ['NRMSE', f'{nrmse_orig:.4f}', f'{nrmse_opt:.4f}',
         f'{(nrmse_opt - nrmse_orig)/nrmse_orig*100:+.1f}%'],
        ['RMSE (MPa)', f'{rmse_orig:.2f}', f'{rmse_opt:.2f}', ''],
        ['R²', f'{r2_orig:.6f}', f'{r2_opt:.6f}', ''],
    ]

    table = ax.table(
        cellText=[row for row in table_data],
        cellLoc='center',
        loc='center',
        colWidths=[0.28, 0.2, 0.2, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Header styling
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#D6E4F0' if i % 2 == 0 else 'white'
        for j in range(4):
            table[i, j].set_facecolor(color)

    ax.set_title('Parameter Comparison: Original vs Inverse-Optimized',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_parameter_comparison.png'))
    plt.close()
    print("Saved: fig_parameter_comparison.png")

    print(f"\nDone — 5 final comparison figures generated.")


if __name__ == '__main__':
    main()
