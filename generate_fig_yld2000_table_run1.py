"""
Generate figures for the run-1 yld2000_table FEA-based identification:
  Fig A: convergence trace (per-direction NRMSE + weighted combined vs iter)
  Fig B: parameter trajectories (5 panels: sigma0, C1, gamma1, C2, gamma2)
  Fig C: simulated vs experimental flow curves (requires sim CSVs from lab PC)

Run:
    python generate_fig_yld2000_table_run1.py
"""
from __future__ import print_function
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'output', 'yld2000_table_run1')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'output', 'figures_yld2000_table_run1')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Read convergence history
# ---------------------------------------------------------------------------
hist_path = os.path.join(RUN_DIR, 'convergence_history_multidir_stress.csv')
data = {}
with open(hist_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k, v in row.items():
            data.setdefault(k, []).append(v)

ITER = np.array([int(x) for x in data['Iteration']])
S0   = np.array([float(x) for x in data['sigma0']])
C1   = np.array([float(x) for x in data['C1']])
G1   = np.array([float(x) for x in data['gamma1']])
C2   = np.array([float(x) for x in data['C2']])
G2   = np.array([float(x) for x in data['gamma2']])
NW   = np.array([float(x) for x in data['NRMSE_weighted']])
N00  = np.array([float(x) for x in data['NRMSE_00']])
N45  = np.array([float(x) for x in data['NRMSE_45']])
N90  = np.array([float(x) for x in data['NRMSE_90']])

i_best = int(np.argmin(NW))
running_best = np.minimum.accumulate(NW)

# ---------------------------------------------------------------------------
# Fig A: convergence
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(ITER, N00 * 100, 'o-', ms=4, lw=1.0, color='#1f77b4', label=r'0$^\circ$ NRMSE')
ax.plot(ITER, N45 * 100, 's-', ms=4, lw=1.0, color='#d62728', label=r'45$^\circ$ NRMSE (w=2)')
ax.plot(ITER, N90 * 100, '^-', ms=4, lw=1.0, color='#2ca02c', label=r'90$^\circ$ NRMSE')
ax.plot(ITER, NW  * 100, 'k-', lw=1.8, label='Weighted combined')
ax.plot(ITER, running_best * 100, 'k--', lw=1.0, alpha=0.6, label='Running best')
ax.axvline(ITER[i_best], color='gold', lw=2.5, alpha=0.5, zorder=0,
           label='Best (iter %d)' % ITER[i_best])
ax.set_xlabel('Iteration')
ax.set_ylabel('NRMSE [%]')
ax.set_title('Yld2000-2d FEA-based identification: convergence\n'
             '(Nelder-Mead, $\\sigma_0$, $C_1$, $\\gamma_1$, $C_2$, $\\gamma_2$ live; '
             '$Q_\\infty$, $b$, $\\alpha_i$ fixed)')
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(N45.max(), N90.max(), N00.max()) * 105)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_convergence.pdf'))
fig.savefig(os.path.join(FIG_DIR, 'fig_convergence.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig B: parameter trajectories
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.8), sharex=True)
labels = [r'$\sigma_0$ [MPa]', r'$C_1$ [MPa]', r'$\gamma_1$',
          r'$C_2$ [MPa]', r'$\gamma_2$']
arrs = [S0, C1, G1, C2, G2]
for ax, arr, lab in zip(axes, arrs, labels):
    ax.plot(ITER, arr, '-', lw=1.0, color='#444')
    ax.scatter(ITER[i_best], arr[i_best], color='gold', edgecolor='k',
               zorder=5, s=60, label='Best')
    ax.set_xlabel('Iter')
    ax.set_ylabel(lab)
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=8, loc='best')
fig.suptitle('Parameter trajectories along the simplex')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_parameters.pdf'))
fig.savefig(os.path.join(FIG_DIR, 'fig_parameters.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig C: identified flow curve (analytical Voce + Chaboche*2 monotonic eq.)
# ---------------------------------------------------------------------------
from material_constants import Q_INF, B_ISO
from hardening_table import build_flow_curve

s0_b = S0[i_best]; C1b = C1[i_best]; g1b = G1[i_best]
C2b = C2[i_best]; g2b = G2[i_best]
kappa, sigy = build_flow_curve(s0_b, Q_INF, B_ISO, C1b, g1b, C2b, g2b,
                               n_points=200, eps_max=0.30)
# Decompose contributions
voce = Q_INF * (1.0 - np.exp(-B_ISO * kappa))
chab1 = (C1b / g1b) * (1.0 - np.exp(-g1b * kappa))
chab2 = (C2b / g2b) * (1.0 - np.exp(-g2b * kappa))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(kappa, sigy, 'k-', lw=2.0, label=r'$\sigma_y(\kappa)$ identified')
ax.plot(kappa, np.full_like(kappa, s0_b), '--', color='#888', lw=1.0,
        label=r'$\sigma_0=%.1f$ MPa' % s0_b)
ax.plot(kappa, s0_b + voce,  '-.', color='#1f77b4', lw=1.0,
        label=r'$+ Q_\infty(1-e^{-b\kappa})$')
ax.plot(kappa, s0_b + voce + chab1, ':',  color='#d62728', lw=1.5,
        label=r'$+\,(C_1/\gamma_1)(1-e^{-\gamma_1\kappa})$')
ax.set_xlabel(r'Equivalent plastic strain $\kappa$')
ax.set_ylabel('Flow stress [MPa]')
ax.set_title('Identified monotonic flow curve\n'
             '($C_1/\\gamma_1=%.1f$,  $C_2/\\gamma_2=%.1f$ MPa)'
             % (C1b/g1b, C2b/g2b))
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_flow_curve.pdf'))
fig.savefig(os.path.join(FIG_DIR, 'fig_flow_curve.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig D (optional): sim-vs-exp if best-iter sim CSVs are available
# ---------------------------------------------------------------------------
sim_csvs = {
    0:  os.path.join(RUN_DIR, 'sim_best_00.csv'),
    45: os.path.join(RUN_DIR, 'sim_best_45.csv'),
    90: os.path.join(RUN_DIR, 'sim_best_90.csv'),
}
if all(os.path.exists(p) for p in sim_csvs.values()):
    # The first lab-PC extraction used HALF_WIDTH=6.25, HALF_LENGTH=12.5,
    # THICKNESS=1.0 in extract_yld2000_table_best.py, but the actual model
    # uses 10.0 / 40.0 / 1.5 in optimize_hardening_multidir.py.
    # Recover correct eng quantities and re-derive true stress/strain.
    SIM_AREA_FIX    = (2 * 10.0 * 1.5) / (2 * 6.25 * 1.0)   # 30/12.5 = 2.4
    SIM_LENGTH_FIX  = 40.0 / 12.5                            # 3.2

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    nrmse_per = {0: N00[i_best], 45: N45[i_best], 90: N90[i_best]}
    for ax, ang in zip(axes, (0, 45, 90)):
        # Experimental specimens — CSV columns: Time, Load, Stress, Strain (engineering)
        exp_strain_max = 0.0
        for k in (1, 2, 3):
            exp = os.path.join(os.path.dirname(__file__),
                               'stress-%02d-%02d.csv' % (ang, k))
            if not os.path.exists(exp):
                continue
            arr = np.loadtxt(exp, delimiter=',', skiprows=1)
            eng_stress = arr[:, 2]
            eng_strain = arr[:, 3]
            uts = int(np.argmax(eng_stress))
            eng_stress = eng_stress[:uts + 1]
            eng_strain = eng_strain[:uts + 1]
            m = (eng_strain > 0) & (eng_stress > 0)
            true_strain = np.log(1.0 + eng_strain[m])
            true_stress = eng_stress[m] * (1.0 + eng_strain[m])
            exp_strain_max = max(exp_strain_max, true_strain.max())
            ax.plot(true_strain, true_stress, '-', color='#888', lw=0.9,
                    label='Exp' if k == 1 else None)

        # FEA — undo wrong geometry, rebuild eng-level, redo true conversion
        sim = np.loadtxt(sim_csvs[ang], delimiter=',', skiprows=1)
        ts_rec, tsig_rec = sim[:, 0], sim[:, 1]
        eps_eng_rec = np.exp(ts_rec) - 1.0
        sig_eng_rec = tsig_rec / (1.0 + eps_eng_rec)
        eps_eng = eps_eng_rec / SIM_LENGTH_FIX
        sig_eng = sig_eng_rec / SIM_AREA_FIX
        ts = np.log(1.0 + eps_eng)
        tsig = sig_eng * (1.0 + eps_eng)

        sim_mask = ts <= exp_strain_max * 1.05
        ax.plot(ts[sim_mask], tsig[sim_mask], 'r-', lw=1.8, label='FEA (best)')
        ax.set_title(r'%d$^\circ$  (NRMSE=%.4f)' % (ang, nrmse_per[ang]))
        ax.set_xlabel('True strain')
        ax.set_xlim(0, exp_strain_max * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('True stress [MPa]')
    fig.suptitle('Yld2000-2d FEA at identified ($\\sigma_0$, $C_1$, $\\gamma_1$, $C_2$, $\\gamma_2$)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_sim_vs_exp.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'fig_sim_vs_exp.png'), dpi=200)
    plt.close(fig)
    print('Wrote sim-vs-exp figure.')
else:
    print('NOTE: sim_best_*.csv not found in %s' % RUN_DIR)
    print('      Run extract_yld2000_table_best.py on the lab PC and copy the CSVs back.')

print('\nFigures written to %s' % FIG_DIR)
print('Best parameters (iter %d, NRMSE=%.4f):' % (ITER[i_best], NW[i_best]))
print('  sigma0=%.3f  C1=%.3f  gamma1=%.3f  C2=%.3f  gamma2=%.3f'
      % (S0[i_best], C1[i_best], G1[i_best], C2[i_best], G2[i_best]))
