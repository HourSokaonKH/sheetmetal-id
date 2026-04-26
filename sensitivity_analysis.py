#!/usr/bin/env python3
"""
=============================================================================
Monte Carlo Sensitivity & Uncertainty Analysis
=============================================================================
Propagates measurement uncertainty through the material characterization
pipeline to quantify confidence intervals on:
  - Lankford r-values
  - Hill'48 parameters
  - Hardening law parameters
  - Combined hardening parameters

Perturbation sources:
  - DIC strain measurement noise (±5%)
  - Yield stress determination (±σ from 3 specimens)
  - Young's modulus assumption (±5 GPa)

Author: PhD Candidate
Date:   2026
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os
import json

from anisotropy_reference import load_canonical_anisotropy, sample_direction_r_value

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_MONTE_CARLO = 2000   # Number of MC samples
SEED = 42

# Nominal material properties
E_NOMINAL = 200000.0    # MPa
E_STD = 5000.0          # ±5 GPa uncertainty
PLASTIC_STRAIN_THRESHOLD = 0.002

# DIC noise level (relative, ±5%)
DIC_NOISE_LEVEL = 0.05

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
    """Load legacy single-zone DIC data (UFreckles format with 8-line header)."""
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


def load_multizone_dic_data(specimen_name):
    """
    Load multi-zone DIC strain data from raw_data/<specimen>/strain_export/.
    Returns mean DataFrame across all zones with columns [Step, Eyy, Exx, Exy].
    """
    raw_dir = os.path.join(DATA_DIR, 'raw_data')
    export_dir = os.path.join(raw_dir, specimen_name, 'strain_export')
    
    zone_dfs = []
    z = 1
    while True:
        zfile = os.path.join(export_dir, f'{specimen_name}-zone{z:02d}-strains.csv')
        if not os.path.exists(zfile):
            break
        zone_dfs.append(pd.read_csv(zfile))
        z += 1
    
    if not zone_dfs:
        return None
    
    n_frames = min(len(df) for df in zone_dfs)
    mean_exx = np.mean([df['Exx'].values[:n_frames] for df in zone_dfs], axis=0)
    mean_eyy = np.mean([df['Eyy'].values[:n_frames] for df in zone_dfs], axis=0)
    mean_exy = np.mean([df['Exy'].values[:n_frames] for df in zone_dfs], axis=0)
    
    return pd.DataFrame({
        'Step': np.arange(1, n_frames + 1),
        'Eyy': mean_eyy, 'Exx': mean_exx, 'Exy': mean_exy,
    })


def smooth_strain(strain, window=31):
    if len(strain) < 5:
        return np.array(strain)
    if len(strain) < window:
        window = max(5, len(strain) // 2 * 2 + 1)
        if window % 2 == 0:
            window += 1
    return savgol_filter(strain, window, 3)


# ============================================================================
# CORE COMPUTATION FUNCTIONS
# ============================================================================

def compute_r_value(eyy, exx):
    """Compute r-value from DIC strains."""
    mask = np.abs(eyy) > 0.005
    if np.sum(mask) < 10:
        mask = np.abs(eyy) > 0.002
    if np.sum(mask) < 5:
        return np.nan
    eyy_p = eyy[mask]
    exx_p = exx[mask]
    coeffs = np.polyfit(eyy_p, exx_p, 1)
    slope = coeffs[0]
    return -slope / (1 + slope)


def compute_hill48(r0, r45, r90):
    """Compute Hill'48 parameters from r-values."""
    F = r0 / (r90 * (1 + r0))
    G = 1.0 / (1 + r0)
    H = r0 / (1 + r0)
    N = (r0 + r90) * (1 + 2*r45) / (2 * r90 * (1 + r0))

    R11 = 1.0
    R22 = np.sqrt(r90 * (1 + r0) / (r0 * (1 + r90)))
    R33 = np.sqrt(r90 * (1 + r0) / (r0 + r90))
    R12 = np.sqrt(3.0 * r90 * (1 + r0) / ((2*r45 + 1) * (r0 + r90)))

    return {'F': F, 'G': G, 'H': H, 'N': N,
            'R11': R11, 'R22': R22, 'R33': R33, 'R12': R12}


def voce_law(eps_p, sigma_y, Q, b):
    return sigma_y + Q * (1.0 - np.exp(-b * eps_p))


def fit_voce(plastic_strain, true_stress):
    """Fit Voce law and return parameters."""
    mask = plastic_strain >= PLASTIC_STRAIN_THRESHOLD
    if np.sum(mask) < 20:
        return None
    ep = plastic_strain[mask]
    ts = true_stress[mask]
    try:
        popt, _ = curve_fit(voce_law, ep, ts, p0=[250, 200, 10],
                           maxfev=10000,
                           bounds=([50, 10, 0.1], [1000, 1000, 200]))
        residuals = ts - voce_law(ep, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ts - np.mean(ts))**2)
        r2 = 1 - ss_res / ss_tot
        return {'sigma_y': popt[0], 'Q': popt[1], 'b': popt[2], 'R2': r2}
    except:
        return None


# ============================================================================
# MONTE CARLO ENGINE
# ============================================================================

def load_all_raw_data():
    """Load all raw data files for MC perturbation.
    Uses multi-zone DIC data from raw_data/*/strain_export/ when available."""
    raw = {
        'canonical_anisotropy': load_canonical_anisotropy(DATA_DIR)
    }
    if raw['canonical_anisotropy']['available']:
        print("  canonical anisotropy: loaded MATLAB multi-zone result files")

    for direction in ['00', '45', '90']:
        for specimen in ['01', '02', '03']:
            # Stress data
            sf = os.path.join(DATA_DIR, f'stress-{direction}-{specimen}.csv')
            if os.path.exists(sf):
                raw[f'stress_{direction}_{specimen}'] = load_stress_data(sf)

            # DIC data — prefer multi-zone
            specimen_name = f'{direction}-{specimen}'
            mz = load_multizone_dic_data(specimen_name)
            if mz is not None:
                raw[f'dic_{direction}_{specimen}'] = mz
                print(f"  {specimen_name}: loaded multi-zone DIC (zone-averaged)")
            else:
                df_path = os.path.join(DATA_DIR, f'{direction}-{specimen}.csv')
                if os.path.exists(df_path):
                    raw[f'dic_{direction}_{specimen}'] = load_dic_data(df_path)
                    print(f"  {specimen_name}: loaded single-zone DIC (fallback)")

    return raw


def run_single_mc_sample(raw_data, rng, E_sample):
    """
    Run one Monte Carlo sample: perturb data, compute all outputs.
    """
    results = {}

    canonical = raw_data.get('canonical_anisotropy', {})
    # Combined MC path: always perturb per-specimen DIC strains point-by-point
    # and recompute r from the noisy curves; then blend with the canonical
    # MATLAB specimen statistics (when available) so that both measurement-
    # noise and specimen-to-specimen scatter are represented.
    r_values = {'00': [], '45': [], '90': []}

    for direction in ['00', '45', '90']:
        for specimen in ['01', '02', '03']:
            dic_key = f'dic_{direction}_{specimen}'
            if dic_key not in raw_data:
                continue

            dic = raw_data[dic_key]
            eyy = dic['Eyy'].values.copy()
            exx = dic['Exx'].values.copy()

            # Per-point multiplicative DIC noise
            eyy_noisy = eyy * (1 + rng.normal(0, DIC_NOISE_LEVEL, len(eyy)))
            exx_noisy = exx * (1 + rng.normal(0, DIC_NOISE_LEVEL, len(exx)))

            eyy_smooth = smooth_strain(eyy_noisy, window=31)
            exx_smooth = smooth_strain(exx_noisy, window=31)

            r = compute_r_value(eyy_smooth, exx_smooth)
            if not np.isnan(r) and r > 0:
                r_values[direction].append(r)

    r_from_dic = {
        d: (float(np.mean(r_values[d])) if r_values[d] else np.nan)
        for d in ['00', '45', '90']
    }

    if canonical.get('available'):
        try:
            r0_spec = sample_direction_r_value(
                canonical['direction_results']['00'], rng, relative_noise=0.0)
            r45_spec = sample_direction_r_value(
                canonical['direction_results']['45'], rng, relative_noise=0.0)
            r90_spec = sample_direction_r_value(
                canonical['direction_results']['90'], rng, relative_noise=0.0)
        except KeyError:
            return None

        # Anchor to canonical MATLAB baseline and add the DIC-perturbation
        # residual (measurement noise) as an additive correction.
        baseline = canonical['r_values']
        def _combine(baseline_val, spec_val, dic_val):
            residual = 0.0 if np.isnan(dic_val) else (dic_val - baseline_val)
            return float(max(spec_val + residual, 1e-6))

        r0 = _combine(baseline['00'], r0_spec, r_from_dic['00'])
        r45 = _combine(baseline['45'], r45_spec, r_from_dic['45'])
        r90 = _combine(baseline['90'], r90_spec, r_from_dic['90'])
    else:
        r0, r45, r90 = r_from_dic['00'], r_from_dic['45'], r_from_dic['90']

    if np.isnan(r0) or np.isnan(r45) or np.isnan(r90):
        return None

    results['r0'] = r0
    results['r45'] = r45
    results['r90'] = r90

    # Hill'48
    hill = compute_hill48(r0, r45, r90)
    results.update({f'hill_{k}': v for k, v in hill.items()})

    # Hardening fits (perturbed E affects plastic strain)
    for direction in ['00', '45', '90']:
        all_ep = []
        all_ts = []
        for specimen in ['01', '02', '03']:
            sk = f'stress_{direction}_{specimen}'
            if sk not in raw_data:
                continue
            df = raw_data[sk]
            eng_s = df['Stress'].values
            eng_e = df['Strain'].values

            uts_idx = np.argmax(eng_s)
            eng_s = eng_s[1:uts_idx+1]
            eng_e = eng_e[1:uts_idx+1]

            # Per-point relative stress noise (not a single-sample rescale)
            stress_noise = rng.normal(0.0, 0.02, size=eng_s.shape)
            eng_s = eng_s * (1.0 + stress_noise)

            true_strain = np.log(1 + eng_e)
            true_stress = eng_s * (1 + eng_e)
            plastic_strain = true_strain - true_stress / E_sample

            mask = plastic_strain > 0
            if np.sum(mask) > 10:
                all_ep.extend(plastic_strain[mask].tolist())
                all_ts.extend(true_stress[mask].tolist())

        if all_ep:
            ep = np.array(all_ep)
            ts = np.array(all_ts)
            voce = fit_voce(ep, ts)
            if voce:
                results[f'voce_{direction}_sigma_y'] = voce['sigma_y']
                results[f'voce_{direction}_Q'] = voce['Q']
                results[f'voce_{direction}_b'] = voce['b']
                results[f'voce_{direction}_R2'] = voce['R2']

    return results


def run_monte_carlo(n_samples=N_MONTE_CARLO):
    """
    Run full Monte Carlo analysis.
    """
    print("=" * 70)
    print(f"MONTE CARLO UNCERTAINTY ANALYSIS ({n_samples} samples)")
    print("=" * 70)

    print("\nLoading raw data...")
    raw_data = load_all_raw_data()
    print(f"  Loaded {len(raw_data)} datasets")
    if raw_data['canonical_anisotropy']['available']:
        print(f"  Anisotropy source: {raw_data['canonical_anisotropy']['description']}")

    print(f"\nPerturbation sources:")
    print(f"  DIC strain noise: ±{DIC_NOISE_LEVEL*100:.0f}% (relative)")
    print(f"  Young's modulus:  {E_NOMINAL/1000:.0f} ± {E_STD/1000:.0f} GPa")
    print(f"  Stress scatter:   ±2% (from specimen variation)")

    rng = np.random.default_rng(SEED)
    all_results = []

    print(f"\nRunning {n_samples} MC samples...")
    for i in range(n_samples):
        if (i+1) % 500 == 0:
            print(f"  Sample {i+1}/{n_samples}")

        E_sample = rng.normal(E_NOMINAL, E_STD)
        E_sample = max(E_sample, 150000)  # physical lower bound

        result = run_single_mc_sample(raw_data, rng, E_sample)
        if result is not None:
            all_results.append(result)

    print(f"  Valid samples: {len(all_results)}/{n_samples}")

    # Convert to DataFrame
    mc_df = pd.DataFrame(all_results)
    return mc_df


# ============================================================================
# ANALYSIS & PLOTTING
# ============================================================================

def analyze_and_plot(mc_df):
    """Analyze MC results and generate plots."""
    print("\n--- Uncertainty Summary ---")

    # Key parameters to analyze
    params = [
        ('r0', 'r₀'), ('r45', 'r₄₅'), ('r90', 'r₉₀'),
        ('hill_F', 'F'), ('hill_G', 'G'), ('hill_H', 'H'), ('hill_N', 'N'),
        ('hill_R22', 'R₂₂'), ('hill_R33', 'R₃₃'), ('hill_R12', 'R₁₂'),
    ]

    print(f"\n  {'Parameter':>12s}  {'Mean':>10s}  {'Std':>10s}  {'95% CI Low':>10s}  {'95% CI High':>10s}  {'CoV %':>8s}")
    print(f"  {'-'*68}")

    for col, name in params:
        if col in mc_df.columns:
            vals = mc_df[col].dropna()
            mean = vals.mean()
            std = vals.std()
            ci_lo = np.percentile(vals, 2.5)
            ci_hi = np.percentile(vals, 97.5)
            cov = std / mean * 100 if mean != 0 else 0
            print(f"  {name:>12s}  {mean:>10.4f}  {std:>10.4f}  {ci_lo:>10.4f}  {ci_hi:>10.4f}  {cov:>7.2f}%")

    # --- Plot 1: r-value distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (col, name) in enumerate([('r0', 'r₀'), ('r45', 'r₄₅'), ('r90', 'r₉₀')]):
        if col in mc_df.columns:
            vals = mc_df[col].dropna()
            ax = axes[i]
            ax.hist(vals, bins=50, density=True, alpha=0.7, color='steelblue',
                   edgecolor='white')
            ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean={vals.mean():.4f}')
            ax.axvline(np.percentile(vals, 2.5), color='orange', linestyle=':',
                      label=f'95% CI: [{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]')
            ax.axvline(np.percentile(vals, 97.5), color='orange', linestyle=':')
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Distribution (N={len(vals)})')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_mc_r_value_distributions.png'))
    plt.close()
    print("  Saved: fig_mc_r_value_distributions.png")

    # --- Plot 2: Hill'48 parameter distributions ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    hill_params = [('hill_F', 'F'), ('hill_G', 'G'), ('hill_H', 'H'), ('hill_N', 'N'),
                   ('hill_R22', 'R₂₂'), ('hill_R33', 'R₃₃'), ('hill_R12', 'R₁₂')]
    for i, (col, name) in enumerate(hill_params):
        if col in mc_df.columns:
            ax = axes[i//4][i%4]
            vals = mc_df[col].dropna()
            ax.hist(vals, bins=50, density=True, alpha=0.7, color='coral',
                   edgecolor='white')
            ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(name)
            ax.set_title(f'{name}: {vals.mean():.4f} ± {vals.std():.4f}')
    # Hide unused subplot
    axes[1][3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_mc_hill48_distributions.png'))
    plt.close()
    print("  Saved: fig_mc_hill48_distributions.png")

    # --- Plot 3: Voce parameter distributions ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    voce_params = ['sigma_y', 'Q', 'b']
    voce_labels = ['σ_y (MPa)', 'Q (MPa)', 'b']

    for row, direction in enumerate(['00', '45', '90']):
        for col_idx, (pname, plabel) in enumerate(zip(voce_params, voce_labels)):
            col = f'voce_{direction}_{pname}'
            if col in mc_df.columns:
                ax = axes[row][col_idx]
                vals = mc_df[col].dropna()
                ax.hist(vals, bins=50, density=True, alpha=0.7, color='mediumseagreen',
                       edgecolor='white')
                ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=2)
                ci_lo = np.percentile(vals, 2.5)
                ci_hi = np.percentile(vals, 97.5)
                ax.axvline(ci_lo, color='orange', linestyle=':')
                ax.axvline(ci_hi, color='orange', linestyle=':')
                ax.set_xlabel(plabel)
                ax.set_title(f'{int(direction)}° {plabel}: {vals.mean():.1f} ± {vals.std():.1f}')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_mc_voce_distributions.png'))
    plt.close()
    print("  Saved: fig_mc_voce_distributions.png")

    # --- Plot 4: Correlation matrix of key parameters ---
    key_cols = ['r0', 'r45', 'r90', 'hill_R22', 'hill_R33', 'hill_R12']
    available = [c for c in key_cols if c in mc_df.columns]

    if len(available) >= 3:
        fig, ax = plt.subplots(figsize=(8, 7))
        corr = mc_df[available].corr()
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(available)))
        ax.set_yticks(range(len(available)))
        labels = [c.replace('hill_', '') for c in available]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        for i in range(len(available)):
            for j in range(len(available)):
                ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                       fontsize=10, color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')

        plt.colorbar(im, label='Correlation')
        ax.set_title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig_mc_correlation_matrix.png'))
        plt.close()
        print("  Saved: fig_mc_correlation_matrix.png")

    # --- Plot 5: Tornado sensitivity chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sensitivities = []
    for col, name in params:
        if col in mc_df.columns:
            vals = mc_df[col].dropna()
            cov = vals.std() / vals.mean() * 100 if vals.mean() != 0 else 0
            sensitivities.append((name, cov))

    sensitivities.sort(key=lambda x: x[1])
    names_sorted = [s[0] for s in sensitivities]
    covs_sorted = [s[1] for s in sensitivities]

    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sensitivities)))
    ax.barh(names_sorted, covs_sorted, color=colors)
    ax.set_xlabel('Coefficient of Variation (%)')
    ax.set_title('Parameter Sensitivity (Tornado Chart)')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_mc_tornado_sensitivity.png'))
    plt.close()
    print("  Saved: fig_mc_tornado_sensitivity.png")

    return mc_df


def save_mc_results(mc_df):
    """Save MC results summary."""
    canonical = load_canonical_anisotropy(DATA_DIR)
    summary = {}
    for col in mc_df.columns:
        vals = mc_df[col].dropna()
        summary[col] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'ci_2.5': float(np.percentile(vals, 2.5)),
            'ci_97.5': float(np.percentile(vals, 97.5)),
            'cov_pct': float(vals.std() / vals.mean() * 100) if vals.mean() != 0 else 0,
            'n_valid': int(len(vals))
        }

    with open(os.path.join(OUTPUT_DIR, 'mc_uncertainty_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved: mc_uncertainty_results.json")

    with open(os.path.join(OUTPUT_DIR, 'mc_uncertainty_summary.txt'), 'w') as f:
        f.write("MONTE CARLO UNCERTAINTY ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of MC samples: {N_MONTE_CARLO}\n")
        if canonical['available']:
            f.write(f"Anisotropy source: {canonical['description']}\n")
        f.write(f"DIC noise: ±{DIC_NOISE_LEVEL*100:.0f}%\n")
        f.write(f"E uncertainty: ±{E_STD/1000:.0f} GPa\n")
        f.write(f"Stress scatter: ±2%\n\n")

        f.write(f"{'Parameter':>15s}  {'Mean':>10s}  {'Std':>10s}  "
               f"{'95% CI':>22s}  {'CoV %':>8s}\n")
        f.write(f"{'-'*70}\n")

        for col, s in sorted(summary.items()):
            f.write(f"{col:>15s}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
                   f"[{s['ci_2.5']:>8.4f}, {s['ci_97.5']:>8.4f}]  "
                   f"{s['cov_pct']:>7.2f}%\n")

    print("  Saved: mc_uncertainty_summary.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    mc_df = run_monte_carlo()
    mc_df = analyze_and_plot(mc_df)
    save_mc_results(mc_df)

    print("\n" + "=" * 70)
    print("UNCERTAINTY ANALYSIS COMPLETE")
    print("=" * 70)

    return mc_df


if __name__ == '__main__':
    mc_df = main()
