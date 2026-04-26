#!/usr/bin/env python3
"""
=============================================================================
Exy Validation & Specimen Alignment Quality Analysis
=============================================================================
Validates DIC strain field quality by analyzing:
  1. Exy (shear strain) magnitude for all specimens
  2. Strain ratios εxy/εyy across strain levels
  3. Strain component evolution during test
  4. Alignment quality assessment (should be |Exy/Eyy| << 1 for good alignment)

Low shear strain confirms proper specimen alignment and loading.
Non-zero Exy may indicate misalignment, grip issues, or material effects.

Author: PhD Candidate
Date:   2026
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'lines.linewidth': 1.5, 'font.family': 'serif',
})


# ============================================================================
# DATA LOADING
# ============================================================================

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
    Returns: dict with 'zones' (list of DataFrames), 'mean' (DataFrame), 'n_zones'
    """
    export_dir = os.path.join(RAW_DATA_DIR, specimen_name, 'strain_export')
    
    zone_dfs = []
    z = 1
    while True:
        zfile = os.path.join(export_dir, f'{specimen_name}-zone{z:02d}-strains.csv')
        if not os.path.exists(zfile):
            break
        df = pd.read_csv(zfile)
        zone_dfs.append(df)
        z += 1
    
    if not zone_dfs:
        return None
    
    n_frames = min(len(df) for df in zone_dfs)
    mean_exx = np.mean([df['Exx'].values[:n_frames] for df in zone_dfs], axis=0)
    mean_eyy = np.mean([df['Eyy'].values[:n_frames] for df in zone_dfs], axis=0)
    mean_exy = np.mean([df['Exy'].values[:n_frames] for df in zone_dfs], axis=0)
    
    mean_df = pd.DataFrame({
        'Step': np.arange(1, n_frames + 1),
        'Eyy': mean_eyy, 'Exx': mean_exx, 'Exy': mean_exy,
    })
    
    return {'zones': zone_dfs, 'mean': mean_df, 'n_zones': len(zone_dfs)}


def smooth(x, window=31):
    if len(x) < window:
        return x
    return savgol_filter(x, window, 3)


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_shear_strain():
    """Load all DIC data and analyze Exy quality."""
    print("=" * 70)
    print("Exy SHEAR STRAIN VALIDATION & ALIGNMENT QUALITY")
    print("=" * 70)

    directions = ['00', '45', '90']
    specimens = ['01', '02', '03']
    colors = {'01': '#1f77b4', '02': '#ff7f0e', '03': '#2ca02c'}
    dir_labels = {'00': '0°', '45': '45°', '90': '90°'}

    all_data = {}
    all_multizone = {}

    # Load all data — prefer multi-zone
    for d in directions:
        for s in specimens:
            specimen_name = f'{d}-{s}'
            mz = load_multizone_dic_data(specimen_name)
            if mz is not None:
                all_data[specimen_name] = mz['mean']
                all_multizone[specimen_name] = mz
                print(f"  {specimen_name}: loaded {mz['n_zones']} zones from strain_export")
            else:
                fname = os.path.join(DATA_DIR, f'{d}-{s}.csv')
                if os.path.exists(fname):
                    all_data[specimen_name] = load_dic_data(fname)
                    print(f"  {specimen_name}: loaded single-zone (fallback)")
                else:
                    print(f"  WARNING: {specimen_name} — no data found")

    # ----------------------------------------------------------------
    # 1. Summary statistics
    # ----------------------------------------------------------------
    print("\n--- Shear Strain Summary ---")
    print(f"  {'Specimen':>12s}  {'|Exy| max':>10s}  {'|Exy| mean':>10s}  "
          f"{'|Exy/Eyy| max':>13s}  {'|Exy/Eyy| mean':>14s}  {'Quality':>10s}")
    print(f"  {'-'*75}")

    summary_data = []

    for d in directions:
        for s in specimens:
            key = f'{d}-{s}'
            if key not in all_data:
                continue
            df = all_data[key]
            eyy = df['Eyy'].values
            exy = df['Exy'].values

            mask = np.abs(eyy) > 0.005
            if np.sum(mask) < 5:
                continue

            exy_filtered = exy[mask]
            eyy_filtered = eyy[mask]
            ratio = np.abs(exy_filtered / eyy_filtered)

            max_exy = np.max(np.abs(exy_filtered))
            mean_exy = np.mean(np.abs(exy_filtered))
            max_ratio = np.max(ratio)
            mean_ratio = np.mean(ratio)

            # Per-zone Exy analysis if multi-zone available
            zone_str = ''
            if key in all_multizone:
                mz = all_multizone[key]
                zone_exy_maxes = []
                for zdf in mz['zones']:
                    zm = np.abs(zdf['Eyy'].values) > 0.005
                    if np.sum(zm) > 5:
                        zone_exy_maxes.append(np.max(np.abs(zdf['Exy'].values[zm])))
                if zone_exy_maxes:
                    zone_str = f'  ZoneMax:{np.max(zone_exy_maxes):.5f}'

            quality = 'Good' if mean_ratio < 0.05 else ('Fair' if mean_ratio < 0.10 else 'Poor')

            summary_data.append({
                'specimen': key, 'direction': d,
                'max_exy': max_exy, 'mean_exy': mean_exy,
                'max_ratio': max_ratio, 'mean_ratio': mean_ratio,
                'quality': quality
            })

            print(f"  {key:>12s}  {max_exy:>10.6f}  {mean_exy:>10.6f}  "
                  f"{max_ratio:>13.6f}  {mean_ratio:>14.6f}  {quality:>10s}")

    # ----------------------------------------------------------------
    # 2. Plot: Exy evolution for all specimens
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, d in enumerate(directions):
        ax = axes[col]
        for s in specimens:
            key = f'{d}-{s}'
            if key not in all_data:
                continue
            
            if key in all_multizone:
                # Multi-zone: show mean ± std band
                mz = all_multizone[key]
                n_frames = min(len(zdf) for zdf in mz['zones'])
                eyy_arr = np.array([zdf['Eyy'].values[:n_frames] for zdf in mz['zones']])
                exy_arr = np.array([zdf['Exy'].values[:n_frames] for zdf in mz['zones']])
                eyy_mean = smooth(np.mean(eyy_arr, axis=0))
                exy_mean = smooth(np.mean(exy_arr, axis=0))
                exy_std = smooth(np.std(exy_arr, axis=0))
                ax.plot(eyy_mean, exy_mean, label=f'Spec {s} ({mz["n_zones"]}z)',
                       color=colors[s], linewidth=1.5)
                ax.fill_between(eyy_mean, exy_mean - exy_std, exy_mean + exy_std,
                               color=colors[s], alpha=0.15)
            else:
                df = all_data[key]
                eyy = smooth(df['Eyy'].values)
                exy = smooth(df['Exy'].values)
                ax.plot(eyy, exy, label=f'Specimen {s}', color=colors[s], linewidth=1.5)

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Eyy (Axial Strain)')
        ax.set_ylabel('Exy (Shear Strain)')
        ax.set_title(f'{dir_labels[d]} Direction')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Shear Strain (Exy) vs Axial Strain (Eyy)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_exy_evolution.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_exy_evolution.png")

    # ----------------------------------------------------------------
    # 3. Plot: |Exy/Eyy| ratio vs strain level
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, d in enumerate(directions):
        ax = axes[col]
        for s in specimens:
            key = f'{d}-{s}'
            if key not in all_data:
                continue
            
            if key in all_multizone:
                mz = all_multizone[key]
                n_frames = min(len(zdf) for zdf in mz['zones'])
                eyy_arr = np.array([zdf['Eyy'].values[:n_frames] for zdf in mz['zones']])
                exy_arr = np.array([zdf['Exy'].values[:n_frames] for zdf in mz['zones']])
                eyy_mean = smooth(np.mean(eyy_arr, axis=0))
                exy_mean = smooth(np.mean(exy_arr, axis=0))
                mask = np.abs(eyy_mean) > 0.002
                if np.sum(mask) > 5:
                    ratio = np.abs(exy_mean[mask] / eyy_mean[mask])
                    ax.plot(eyy_mean[mask], ratio, label=f'Spec {s} ({mz["n_zones"]}z)',
                           color=colors[s], linewidth=1.5)
            else:
                df = all_data[key]
                eyy = smooth(df['Eyy'].values)
                exy = smooth(df['Exy'].values)
                mask = np.abs(eyy) > 0.002
                if np.sum(mask) > 5:
                    ratio = np.abs(exy[mask] / eyy[mask])
                    ax.plot(eyy[mask], ratio, label=f'Specimen {s}',
                           color=colors[s], linewidth=1.5)

        ax.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='5% threshold')
        ax.set_xlabel('Eyy (Axial Strain)')
        ax.set_ylabel('|Exy/Eyy|')
        ax.set_title(f'{dir_labels[d]} Direction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle('Strain Ratio |Exy/Eyy| — Alignment Quality Indicator', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_exy_ratio_validation.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_exy_ratio_validation.png")

    # ----------------------------------------------------------------
    # 4. Plot: All 3 strain components (Eyy, Exx, Exy) per specimen
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    for row, d in enumerate(directions):
        for col, s in enumerate(specimens):
            key = f'{d}-{s}'
            if key not in all_data:
                continue

            ax = axes[row][col]
            
            if key in all_multizone:
                mz = all_multizone[key]
                n_frames = min(len(zdf) for zdf in mz['zones'])
                steps = np.arange(1, n_frames + 1)
                eyy_arr = np.array([zdf['Eyy'].values[:n_frames] for zdf in mz['zones']])
                exx_arr = np.array([zdf['Exx'].values[:n_frames] for zdf in mz['zones']])
                exy_arr = np.array([zdf['Exy'].values[:n_frames] for zdf in mz['zones']])
                
                ax.plot(steps, smooth(np.mean(eyy_arr, axis=0)), 'b-', label='Eyy', linewidth=1.5)
                ax.plot(steps, smooth(np.mean(exx_arr, axis=0)), 'r-', label='Exx', linewidth=1.5)
                ax.plot(steps, smooth(np.mean(exy_arr, axis=0)), 'g-', label='Exy', linewidth=1.5)
                # ± std band for Exy
                exy_mean = smooth(np.mean(exy_arr, axis=0))
                exy_std = smooth(np.std(exy_arr, axis=0))
                ax.fill_between(steps, exy_mean - exy_std, exy_mean + exy_std,
                               color='green', alpha=0.15)
            else:
                df = all_data[key]
                steps = np.arange(len(df))
                ax.plot(steps, smooth(df['Eyy'].values), 'b-', label='Eyy', linewidth=1.5)
                ax.plot(steps, smooth(df['Exx'].values), 'r-', label='Exx', linewidth=1.5)
                ax.plot(steps, smooth(df['Exy'].values), 'g-', label='Exy', linewidth=1.5)
            
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Strain')
            nz_str = f' ({all_multizone[key]["n_zones"]}z)' if key in all_multizone else ''
            ax.set_title(f'{dir_labels[d]} — Specimen {s}{nz_str}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('DIC Strain Component Evolution', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_dic_strain_components.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_dic_strain_components.png")

    # ----------------------------------------------------------------
    # 5. Plot: Box plot of |Exy/Eyy| per direction
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = []
    box_labels = []

    for d in directions:
        dir_ratios = []
        for s in specimens:
            key = f'{d}-{s}'
            if key not in all_data:
                continue
            
            if key in all_multizone:
                mz = all_multizone[key]
                n_frames = min(len(zdf) for zdf in mz['zones'])
                eyy_mean = smooth(np.mean([zdf['Eyy'].values[:n_frames] for zdf in mz['zones']], axis=0))
                exy_mean = smooth(np.mean([zdf['Exy'].values[:n_frames] for zdf in mz['zones']], axis=0))
                mask = np.abs(eyy_mean) > 0.005
                if np.sum(mask) > 5:
                    ratio = np.abs(exy_mean[mask] / eyy_mean[mask])
                    dir_ratios.extend(ratio.tolist())
            else:
                df = all_data[key]
                eyy = smooth(df['Eyy'].values)
                exy = smooth(df['Exy'].values)
                mask = np.abs(eyy) > 0.005
                if np.sum(mask) > 5:
                    ratio = np.abs(exy[mask] / eyy[mask])
                    dir_ratios.extend(ratio.tolist())
        if dir_ratios:
            box_data.append(dir_ratios)
            box_labels.append(dir_labels[d])

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    dir_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, c in zip(bp['boxes'], dir_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    ax.set_ylabel('|Exy/Eyy|')
    ax.set_title('Shear-to-Axial Strain Ratio by Direction')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_exy_ratio_boxplot.png'))
    plt.close()
    print("  Saved: fig_exy_ratio_boxplot.png")

    # ----------------------------------------------------------------
    # 6. Alignment quality report
    # ----------------------------------------------------------------
    print("\n--- Alignment Quality Assessment ---")
    for d in directions:
        d_items = [s for s in summary_data if s['direction'] == d]
        if d_items:
            avg_ratio = np.mean([s['mean_ratio'] for s in d_items])
            max_ratio = np.max([s['max_ratio'] for s in d_items])
            print(f"\n  {dir_labels[d]} direction:")
            print(f"    Average |Exy/Eyy| ratio: {avg_ratio:.4f}")
            print(f"    Maximum |Exy/Eyy| ratio: {max_ratio:.4f}")
            if avg_ratio < 0.03:
                print(f"    Assessment: EXCELLENT alignment — negligible shear")
            elif avg_ratio < 0.05:
                print(f"    Assessment: GOOD alignment — minor shear present")
            elif avg_ratio < 0.10:
                print(f"    Assessment: FAIR — measurable shear, consider material anisotropy effect")
            else:
                print(f"    Assessment: POOR — significant shear, check grip alignment")

    return summary_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    summary = analyze_shear_strain()

    print("\n" + "=" * 70)
    print("Exy VALIDATION COMPLETE")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    main()
