#!/usr/bin/env python3
"""
=============================================================================
PhD Research: SGCC JIS G 3302 Sheet Metal - Data Processing & Material 
              Characterization
=============================================================================
This script processes:
  1. UTM tensile test data (load-time) 
  2. DIC strain data (Eyy, Exx, Exy from Ufreckles/FEM)
  3. Combines stress (UTM) and strain (DIC) into true stress-strain curves
  4. Computes Lankford r-values (r0, r45, r90) for Hill'48 criterion
  5. Fits Swift and Voce hardening laws
  6. Generates all plots for the thesis

Material: SGCC JIS G 3302 (galvanized steel sheet)
Specimen: DIN 50125 dog-bone, gauge: 80mm x 20mm x 1.5mm
Test:     Uniaxial tension, 2mm/min, 0.2kN/s, 1Hz output
DIC:      Canon EOS R6 Mark II (24.2 MP), 4K 25fps, 1 frame/s extraction

Author: PhD Candidate
Date:   2026
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter
import os
import warnings
warnings.filterwarnings('ignore')

from anisotropy_reference import load_canonical_anisotropy

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Specimen geometry
GAUGE_LENGTH = 80.0      # mm
GAUGE_WIDTH  = 20.0      # mm
THICKNESS    = 1.5        # mm
CROSS_AREA   = GAUGE_WIDTH * THICKNESS  # 30 mm^2

# Standard material properties for SGCC JIS G 3302 (reference)
E_STANDARD_GPA = 200.0     # GPa — converted to MPa (×1000) wherever used in calculations
NU_STANDARD  = 0.3       # Poisson's ratio

# Plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    
    'lines.linewidth': 1.5,
})

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dic_data(filepath):
    """
    Load DIC strain data exported from Ufreckles (legacy single-zone format).
    Handles both comma-separated and semicolon-separated formats.
    
    Returns: DataFrame with columns [Step, Eyy, Exx, Exy]
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Detect delimiter
    if ';' in lines[7]:
        delimiter = ';'
    else:
        delimiter = ','
    
    # Data starts at line 8 (0-indexed: 7)
    data_lines = lines[7:]
    
    steps = []
    eyy = []    # Longitudinal strain
    exx = []    # Transverse strain
    exy = []    # Shear strain
    
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(delimiter)
        if len(parts) >= 5:
            try:
                steps.append(int(parts[1]))
                eyy.append(float(parts[2]))
                exx.append(float(parts[3]))
                exy.append(float(parts[4]))
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame({
        'Step': steps,
        'Eyy': eyy,
        'Exx': exx,
        'Exy': exy
    })
    return df


def load_multizone_dic_data(specimen_name):
    """
    Load multi-zone DIC strain data from raw_data/<specimen>/strain_export/.
    These CSVs are retained for diagnostic plots and fallback use.
    
    Returns: dict with:
        'zones': list of DataFrames (one per zone), each with [Frame, Exx, Eyy, Exy]
        'mean':  DataFrame with zone-averaged [Frame, Exx, Eyy, Exy] for plotting
        'n_zones': int
    """
    export_dir = os.path.join(DATA_DIR, 'raw_data', specimen_name, 'strain_export')
    
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
    
    # Compute mean across all zones
    n_frames = len(zone_dfs[0])
    mean_exx = np.zeros(n_frames)
    mean_eyy = np.zeros(n_frames)
    mean_exy = np.zeros(n_frames)
    for df in zone_dfs:
        mean_exx += df['Exx'].values[:n_frames]
        mean_eyy += df['Eyy'].values[:n_frames]
        mean_exy += df['Exy'].values[:n_frames]
    n = len(zone_dfs)
    mean_df = pd.DataFrame({
        'Step': np.arange(1, n_frames + 1),
        'Eyy': mean_eyy / n,
        'Exx': mean_exx / n,
        'Exy': mean_exy / n,
    })
    
    return {
        'zones': zone_dfs,
        'mean': mean_df,
        'n_zones': n,
    }


def compute_zone_r_value(eyy, exx, strain_min=0.02, strain_max=0.10):
    """
    Compute r-value for a single zone using linear regression in [strain_min, strain_max].
    
    Returns: (r_value, R2, n_pts, slope, intercept, quality)
        quality: 'GOOD' if R² > 0.98, 'LOW_R2' otherwise
    """
    mask = (np.abs(eyy) >= strain_min) & (np.abs(eyy) <= strain_max)
    n_pts = np.sum(mask)
    
    if n_pts < 10:
        return np.nan, np.nan, 0, np.nan, np.nan, 'INSUFFICIENT'
    
    eyy_sel = eyy[mask]
    exx_sel = exx[mask]
    
    # Linear regression
    coeffs = np.polyfit(eyy_sel, exx_sel, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # R²
    predicted = np.polyval(coeffs, eyy_sel)
    ss_res = np.sum((exx_sel - predicted) ** 2)
    ss_tot = np.sum((exx_sel - np.mean(exx_sel)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    r_value = -slope / (1 + slope) if (1 + slope) != 0 else np.nan
    
    # Quality classification — R² only (spatial CV was already handled by MATLAB extraction)
    if r2 > 0.98:
        quality = 'GOOD'
    else:
        quality = 'LOW_R2'
    
    return r_value, r2, n_pts, slope, intercept, quality


def compute_multizone_r_value(multizone_data, strain_min=0.02, strain_max=0.10):
    """
    Compute a diagnostic quality-filtered r-value from multi-zone CSV data.

    The canonical anisotropy values should come from the MATLAB .mat outputs,
    which include the full CV(Eyy) screening and IQR rejection workflow.
    
    Returns: dict with per-zone details and weighted mean r-value
    """
    zone_results = []
    
    for i, zdf in enumerate(multizone_data['zones']):
        eyy = zdf['Eyy'].values
        exx = zdf['Exx'].values
        r_val, r2, n_pts, slope, intercept, quality = compute_zone_r_value(
            eyy, exx, strain_min, strain_max)
        zone_results.append({
            'zone': i + 1, 'r': r_val, 'R2': r2, 'n_pts': n_pts,
            'slope': slope, 'intercept': intercept, 'quality': quality
        })
    
    # Weighted mean of GOOD zones only
    good = [z for z in zone_results if z['quality'] == 'GOOD']
    if good:
        r_weighted = np.mean([z['r'] for z in good])
        r_std = np.std([z['r'] for z in good])
    else:
        # Fallback to all zones
        valid = [z for z in zone_results if not np.isnan(z['r'])]
        r_weighted = np.mean([z['r'] for z in valid]) if valid else np.nan
        r_std = np.std([z['r'] for z in valid]) if valid else np.nan
    
    return {
        'zones': zone_results,
        'r_weighted': r_weighted,
        'r_std': r_std,
        'n_good': len(good),
        'n_total': len(zone_results),
    }


def load_stress_data(filepath):
    """
    Load combined stress-strain data (UTM load + DIC strain).
    Format: Time, Load(kN), Stress(MPa), Strain
    
    Returns: DataFrame with columns [Time, Load, Stress, Strain]
    """
    df = pd.read_csv(filepath)
    df.columns = ['Time', 'Load', 'Stress', 'Strain']
    return df


def load_all_data():
    """
    Load all data for 3 directions x 3 specimens.
    Uses multi-zone DIC data from raw_data/*/strain_export/ when available,
    falling back to single-zone root CSV files.
    
    Returns: dict with keys like 'dic_00_01', 'stress_00_01', 'multizone_00_01', etc.
    """
    data = {}
    directions = ['00', '45', '90']
    specimens = ['01', '02', '03']
    
    for d in directions:
        for s in specimens:
            specimen_name = f'{d}-{s}'
            stress_file = os.path.join(DATA_DIR, f'stress-{d}-{s}.csv')
            
            # Load multi-zone DIC data
            mz = load_multizone_dic_data(specimen_name)
            if mz is not None:
                data[f'dic_{d}_{s}'] = mz['mean']  # zone-averaged for compatibility
                data[f'multizone_{d}_{s}'] = mz     # full zone-level data
                print(f"    {specimen_name}: loaded {mz['n_zones']} zones from strain_export")
            else:
                # Fallback to old single-zone CSV
                dic_file = os.path.join(DATA_DIR, f'{d}-{s}.csv')
                if os.path.exists(dic_file):
                    data[f'dic_{d}_{s}'] = load_dic_data(dic_file)
                    print(f"    {specimen_name}: loaded single-zone (fallback)")
            
            if os.path.exists(stress_file):
                data[f'stress_{d}_{s}'] = load_stress_data(stress_file)
    
    return data


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def smooth_strain(strain_data, window=21, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth noisy DIC strain data.
    """
    if len(strain_data) < polyorder + 2:
        return np.array(strain_data)
    if len(strain_data) < window:
        window = max(polyorder + 2, len(strain_data) // 2 * 2 + 1)
        if window % 2 == 0:
            window += 1
    return savgol_filter(strain_data, window, polyorder)


def find_yield_point(stress, strain, offset=0.002, E=None):
    """
    Find yield point using 0.2% offset method.
    If E (Young's modulus) is not reliable from DIC, use standard value.
    
    Parameters:
        stress: engineering stress (MPa)
        strain: engineering strain
        offset: offset strain (default 0.002 = 0.2%)
        E:      Young's modulus in MPa (if None, use standard)
    
    Returns: (yield_stress, yield_strain, E_used)
    """
    if E is None:
        E = E_STANDARD_GPA * 1000  # Convert GPa to MPa
    
    # Offset line: sigma = E * (epsilon - offset)
    offset_line = E * (strain - offset)
    
    # Find intersection
    diff = stress - offset_line
    
    # Find where diff changes sign (from positive to negative)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation
        x1, x2 = strain[idx], strain[idx+1]
        y1, y2 = diff[idx], diff[idx+1]
        strain_yield = x1 - y1 * (x2 - x1) / (y2 - y1)
        stress_yield = E * (strain_yield - offset)
        return stress_yield, strain_yield, E
    else:
        # Fallback: use a reasonable estimate
        # Find stress at 0.2% strain
        idx = np.argmin(np.abs(strain - offset))
        return stress[idx], strain[idx], E


def find_uts_and_fracture(stress, strain):
    """
    Find Ultimate Tensile Strength (UTS) and fracture point.
    
    Returns: (uts_stress, uts_strain, fracture_stress, fracture_strain)
    """
    uts_idx = np.argmax(stress)
    uts_stress = stress[uts_idx]
    uts_strain = strain[uts_idx]
    
    # Fracture: last point (or where stress drops significantly after UTS)
    fracture_stress = stress[-1]
    fracture_strain = strain[-1]
    
    return uts_stress, uts_strain, fracture_stress, fracture_strain


def engineering_to_true(eng_stress, eng_strain):
    """
    Convert engineering stress-strain to true stress-strain.
    
    True stress:  sigma_true = sigma_eng * (1 + epsilon_eng)
    True strain:  epsilon_true = ln(1 + epsilon_eng)
    
    Valid only up to necking (uniform elongation).
    """
    true_strain = np.log(1 + eng_strain)
    true_stress = eng_stress * (1 + eng_strain)
    return true_stress, true_strain


def compute_plastic_strain(true_stress, true_strain, E=None):
    """
    Compute plastic strain from true stress-strain.
    epsilon_plastic = epsilon_true - sigma_true / E
    """
    if E is None:
        E = E_STANDARD_GPA * 1000  # GPa to MPa
    
    plastic_strain = true_strain - true_stress / E
    # Remove negative values
    mask = plastic_strain >= 0
    return plastic_strain[mask], true_stress[mask], true_strain[mask]


# ============================================================================
# LANKFORD R-VALUE (ANISOTROPY) COMPUTATION
# ============================================================================

def compute_r_value(eyy, exx):
    """
    Compute Lankford coefficient (r-value) from DIC strain data.
    
    r = epsilon_width / epsilon_thickness
    
    For uniaxial tension with volume conservation:
        epsilon_thickness = -(epsilon_longitudinal + epsilon_transverse)
    Therefore:
        r = epsilon_transverse / (-(epsilon_longitudinal + epsilon_transverse))
        r = -exx / (eyy + exx)
    
    where eyy = longitudinal strain, exx = transverse strain
    
    We compute r from the slope of exx vs eyy in the plastic region.
    """
    # Use only plastic region (strain > ~0.002 to avoid elastic noise)
    mask = np.abs(eyy) > 0.005
    
    if np.sum(mask) < 10:
        mask = np.abs(eyy) > 0.002
    
    eyy_plastic = eyy[mask]
    exx_plastic = exx[mask]
    
    # Linear fit: exx = slope * eyy + intercept
    # r = -slope / (1 + slope)
    if len(eyy_plastic) < 5:
        return np.nan, np.nan, np.nan
    
    coeffs = np.polyfit(eyy_plastic, exx_plastic, 1)
    slope = coeffs[0]
    
    r_value = -slope / (1 + slope)
    
    return r_value, slope, coeffs[1]


def compute_all_r_values(data):
    """
    Compute r-values for all directions and specimens.
    Uses canonical MATLAB multi-zone results when available and stores the
    Python CSV recomputation as a diagnostic fallback.
    
    Returns: dict with r0, r45, r90 (weighted average of specimens)
    """
    r_values = {'00': [], '45': [], '90': []}
    r_details = {}
    r_weights = {'00': [], '45': [], '90': []}
    canonical = load_canonical_anisotropy(DATA_DIR)

    if canonical['available']:
        print(f"\n  Anisotropy source: {canonical['description']}")
        for source_file in canonical['source_files']:
            print(f"    source file: {os.path.basename(source_file)}")

    r_details['source'] = canonical['description'] if canonical['available'] else (
        'Python CSV recomputation fallback; MATLAB anisotropy result files not found.'
    )
    r_details['source_files'] = canonical.get('source_files', [])
    
    for direction in ['00', '45', '90']:
        direction_result = canonical['direction_results'].get(direction) if canonical['available'] else None
        specimen_summary = {}
        if direction_result:
            specimen_summary = {
                item['specimen']: item for item in direction_result['specimen_summary']
            }

        for specimen in ['01', '02', '03']:
            mz_key = f'multizone_{direction}_{specimen}'
            dic_key = f'dic_{direction}_{specimen}'
            detail_key = f'r_{direction}_{specimen}'
            diagnostic = None
            
            if mz_key in data:
                mz_result = compute_multizone_r_value(data[mz_key])
                diagnostic = {
                    'r': mz_result['r_weighted'],
                    'r_std': mz_result['r_std'],
                    'n_good': mz_result['n_good'],
                    'n_total': mz_result['n_total'],
                    'zones': mz_result['zones'],
                    'weight': max(mz_result['n_good'], 1),
                    'method': 'python_multizone_diagnostic',
                }
            elif dic_key in data:
                dic = data[dic_key]
                eyy_smooth = smooth_strain(dic['Eyy'].values, window=31)
                exx_smooth = smooth_strain(dic['Exx'].values, window=31)
                
                r_val, slope, intercept = compute_r_value(eyy_smooth, exx_smooth)
                diagnostic = {
                    'r': r_val, 'slope': slope, 'intercept': intercept,
                    'weight': 1,
                    'method': 'single_zone_fallback'
                }

            if specimen in specimen_summary:
                summary = specimen_summary[specimen]
                r_val = summary['r']
                r_values[direction].append(r_val)
                r_weights[direction].append(max(summary['n_good'], 1))
                r_details[detail_key] = {
                    'r': r_val,
                    'r_std': summary['r_std'],
                    'n_good': summary['n_good'],
                    'n_total': summary['n_good'],
                    'method': 'matlab_multizone_summary',
                    'source_file': direction_result['path'],
                }
                if diagnostic is not None:
                    r_details[detail_key]['python_diagnostic'] = diagnostic

                print(f"  r_{direction}_{specimen} = {r_val:.4f} "
                      f"(MATLAB summary, {summary['n_good']} GOOD zones)")
                if diagnostic is not None and np.isfinite(diagnostic['r']):
                    delta = diagnostic['r'] - r_val
                    print(f"    diagnostic CSV recomputation = {diagnostic['r']:.4f} "
                          f"(delta = {delta:+.4f})")
            elif diagnostic is not None:
                r_values[direction].append(diagnostic['r'])
                r_weights[direction].append(diagnostic['weight'])
                r_details[detail_key] = diagnostic
                print(f"  r_{direction}_{specimen} = {diagnostic['r']:.4f} "
                      f"({diagnostic['method']})")
    
    r_avg = {}
    for d in ['00', '45', '90']:
        if canonical['available'] and d in canonical['direction_results']:
            direction_result = canonical['direction_results'][d]
            r_avg[d] = canonical['r_values'][d]
            print(f"\n  r_{d} (canonical MATLAB) = {r_avg[d]:.4f} "
                  f"+- {direction_result['specimen_std']:.4f}")
        else:
            vals = np.array([v for v in r_values[d] if not np.isnan(v)])
            wts = np.array([w for v, w in zip(r_values[d], r_weights[d]) if not np.isnan(v)])
            if len(vals) > 0 and np.sum(wts) > 0:
                r_avg[d] = np.average(vals, weights=wts)
            else:
                r_avg[d] = np.nan
            print(f"\n  r_{d} (weighted avg fallback) = {r_avg[d]:.4f} "
                  f"+- {np.std(vals):.4f}")
    
    return r_avg, r_values, r_details


# ============================================================================
# HILL'48 CRITERION PARAMETERS
# ============================================================================

def compute_hill48_parameters(r0, r45, r90):
    """
    Compute Hill'48 anisotropic yield criterion parameters.
    
    Hill'48: F*(sigma_22)^2 + G*(sigma_11)^2 + H*(sigma_11 - sigma_22)^2 
             + 2N*(sigma_12)^2 = 1
    
    With r-values:
        F = r0 / (r90 * (1 + r0))
        G = 1 / (1 + r0)
        H = r0 / (1 + r0)
        N = (r0 + r90) * (1 + 2*r45) / (2 * r90 * (1 + r0))
    
    For Abaqus R-values (stress ratios):
        R11 = 1 (reference direction)
        R22 = sqrt(r90 * (1 + r0) / (r0 * (1 + r90)))
        R33 = sqrt(r90 * (1 + r0) / ((r0 + r90)))
        R12 = sqrt(3 * r90 * (1 + r0) / ((2*r45 + 1) * (r0 + r90)))
        R13 = 1 (default, plane stress)
        R23 = 1 (default, plane stress)
    """
    F = r0 / (r90 * (1 + r0))
    G = 1.0 / (1 + r0)
    H = r0 / (1 + r0)
    N = (r0 + r90) * (1 + 2*r45) / (2 * r90 * (1 + r0))
    
    # Abaqus stress ratios
    R11 = 1.0
    R22 = np.sqrt(r90 * (1 + r0) / (r0 * (1 + r90)))
    R33 = np.sqrt(r90 * (1 + r0) / (r0 + r90))
    R12 = np.sqrt(3.0 * r90 * (1 + r0) / ((2*r45 + 1) * (r0 + r90)))
    R13 = 1.0
    R23 = 1.0
    
    # Normal anisotropy and planar anisotropy
    r_bar = (r0 + 2*r45 + r90) / 4.0
    delta_r = (r0 - 2*r45 + r90) / 2.0
    
    params = {
        'F': F, 'G': G, 'H': H, 'N': N,
        'R11': R11, 'R22': R22, 'R33': R33,
        'R12': R12, 'R13': R13, 'R23': R23,
        'r_bar': r_bar, 'delta_r': delta_r
    }
    
    return params


# ============================================================================
# HARDENING LAW FITTING
# ============================================================================

def swift_law(eps_p, K, eps0, n):
    """
    Swift hardening law: sigma = K * (eps0 + eps_p)^n
    """
    return K * (eps0 + eps_p)**n


def voce_law(eps_p, sigma_y, Q, b):
    """
    Voce hardening law: sigma = sigma_y + Q * (1 - exp(-b * eps_p))
    """
    return sigma_y + Q * (1.0 - np.exp(-b * eps_p))


def swift_voce_combined(eps_p, alpha, K, eps0, n, sigma_y, Q, b):
    """
    Combined Swift-Voce: sigma = alpha * Swift + (1-alpha) * Voce
    """
    s = swift_law(eps_p, K, eps0, n)
    v = voce_law(eps_p, sigma_y, Q, b)
    return alpha * s + (1 - alpha) * v


def hockett_sherby_law(eps_p, sigma_s, sigma_y, m, p):
    """
    Hockett-Sherby hardening law:
    sigma = sigma_s - (sigma_s - sigma_y) * exp(-m * eps_p^p)
    
    Best overall R² in diagnostic tests.
    """
    return sigma_s - (sigma_s - sigma_y) * np.exp(-m * eps_p**p)


def modified_voce_law(eps_p, sigma_y, Q1, b1, Q2, b2):
    """
    Modified Voce with 2 exponential terms:
    sigma = sigma_y + Q1*(1 - exp(-b1*eps_p)) + Q2*(1 - exp(-b2*eps_p))
    
    Captures both fast initial hardening and slow saturation.
    """
    return sigma_y + Q1*(1-np.exp(-b1*eps_p)) + Q2*(1-np.exp(-b2*eps_p))


def ludwik_law(eps_p, sigma_y, K, n):
    """
    Ludwik hardening law: sigma = sigma_y + K * eps_p^n
    """
    return sigma_y + K * eps_p**n


# Minimum plastic strain threshold for fitting (skip elastic/transition noise)
PLASTIC_STRAIN_THRESHOLD = 0.002


def fit_swift(plastic_strain, true_stress, p0=None):
    """
    Fit Swift hardening law to true stress - plastic strain data.
    """
    if p0 is None:
        p0 = [600.0, 0.005, 0.2]
    
    # Filter to plastic region only (skip elastic/noisy transition)
    mask = plastic_strain >= PLASTIC_STRAIN_THRESHOLD
    if np.sum(mask) < 20:
        mask = plastic_strain >= 0
    plastic_strain = plastic_strain[mask]
    true_stress = true_stress[mask]
    
    try:
        popt, pcov = curve_fit(swift_law, plastic_strain, true_stress,
                               p0=p0, maxfev=10000,
                               bounds=([100, 1e-6, 0.01], [2000, 0.1, 1.0]))
        perr = np.sqrt(np.diag(pcov))
        residuals = true_stress - swift_law(plastic_strain, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true_stress - np.mean(true_stress))**2)
        r_squared = 1 - ss_res / ss_tot
        return {'K': popt[0], 'eps0': popt[1], 'n': popt[2],
                'K_err': perr[0], 'eps0_err': perr[1], 'n_err': perr[2],
                'R2': r_squared}
    except Exception as e:
        print(f"  Swift fitting failed: {e}")
        return None


def fit_voce(plastic_strain, true_stress, p0=None):
    """
    Fit Voce hardening law to true stress - plastic strain data.
    """
    if p0 is None:
        p0 = [250.0, 200.0, 10.0]
    
    # Filter to plastic region only
    mask = plastic_strain >= PLASTIC_STRAIN_THRESHOLD
    if np.sum(mask) < 20:
        mask = plastic_strain >= 0
    plastic_strain = plastic_strain[mask]
    true_stress = true_stress[mask]
    
    try:
        popt, pcov = curve_fit(voce_law, plastic_strain, true_stress,
                               p0=p0, maxfev=10000,
                               bounds=([50, 10, 0.1], [1000, 1000, 200]))
        perr = np.sqrt(np.diag(pcov))
        residuals = true_stress - voce_law(plastic_strain, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true_stress - np.mean(true_stress))**2)
        r_squared = 1 - ss_res / ss_tot
        return {'sigma_y': popt[0], 'Q': popt[1], 'b': popt[2],
                'sigma_y_err': perr[0], 'Q_err': perr[1], 'b_err': perr[2],
                'R2': r_squared}
    except Exception as e:
        print(f"  Voce fitting failed: {e}")
        return None


def fit_hockett_sherby(plastic_strain, true_stress):
    """
    Fit Hockett-Sherby hardening law.
    Best overall R² among all models tested.
    """
    mask = plastic_strain >= PLASTIC_STRAIN_THRESHOLD
    if np.sum(mask) < 20:
        mask = plastic_strain >= 0
    plastic_strain = plastic_strain[mask]
    true_stress = true_stress[mask]

    try:
        popt, pcov = curve_fit(hockett_sherby_law, plastic_strain, true_stress,
                               p0=[500, 250, 5, 0.5], maxfev=20000,
                               bounds=([200, 50, 0.01, 0.01], [1500, 500, 100, 5]))
        perr = np.sqrt(np.diag(pcov))
        residuals = true_stress - hockett_sherby_law(plastic_strain, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true_stress - np.mean(true_stress))**2)
        r_squared = 1 - ss_res / ss_tot
        return {'sigma_s': popt[0], 'sigma_y': popt[1], 'm': popt[2], 'p': popt[3],
                'sigma_s_err': perr[0], 'sigma_y_err': perr[1],
                'm_err': perr[2], 'p_err': perr[3],
                'R2': r_squared}
    except Exception as e:
        print(f"  Hockett-Sherby fitting failed: {e}")
        return None


def fit_modified_voce(plastic_strain, true_stress):
    """
    Fit Modified Voce with 2 exponential terms.
    """
    mask = plastic_strain >= PLASTIC_STRAIN_THRESHOLD
    if np.sum(mask) < 20:
        mask = plastic_strain >= 0
    plastic_strain = plastic_strain[mask]
    true_stress = true_stress[mask]

    try:
        popt, pcov = curve_fit(modified_voce_law, plastic_strain, true_stress,
                               p0=[250, 100, 20, 100, 2], maxfev=20000,
                               bounds=([50, 1, 0.1, 1, 0.01],
                                       [1000, 500, 200, 500, 50]))
        perr = np.sqrt(np.diag(pcov))
        residuals = true_stress - modified_voce_law(plastic_strain, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true_stress - np.mean(true_stress))**2)
        r_squared = 1 - ss_res / ss_tot
        return {'sigma_y': popt[0], 'Q1': popt[1], 'b1': popt[2],
                'Q2': popt[3], 'b2': popt[4],
                'R2': r_squared}
    except Exception as e:
        print(f"  Modified Voce fitting failed: {e}")
        return None


def fit_per_specimen_then_average(results, direction, fit_func, law_name):
    """
    Fit each specimen individually, then average the parameters.
    This avoids inter-specimen scatter degrading R² when pooling data.
    
    Returns: dict with averaged params and per-specimen R² values
    """
    all_params = {}
    r2_values = []

    for specimen in ['01', '02', '03']:
        key = f'{direction}_{specimen}'
        if key not in results or 'plastic_strain' not in results[key]:
            continue
        ep = results[key]['plastic_strain']
        ts = results[key]['true_stress_plastic']
        if len(ep) < 20:
            continue
        params = fit_func(ep, ts)
        if params is None:
            continue
        r2_values.append(params['R2'])
        for k, v in params.items():
            if k == 'R2':
                continue
            if k not in all_params:
                all_params[k] = []
            all_params[k].append(v)

    if not r2_values:
        return None

    avg_params = {}
    for k, vals in all_params.items():
        avg_params[k] = np.mean(vals)
        avg_params[f'{k}_std'] = np.std(vals)

    avg_params['R2_per_specimen'] = r2_values
    avg_params['R2_mean'] = np.mean(r2_values)
    avg_params['R2_std'] = np.std(r2_values)

    return avg_params


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_engineering_stress_strain(data, output_dir):
    """
    Plot engineering stress-strain curves for all specimens.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, direction in enumerate(['00', '45', '90']):
        ax = axes[idx]
        for j, specimen in enumerate(['01', '02', '03']):
            key = f'stress_{direction}_{specimen}'
            if key in data:
                df = data[key]
                ax.plot(df['Strain'].values, df['Stress'].values,
                       color=colors[j], label=f'Specimen {j+1}', alpha=0.8)
        
        ax.set_xlabel('Engineering Strain')
        ax.set_ylabel('Engineering Stress (MPa)')
        ax.set_title(f'{int(direction)}° Rolling Direction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_engineering_stress_strain.png'))
    plt.close()
    print("  Saved: fig_engineering_stress_strain.png")


def plot_true_stress_strain(results, output_dir):
    """
    Plot true stress-strain curves for all specimens.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, direction in enumerate(['00', '45', '90']):
        ax = axes[idx]
        for j, specimen in enumerate(['01', '02', '03']):
            key = f'{direction}_{specimen}'
            if key in results:
                r = results[key]
                ax.plot(r['true_strain'], r['true_stress'],
                       color=colors[j], label=f'Specimen {j+1}', alpha=0.8)
        
        ax.set_xlabel('True Strain')
        ax.set_ylabel('True Stress (MPa)')
        ax.set_title(f'{int(direction)}° Rolling Direction (True)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_true_stress_strain.png'))
    plt.close()
    print("  Saved: fig_true_stress_strain.png")


def plot_r_value_regression(data, output_dir, r_details=None):
    """
    Plot transverse vs longitudinal strain for r-value determination.
    Shows per-zone scatter with quality coloring when multi-zone data available.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    specimen_markers = ['o', 's', '^']
    
    for idx, direction in enumerate(['00', '45', '90']):
        ax = axes[idx]
        for j, specimen in enumerate(['01', '02', '03']):
            mz_key = f'multizone_{direction}_{specimen}'
            dic_key = f'dic_{direction}_{specimen}'
            
            if mz_key in data:
                # Multi-zone: show per-zone scatter with quality coloring
                mz = data[mz_key]
                mz_result = compute_multizone_r_value(mz)
                detail = r_details.get(f'r_{direction}_{specimen}', {}) if r_details else {}
                
                for zi, zdf in enumerate(mz['zones']):
                    eyy = zdf['Eyy'].values
                    exx = zdf['Exx'].values
                    mask = (np.abs(eyy) >= 0.02) & (np.abs(eyy) <= 0.10)
                    if np.sum(mask) < 5:
                        continue
                    quality = mz_result['zones'][zi]['quality']
                    color = '#2ca02c' if quality == 'GOOD' else '#d62728'
                    alpha = 0.15 if quality == 'GOOD' else 0.05
                    ax.scatter(eyy[mask], exx[mask], s=3, alpha=alpha,
                              color=color, marker=specimen_markers[j])
                
                # Weighted mean regression line
                r_val = detail.get('r', mz_result['r_weighted'])
                good_zones = [z for z in mz_result['zones'] if z['quality'] == 'GOOD']
                if good_zones and np.isfinite(r_val):
                    mean_slope = -r_val / (1.0 + r_val)
                    mean_intercept = np.mean([z['intercept'] for z in good_zones])
                    x_fit = np.linspace(0.02, 0.10, 100)
                    ax.plot(x_fit, mean_slope * x_fit + mean_intercept, '-',
                           color=f'C{j}', linewidth=2,
                           label=f'Spec {j+1}: r={r_val:.3f} '
                                 f'({len(good_zones)}G/{mz["n_zones"]})')
            
            elif dic_key in data:
                # Fallback: single-zone
                dic = data[dic_key]
                eyy = smooth_strain(dic['Eyy'].values, window=31)
                exx = smooth_strain(dic['Exx'].values, window=31)
                mask = np.abs(eyy) > 0.005
                ax.scatter(eyy[mask], exx[mask], s=5, alpha=0.3,
                          color=f'C{j}', label=f'Specimen {j+1}')
                if np.sum(mask) > 5:
                    coeffs = np.polyfit(eyy[mask], exx[mask], 1)
                    x_fit = np.linspace(eyy[mask].min(), eyy[mask].max(), 100)
                    ax.plot(x_fit, np.polyval(coeffs, x_fit), '--',
                           color=f'C{j}', linewidth=2)
        
        ax.set_xlabel('Longitudinal Strain (εyy)')
        ax.set_ylabel('Transverse Strain (εxx)')
        ax.set_title(f'{int(direction)}° Direction — Multi-Zone r-value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_r_value_regression.png'))
    plt.close()
    print("  Saved: fig_r_value_regression.png")


def plot_hardening_fits(results, output_dir):
    """
    Plot all hardening law fits (Swift, Voce, Hockett-Sherby, Mod. Voce)
    for each direction, with plastic strain threshold applied.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, direction in enumerate(['00', '45', '90']):
        ax = axes[idx]
        
        # Average data for this direction
        all_ep = []
        all_ts = []
        for specimen in ['01', '02', '03']:
            key = f'{direction}_{specimen}'
            if key in results and 'plastic_strain' in results[key]:
                all_ep.extend(results[key]['plastic_strain'].tolist())
                all_ts.extend(results[key]['true_stress_plastic'].tolist())
        
        if all_ep:
            ep = np.array(all_ep)
            ts = np.array(all_ts)
            
            # Sort by strain
            sort_idx = np.argsort(ep)
            ep = ep[sort_idx]
            ts = ts[sort_idx]
            
            ax.scatter(ep, ts, s=2, alpha=0.2, color='gray', label='Experimental')
            
            ep_fit = np.linspace(PLASTIC_STRAIN_THRESHOLD, ep.max(), 200)
            
            # Swift fit
            swift_params = fit_swift(ep, ts)
            if swift_params:
                ax.plot(ep_fit, swift_law(ep_fit, swift_params['K'],
                       swift_params['eps0'], swift_params['n']),
                       'r-', linewidth=2,
                       label=f'Swift (R²={swift_params["R2"]:.3f})')
            
            # Voce fit
            voce_params = fit_voce(ep, ts)
            if voce_params:
                ax.plot(ep_fit, voce_law(ep_fit, voce_params['sigma_y'],
                       voce_params['Q'], voce_params['b']),
                       'b--', linewidth=2,
                       label=f'Voce (R²={voce_params["R2"]:.3f})')
            
            # Hockett-Sherby fit
            hs_params = fit_hockett_sherby(ep, ts)
            if hs_params:
                ax.plot(ep_fit, hockett_sherby_law(ep_fit, hs_params['sigma_s'],
                       hs_params['sigma_y'], hs_params['m'], hs_params['p']),
                       'g-.', linewidth=2,
                       label=f'Hockett-Sherby (R²={hs_params["R2"]:.3f})')
            
            # Modified Voce fit
            mv_params = fit_modified_voce(ep, ts)
            if mv_params:
                ax.plot(ep_fit, modified_voce_law(ep_fit, mv_params['sigma_y'],
                       mv_params['Q1'], mv_params['b1'],
                       mv_params['Q2'], mv_params['b2']),
                       'm:', linewidth=2,
                       label=f'Mod. Voce (R²={mv_params["R2"]:.3f})')
        
        ax.set_xlabel('Plastic Strain')
        ax.set_ylabel('True Stress (MPa)')
        ax.set_title(f'{int(direction)}° Direction - Hardening Law Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_hardening_fits.png'))
    plt.close()
    print("  Saved: fig_hardening_fits.png")


def plot_dic_strain_evolution(data, output_dir):
    """
    Plot DIC strain evolution (Eyy, Exx, Exy) over time/steps.
    Shows multi-zone mean ± std band when available.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    strain_labels = ['Eyy (Longitudinal)', 'Exx (Transverse)', 'Exy (Shear)']
    strain_cols = ['Eyy', 'Exx', 'Exy']
    
    for idx, direction in enumerate(['00', '45', '90']):
        for row, (col, label) in enumerate(zip(strain_cols, strain_labels)):
            ax = axes[row][idx]
            for j, specimen in enumerate(['01', '02', '03']):
                mz_key = f'multizone_{direction}_{specimen}'
                dic_key = f'dic_{direction}_{specimen}'
                
                if mz_key in data:
                    mz = data[mz_key]
                    # Compute mean and std across zones
                    n_frames = min(len(zdf) for zdf in mz['zones'])
                    vals = np.array([zdf[col].values[:n_frames] for zdf in mz['zones']])
                    mean_vals = np.mean(vals, axis=0)
                    std_vals = np.std(vals, axis=0)
                    steps = np.arange(1, n_frames + 1)
                    
                    smoothed = smooth_strain(mean_vals, window=31)
                    ax.plot(steps, smoothed, color=colors[j],
                           label=f'Spec. {j+1} (8z)', linewidth=1.5)
                    ax.fill_between(steps,
                                   smooth_strain(mean_vals - std_vals, window=31),
                                   smooth_strain(mean_vals + std_vals, window=31),
                                   color=colors[j], alpha=0.15)
                
                elif dic_key in data:
                    dic = data[dic_key]
                    raw = dic[col].values
                    smoothed = smooth_strain(raw, window=31)
                    ax.plot(dic['Step'].values, raw, alpha=0.2,
                           color=colors[j])
                    ax.plot(dic['Step'].values, smoothed,
                           color=colors[j], label=f'Spec. {j+1}',
                           linewidth=1.5)
            
            ax.set_xlabel('Step (Time, s)')
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f'{int(direction)}° Direction')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_dic_strain_evolution.png'))
    plt.close()
    print("  Saved: fig_dic_strain_evolution.png")


def plot_yield_surface_hill48(hill_params, sigma_y, output_dir):
    """
    Plot Hill'48 yield surface in sigma_11-sigma_22 plane.
    """
    F = hill_params['F']
    G = hill_params['G']
    H = hill_params['H']
    
    theta = np.linspace(0, 2*np.pi, 500)
    
    # For sigma_12 = 0:
    # G*s11^2 + F*s22^2 + H*(s11-s22)^2 = sigma_y^2
    # (G+H)*s11^2 - 2H*s11*s22 + (F+H)*s22^2 = sigma_y^2
    
    s11_list = []
    s22_list = []
    
    for t in theta:
        ct, st = np.cos(t), np.sin(t)
        # Parametric: s11 = r*cos(t), s22 = r*sin(t)
        denom = (G+H)*ct**2 - 2*H*ct*st + (F+H)*st**2
        if denom > 0:
            r = sigma_y / np.sqrt(denom)
            s11_list.append(r * ct)
            s22_list.append(r * st)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(s11_list, s22_list, 'b-', linewidth=2, label="Hill'48 Yield Surface")
    
    # Von Mises for comparison
    vm_s11 = []
    vm_s22 = []
    for t in theta:
        ct, st = np.cos(t), np.sin(t)
        denom = ct**2 - ct*st + st**2
        r = sigma_y / np.sqrt(denom)
        vm_s11.append(r * ct)
        vm_s22.append(r * st)
    ax.plot(vm_s11, vm_s22, 'r--', linewidth=1.5, label='Von Mises (isotropic)')
    
    ax.set_xlabel('σ₁₁ (MPa)')
    ax.set_ylabel('σ₂₂ (MPa)')
    ax.set_title("Hill'48 vs Von Mises Yield Surface")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.savefig(os.path.join(output_dir, 'fig_yield_surface_hill48.png'))
    plt.close()
    print("  Saved: fig_yield_surface_hill48.png")


def plot_all_directions_comparison(results, output_dir):
    """
    Plot comparison of all directions on single plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    dir_colors = {'00': '#1f77b4', '45': '#ff7f0e', '90': '#2ca02c'}
    
    for direction in ['00', '45', '90']:
        for specimen in ['01', '02', '03']:
            key = f'{direction}_{specimen}'
            if key in results:
                r = results[key]
                label = f'{int(direction)}°' if specimen == '01' else None
                ax.plot(r['true_strain'], r['true_stress'],
                       color=dir_colors[direction], alpha=0.6, label=label)
    
    ax.set_xlabel('True Strain')
    ax.set_ylabel('True Stress (MPa)')
    ax.set_title('True Stress-Strain: All Directions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'fig_all_directions_comparison.png'))
    plt.close()
    print("  Saved: fig_all_directions_comparison.png")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("PhD Research: SGCC JIS G 3302 Material Characterization")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # Step 1: Load all data
    # ------------------------------------------------------------------
    print("\n[1] Loading data...")
    data = load_all_data()
    print(f"  Loaded {len(data)} datasets")
    
    for key in sorted(data.keys()):
        print(f"    {key}: {len(data[key])} points")
    
    # ------------------------------------------------------------------
    # Step 2: Plot raw engineering stress-strain
    # ------------------------------------------------------------------
    print("\n[2] Plotting engineering stress-strain curves...")
    plot_engineering_stress_strain(data, OUTPUT_DIR)
    
    # ------------------------------------------------------------------
    # Step 3: Plot DIC strain evolution
    # ------------------------------------------------------------------
    print("\n[3] Plotting DIC strain evolution...")
    plot_dic_strain_evolution(data, OUTPUT_DIR)
    
    # ------------------------------------------------------------------
    # Step 4: Compute r-values
    # ------------------------------------------------------------------
    print("\n[4] Computing Lankford r-values...")
    r_avg, r_values, r_details = compute_all_r_values(data)
    
    r0 = r_avg['00']
    r45 = r_avg['45']
    r90 = r_avg['90']
    
    print(f"\n  Final r-values:")
    print(f"    r0  = {r0:.4f}")
    print(f"    r45 = {r45:.4f}")
    print(f"    r90 = {r90:.4f}")
    
    # ------------------------------------------------------------------
    # Step 5: Plot r-value regression
    # ------------------------------------------------------------------
    print("\n[5] Plotting r-value regressions...")
    plot_r_value_regression(data, OUTPUT_DIR, r_details)
    
    # ------------------------------------------------------------------
    # Step 6: Compute Hill'48 parameters
    # ------------------------------------------------------------------
    print("\n[6] Computing Hill'48 parameters...")
    hill_params = compute_hill48_parameters(r0, r45, r90)
    
    print(f"\n  Hill'48 Coefficients:")
    print(f"    F = {hill_params['F']:.6f}")
    print(f"    G = {hill_params['G']:.6f}")
    print(f"    H = {hill_params['H']:.6f}")
    print(f"    N = {hill_params['N']:.6f}")
    print(f"\n  Abaqus R-values (Stress Ratios):")
    print(f"    R11 = {hill_params['R11']:.6f}")
    print(f"    R22 = {hill_params['R22']:.6f}")
    print(f"    R33 = {hill_params['R33']:.6f}")
    print(f"    R12 = {hill_params['R12']:.6f}")
    print(f"    R13 = {hill_params['R13']:.6f}")
    print(f"    R23 = {hill_params['R23']:.6f}")
    print(f"\n  Normal anisotropy r_bar = {hill_params['r_bar']:.4f}")
    print(f"  Planar anisotropy Δr    = {hill_params['delta_r']:.4f}")
    
    # ------------------------------------------------------------------
    # Step 7: Convert to true stress-strain & fit hardening laws
    # ------------------------------------------------------------------
    print("\n[7] Converting to true stress-strain and fitting hardening laws...")
    
    results = {}
    E_mpa = E_STANDARD_GPA * 1000  # 200 GPa -> 200000 MPa
    
    for direction in ['00', '45', '90']:
        print(f"\n  --- {int(direction)}° Direction ---")
        for specimen in ['01', '02', '03']:
            stress_key = f'stress_{direction}_{specimen}'
            if stress_key not in data:
                continue
            
            df = data[stress_key]
            eng_stress = df['Stress'].values
            eng_strain = df['Strain'].values
            
            # Find UTS for truncation (true conversion valid only up to UTS)
            uts_idx = np.argmax(eng_stress)
            
            # Truncate at UTS
            eng_stress_trunc = eng_stress[1:uts_idx+1]  # skip zero point
            eng_strain_trunc = eng_strain[1:uts_idx+1]
            
            # Convert to true
            true_stress, true_strain = engineering_to_true(
                eng_stress_trunc, eng_strain_trunc)
            
            # Compute plastic strain
            plastic_strain, true_stress_p, true_strain_p = compute_plastic_strain(
                true_stress, true_strain, E_mpa)
            
            # Find yield point
            yield_stress, yield_strain, _ = find_yield_point(
                eng_stress, eng_strain, E=E_mpa)
            
            results[f'{direction}_{specimen}'] = {
                'eng_stress': eng_stress,
                'eng_strain': eng_strain,
                'true_stress': true_stress,
                'true_strain': true_strain,
                'plastic_strain': plastic_strain,
                'true_stress_plastic': true_stress_p,
                'yield_stress': yield_stress,
                'yield_strain': yield_strain,
                'uts_stress': eng_stress[uts_idx],
                'uts_strain': eng_strain[uts_idx],
            }
            
            print(f"  Spec. {specimen}: σ_y={yield_stress:.1f} MPa, "
                  f"UTS={eng_stress[uts_idx]:.1f} MPa, "
                  f"ε_uts={eng_strain[uts_idx]:.4f}")
    
    # ------------------------------------------------------------------
    # Step 8: Fit hardening laws per direction
    # ------------------------------------------------------------------
    # Two approaches: (A) pooled data with threshold, (B) per-specimen average
    print("\n[8] Fitting hardening laws (improved: threshold + per-specimen)...")
    
    hardening_params = {}
    
    for direction in ['00', '45', '90']:
        print(f"\n  --- {int(direction)}° Direction ---")
        
        # Combine all specimens (pooled)
        all_ep = []
        all_ts = []
        for specimen in ['01', '02', '03']:
            key = f'{direction}_{specimen}'
            if key in results and len(results[key]['plastic_strain']) > 0:
                all_ep.extend(results[key]['plastic_strain'].tolist())
                all_ts.extend(results[key]['true_stress_plastic'].tolist())
        
        if not all_ep:
            continue
        
        ep = np.array(all_ep)
        ts = np.array(all_ts)
        
        # Sort
        sort_idx = np.argsort(ep)
        ep = ep[sort_idx]
        ts = ts[sort_idx]
        
        # --- Approach A: Pooled data (threshold applied inside fit functions) ---
        swift_p = fit_swift(ep, ts)
        if swift_p:
            print(f"  Swift (pooled):    K={swift_p['K']:.2f}, ε0={swift_p['eps0']:.6f}, "
                  f"n={swift_p['n']:.4f}, R²={swift_p['R2']:.4f}")
        
        voce_p = fit_voce(ep, ts)
        if voce_p:
            print(f"  Voce (pooled):     σ_y={voce_p['sigma_y']:.2f}, Q={voce_p['Q']:.2f}, "
                  f"b={voce_p['b']:.4f}, R²={voce_p['R2']:.4f}")
        
        hs_p = fit_hockett_sherby(ep, ts)
        if hs_p:
            print(f"  Hockett-Sherby:    σ_s={hs_p['sigma_s']:.2f}, σ_y={hs_p['sigma_y']:.2f}, "
                  f"m={hs_p['m']:.4f}, p={hs_p['p']:.4f}, R²={hs_p['R2']:.4f}")
        
        mv_p = fit_modified_voce(ep, ts)
        if mv_p:
            print(f"  Mod. Voce (2-term): σ_y={mv_p['sigma_y']:.2f}, Q1={mv_p['Q1']:.2f}, "
                  f"b1={mv_p['b1']:.2f}, Q2={mv_p['Q2']:.2f}, b2={mv_p['b2']:.2f}, "
                  f"R²={mv_p['R2']:.4f}")
        
        # --- Approach B: Per-specimen fit then average ---
        swift_avg = fit_per_specimen_then_average(results, direction, fit_swift, 'Swift')
        voce_avg = fit_per_specimen_then_average(results, direction, fit_voce, 'Voce')
        hs_avg = fit_per_specimen_then_average(results, direction, fit_hockett_sherby, 'Hockett-Sherby')
        
        if voce_avg:
            print(f"  Voce (per-spec avg): σ_y={voce_avg.get('sigma_y',0):.2f}, "
                  f"Q={voce_avg.get('Q',0):.2f}, b={voce_avg.get('b',0):.2f}, "
                  f"R²_mean={voce_avg['R2_mean']:.4f} ± {voce_avg['R2_std']:.4f}")
        
        hardening_params[direction] = {
            'swift': swift_p,
            'voce': voce_p,
            'hockett_sherby': hs_p,
            'modified_voce': mv_p,
            'swift_per_specimen': swift_avg,
            'voce_per_specimen': voce_avg,
            'hs_per_specimen': hs_avg,
            'plastic_strain': ep,
            'true_stress': ts
        }
    
    # ------------------------------------------------------------------
    # Step 9: Generate plots
    # ------------------------------------------------------------------
    print("\n[9] Generating plots...")
    plot_true_stress_strain(results, OUTPUT_DIR)
    plot_hardening_fits(results, OUTPUT_DIR)
    plot_all_directions_comparison(results, OUTPUT_DIR)
    
    # Yield surface
    avg_yield = np.mean([results[k]['yield_stress'] 
                        for k in results if 'yield_stress' in results[k]])
    plot_yield_surface_hill48(hill_params, avg_yield, OUTPUT_DIR)
    
    # ------------------------------------------------------------------
    # Step 10: Export results for Abaqus
    # ------------------------------------------------------------------
    print("\n[10] Exporting results for Abaqus...")
    
    # Use 0-degree direction as reference for isotropic hardening input
    if '00' in hardening_params and hardening_params['00']['swift']:
        sp = hardening_params['00']['swift']
        vp = hardening_params['00']['voce']
        
        # Generate tabular data for Abaqus
        ep_tab = np.linspace(0, 0.3, 200)
        sigma_swift = swift_law(ep_tab, sp['K'], sp['eps0'], sp['n'])
        
        # Save Abaqus isotropic hardening table (yield stress, plastic strain)
        abaqus_data = np.column_stack([sigma_swift, ep_tab])
        np.savetxt(os.path.join(OUTPUT_DIR, 'abaqus_isotropic_hardening.csv'),
                  abaqus_data, delimiter=',', 
                  header='Yield_Stress_MPa, Plastic_Strain',
                  comments='', fmt='%.6f')
        print("  Saved: abaqus_isotropic_hardening.csv")
        
        # Save Voce parameters
        if vp:
            sigma_voce = voce_law(ep_tab, vp['sigma_y'], vp['Q'], vp['b'])
            abaqus_voce = np.column_stack([sigma_voce, ep_tab])
            np.savetxt(os.path.join(OUTPUT_DIR, 'abaqus_voce_hardening.csv'),
                      abaqus_voce, delimiter=',',
                      header='Yield_Stress_MPa, Plastic_Strain',
                      comments='', fmt='%.6f')
            print("  Saved: abaqus_voce_hardening.csv")
        
        # Save Hockett-Sherby (best R²) parameters
        hs = hardening_params['00'].get('hockett_sherby')
        if hs:
            sigma_hs = hockett_sherby_law(ep_tab, hs['sigma_s'], hs['sigma_y'],
                                          hs['m'], hs['p'])
            abaqus_hs = np.column_stack([sigma_hs, ep_tab])
            np.savetxt(os.path.join(OUTPUT_DIR, 'abaqus_hockett_sherby_hardening.csv'),
                      abaqus_hs, delimiter=',',
                      header='Yield_Stress_MPa, Plastic_Strain',
                      comments='', fmt='%.6f')
            print("  Saved: abaqus_hockett_sherby_hardening.csv")
    
    # Save Hill'48 parameters
    with open(os.path.join(OUTPUT_DIR, 'hill48_parameters.txt'), 'w') as f:
        f.write("Hill'48 Anisotropic Yield Criterion Parameters\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Anisotropy source: {r_details['source']}\n")
        for source_file in r_details.get('source_files', []):
            f.write(f"  - {os.path.basename(source_file)}\n")
        f.write("\n")
        f.write(f"Lankford r-values:\n")
        f.write(f"  r0  = {r0:.6f}\n")
        f.write(f"  r45 = {r45:.6f}\n")
        f.write(f"  r90 = {r90:.6f}\n\n")
        f.write(f"Normal anisotropy:  r_bar = {hill_params['r_bar']:.6f}\n")
        f.write(f"Planar anisotropy:  Δr    = {hill_params['delta_r']:.6f}\n\n")
        f.write(f"Hill'48 Coefficients:\n")
        f.write(f"  F = {hill_params['F']:.6f}\n")
        f.write(f"  G = {hill_params['G']:.6f}\n")
        f.write(f"  H = {hill_params['H']:.6f}\n")
        f.write(f"  N = {hill_params['N']:.6f}\n\n")
        f.write(f"Abaqus Stress Ratios:\n")
        f.write(f"  R11 = {hill_params['R11']:.6f}\n")
        f.write(f"  R22 = {hill_params['R22']:.6f}\n")
        f.write(f"  R33 = {hill_params['R33']:.6f}\n")
        f.write(f"  R12 = {hill_params['R12']:.6f}\n")
        f.write(f"  R13 = {hill_params['R13']:.6f}\n")
        f.write(f"  R23 = {hill_params['R23']:.6f}\n")
    print("  Saved: hill48_parameters.txt")
    
    # Save all results summary
    with open(os.path.join(OUTPUT_DIR, 'results_summary.txt'), 'w') as f:
        f.write("COMPLETE MATERIAL CHARACTERIZATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Material: SGCC JIS G 3302\n")
        f.write(f"Specimen: DIN 50125, Gauge: {GAUGE_LENGTH}x{GAUGE_WIDTH}x{THICKNESS} mm\n")
        f.write(f"Reference E = {E_STANDARD_GPA} GPa, ν = {NU_STANDARD}\n\n")
        f.write("ANISOTROPY SOURCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"{r_details['source']}\n")
        for source_file in r_details.get('source_files', []):
            f.write(f"  - {os.path.basename(source_file)}\n")
        f.write("\n")
        
        f.write("MECHANICAL PROPERTIES\n")
        f.write("-" * 40 + "\n")
        for direction in ['00', '45', '90']:
            f.write(f"\n{int(direction)}° Direction:\n")
            for specimen in ['01', '02', '03']:
                key = f'{direction}_{specimen}'
                if key in results:
                    r = results[key]
                    f.write(f"  Spec. {specimen}: σ_y = {r['yield_stress']:.1f} MPa, "
                           f"UTS = {r['uts_stress']:.1f} MPa\n")
        
        f.write("\n\nHARDENING LAW PARAMETERS (with εp threshold = {:.4f})\n".format(PLASTIC_STRAIN_THRESHOLD))
        f.write("-" * 40 + "\n")
        for direction in ['00', '45', '90']:
            if direction in hardening_params:
                hp = hardening_params[direction]
                f.write(f"\n{int(direction)}° Direction:\n")
                if hp['swift']:
                    sp = hp['swift']
                    f.write(f"  Swift (pooled):    K={sp['K']:.2f}, ε0={sp['eps0']:.6f}, "
                           f"n={sp['n']:.4f} (R²={sp['R2']:.4f})\n")
                if hp['voce']:
                    vp = hp['voce']
                    f.write(f"  Voce (pooled):     σ_y={vp['sigma_y']:.2f}, Q={vp['Q']:.2f}, "
                           f"b={vp['b']:.4f} (R²={vp['R2']:.4f})\n")
                if hp.get('hockett_sherby'):
                    hs = hp['hockett_sherby']
                    f.write(f"  Hockett-Sherby:    σ_s={hs['sigma_s']:.2f}, σ_y={hs['sigma_y']:.2f}, "
                           f"m={hs['m']:.4f}, p={hs['p']:.4f} (R²={hs['R2']:.4f})\n")
                if hp.get('modified_voce'):
                    mv = hp['modified_voce']
                    f.write(f"  Mod. Voce:         σ_y={mv['sigma_y']:.2f}, "
                           f"Q1={mv['Q1']:.2f}, b1={mv['b1']:.2f}, "
                           f"Q2={mv['Q2']:.2f}, b2={mv['b2']:.2f} "
                           f"(R²={mv['R2']:.4f})\n")
                if hp.get('voce_per_specimen'):
                    va = hp['voce_per_specimen']
                    f.write(f"  Voce (per-spec):   R²_mean={va['R2_mean']:.4f} ± {va['R2_std']:.4f}\n")
        
        f.write(f"\n\nHILL'48 PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"r0={r0:.4f}, r45={r45:.4f}, r90={r90:.4f}\n")
        f.write(f"R11={hill_params['R11']:.4f}, R22={hill_params['R22']:.4f}, "
               f"R33={hill_params['R33']:.4f}, R12={hill_params['R12']:.4f}\n")

        # ----------------------------------------------------------------
        # Pooled 0.2% offset yield stresses (canonical yield-criterion targets)
        # ----------------------------------------------------------------
        offset_yield_pooled = {}
        f.write(f"\n\nYIELD STRESS DEFINITIONS (READ CAREFULLY)\n")
        f.write("-" * 40 + "\n")
        f.write("Two distinct yield-stress quantities are reported in this repo:\n")
        f.write("  (a) sigma_y,0.2  = 0.2%-offset yield stress from the engineering\n"
                "                     stress-strain curve. This is the canonical\n"
                "                     target for the Hill'48 stress-calibration\n"
                "                     and Yld2000-2d yield-surface identification.\n")
        f.write("  (b) sigma0_Voce  = Voce flow stress extrapolated to plastic\n"
                "                     strain zero (parameter sigma_y in Eq. (8)).\n"
                "                     This is the initial-yield parameter of the\n"
                "                     Chaboche combined hardening kernel; it is\n"
                "                     systematically lower than sigma_y,0.2.\n")
        f.write("\nPOOLED 0.2%-OFFSET YIELD STRESSES (mean of 3 specimens)\n")
        for direction in ['00', '45', '90']:
            vals = [results[f'{direction}_{s}']['yield_stress']
                    for s in ['01', '02', '03']
                    if f'{direction}_{s}' in results]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                offset_yield_pooled[direction] = {'mean': mean, 'std': std,
                                                   'n': len(vals),
                                                   'specimens': vals}
                f.write(f"  sigma_y,0.2 ({int(direction)} deg) = "
                        f"{mean:.2f} +- {std:.2f} MPa (n={len(vals)})\n")

        if all(d in offset_yield_pooled for d in ['00', '45', '90']):
            s0  = offset_yield_pooled['00']['mean']
            s45 = offset_yield_pooled['45']['mean']
            s90 = offset_yield_pooled['90']['mean']
            f.write(f"\n  Ratios: sigma45/sigma0 = {s45/s0:.6f}, "
                    f"sigma90/sigma0 = {s90/s0:.6f}\n")

        f.write("\nPOOLED VOCE FLOW STRESSES (sigma0_Voce, for hardening kernel)\n")
        for direction in ['00', '45', '90']:
            if direction in hardening_params and hardening_params[direction].get('voce'):
                vp = hardening_params[direction]['voce']
                f.write(f"  sigma0_Voce ({int(direction)} deg) = "
                        f"{vp['sigma_y']:.2f} MPa (Voce pooled fit)\n")
    
    print("  Saved: results_summary.txt")

    # Separate machine-readable offset-yield file consumed by barlat_yld2000.py
    # and optimize_hardening_multidir.py downstream.
    if all(d in offset_yield_pooled for d in ['00', '45', '90']):
        offset_yield_path = os.path.join(OUTPUT_DIR, 'offset_yield_pooled.json')
        with open(offset_yield_path, 'w') as f:
            import json as _json
            _json.dump({
                'definition': '0.2%-offset yield stress from engineering stress-strain',
                'E_GPa': float(E_STANDARD_GPA),
                'offset': 0.002,
                'directions': {
                    d: {
                        'mean_MPa': offset_yield_pooled[d]['mean'],
                        'std_MPa': offset_yield_pooled[d]['std'],
                        'n_specimens': offset_yield_pooled[d]['n'],
                        'specimens_MPa': offset_yield_pooled[d]['specimens'],
                    }
                    for d in ['00', '45', '90']
                },
                'ratios': {
                    'sigma45_over_sigma0':
                        offset_yield_pooled['45']['mean'] / offset_yield_pooled['00']['mean'],
                    'sigma90_over_sigma0':
                        offset_yield_pooled['90']['mean'] / offset_yield_pooled['00']['mean'],
                },
            }, f, indent=2)
        print(f"  Saved: offset_yield_pooled.json")

    with open(os.path.join(OUTPUT_DIR, 'anisotropy_diagnostic.txt'), 'w') as f:
        f.write("ANISOTROPY SOURCE TRACE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {r_details['source']}\n")
        for source_file in r_details.get('source_files', []):
            f.write(f"  - {source_file}\n")
        f.write("\n")
        for direction in ['00', '45', '90']:
            f.write(f"{direction} degree direction\n")
            f.write("-" * 40 + "\n")
            for specimen in ['01', '02', '03']:
                detail = r_details.get(f'r_{direction}_{specimen}')
                if not detail:
                    continue
                f.write(f"Specimen {specimen}: r = {detail['r']:.6f} ({detail['method']})\n")
                if 'python_diagnostic' in detail:
                    diag = detail['python_diagnostic']
                    f.write(
                        f"  diagnostic CSV recomputation = {diag['r']:.6f} "
                        f"(delta = {diag['r'] - detail['r']:+.6f})\n"
                    )
            f.write("\n")
    print("  Saved: anisotropy_diagnostic.txt")
    
    print("\n" + "=" * 70)
    print("DATA PROCESSING COMPLETE")
    print("=" * 70)
    
    return data, results, r_avg, hill_params, hardening_params


if __name__ == '__main__':
    data, results, r_avg, hill_params, hardening_params = main()
