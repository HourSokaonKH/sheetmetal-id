# -*- coding: utf-8 -*-
"""
=============================================================================
Post-Processing & Visualization Script
For Abaqus Combined Hardening Optimization Results
=============================================================================
This script can be run OUTSIDE Abaqus (standard Python 3 with matplotlib).
It processes the optimization results, generates publication-quality plots,
and creates comparison figures for the thesis.

Usage:
  python3 post_processing.py

Author: HOUR Sokaon
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
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
OPT_DIR = os.path.join(DATA_DIR, 'optimization_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Material constants
E_YOUNG = 200000.0   # MPa
THICKNESS = 1.5      # mm
WIDTH = 20.0         # mm
AREA = WIDTH * THICKNESS

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

# ============================================================================
# HARDENING LAW FUNCTIONS
# ============================================================================

def swift_law(eps_p, K, eps0, n):
    return K * (eps0 + eps_p)**n

def voce_law(eps_p, sigma_y, Q, b):
    return sigma_y + Q * (1.0 - np.exp(-b * eps_p))

def combined_hardening_stress(eps_p, sigma0, C1, gamma1, C2, gamma2, Q_inf, b_iso):
    """
    Total stress from combined isotropic-kinematic hardening model.
    
    Isotropic part: sigma_y(eps_p) = sigma0 + Q_inf*(1 - exp(-b*eps_p))
    Kinematic part: alpha = sum_i (C_i/gamma_i)*(1 - exp(-gamma_i*eps_p))
    
    For monotonic uniaxial tension:
    sigma = sigma_y(eps_p) + alpha(eps_p)
    sigma = sigma0 + Q_inf*(1-exp(-b*ep)) + C1/g1*(1-exp(-g1*ep)) + C2/g2*(1-exp(-g2*ep))
    """
    iso = sigma0 + Q_inf * (1.0 - np.exp(-b_iso * eps_p))
    kin1 = (C1 / gamma1) * (1.0 - np.exp(-gamma1 * eps_p))
    kin2 = (C2 / gamma2) * (1.0 - np.exp(-gamma2 * eps_p))
    return iso + kin1 + kin2


def backstress_evolution(eps_p, C1, gamma1, C2, gamma2):
    """
    Evolution of backstress components.
    alpha_i = C_i/gamma_i * (1 - exp(-gamma_i * eps_p))
    """
    alpha1 = (C1 / gamma1) * (1.0 - np.exp(-gamma1 * eps_p))
    alpha2 = (C2 / gamma2) * (1.0 - np.exp(-gamma2 * eps_p))
    alpha_total = alpha1 + alpha2
    return alpha1, alpha2, alpha_total


# ============================================================================
# DATA LOADING
# ============================================================================

def load_stress_data(filepath):
    """Load stress-strain CSV (Time,Load,Stress,Strain)."""
    df = pd.read_csv(filepath)
    df.columns = ['Time', 'Load', 'Stress', 'Strain']
    return df

def load_dic_data(filepath):
    """Load DIC strain CSV."""
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


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_specimen_geometry():
    """
    Plot the DIN 50125 dog-bone specimen geometry with dimensions.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Full specimen outline (simplified dog-bone)
    # Total length = 250mm, width at grip = 40mm, gauge width = 20mm
    # Gauge length = 80mm, radius = 25mm
    
    total_L = 250
    grip_W = 40
    gauge_W = 20
    gauge_L = 80
    R = 25
    
    # Symmetric about center
    cx = total_L / 2
    cy = 0
    
    # Outer boundary (simplified)
    from matplotlib.patches import FancyBboxPatch
    
    # Grip regions
    grip_len = (total_L - gauge_L) / 2 - R
    
    # Draw simplified shape
    x_pts = [0, grip_len, grip_len + R, cx - gauge_L/2, cx + gauge_L/2,
             total_L - grip_len - R, total_L - grip_len, total_L]
    
    # Top boundary
    y_top = [grip_W/2, grip_W/2, gauge_W/2, gauge_W/2, gauge_W/2,
             gauge_W/2, grip_W/2, grip_W/2]
    y_bot = [-y for y in y_top]
    
    ax.fill_between(x_pts, y_top, y_bot, color='lightsteelblue', 
                    edgecolor='navy', linewidth=2)
    
    # Dimension lines
    def dim_line(ax, x1, y1, x2, y2, text, offset=3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + offset, text, ha='center', va='bottom',
               fontsize=10, color='red', fontweight='bold')
    
    # Total length
    dim_line(ax, 0, -grip_W/2-8, total_L, -grip_W/2-8, '250 mm', -5)
    
    # Gauge length
    dim_line(ax, cx-gauge_L/2, grip_W/2+3, cx+gauge_L/2, grip_W/2+3, '80 mm')
    
    # Gauge width
    dim_line(ax, total_L+5, -gauge_W/2, total_L+5, gauge_W/2, '20 mm', 2)
    
    # Grip width
    dim_line(ax, -5, -grip_W/2, -5, grip_W/2, '40 mm', 2)
    
    # Thickness annotation
    ax.text(cx, -gauge_W/2 - 5, 't = 1.5 mm', ha='center', fontsize=10,
           color='darkgreen', fontweight='bold')
    
    ax.set_xlim(-20, total_L + 20)
    ax.set_ylim(-grip_W/2 - 15, grip_W/2 + 15)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Width (mm)')
    ax.set_title('DIN 50125 Dog-bone Specimen Geometry - SGCC JIS G 3302')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_specimen_geometry.png'))
    plt.close()
    print("  Saved: fig_specimen_geometry.png")


def plot_combined_hardening_components(sigma0, C1, gamma1, C2, gamma2,
                                       Q_inf, b_iso):
    """
    Plot the decomposition of combined hardening into isotropic and
    kinematic components.
    """
    eps_p = np.linspace(0, 0.3, 500)
    
    # Isotropic component
    iso = sigma0 + Q_inf * (1 - np.exp(-b_iso * eps_p))
    
    # Backstress components
    alpha1, alpha2, alpha_total = backstress_evolution(eps_p, C1, gamma1, C2, gamma2)
    
    # Total
    total = iso + alpha_total
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: All components
    ax1.plot(eps_p, total, 'k-', linewidth=2.5, label='Total (σ)')
    ax1.plot(eps_p, iso, 'b--', linewidth=2, label='Isotropic (σ₀ + R)')
    ax1.plot(eps_p, alpha_total, 'r--', linewidth=2, label='Kinematic (α)')
    ax1.fill_between(eps_p, iso, total, alpha=0.15, color='red',
                    label='Kinematic contribution')
    ax1.fill_between(eps_p, 0, iso, alpha=0.1, color='blue',
                    label='Isotropic contribution')
    
    ax1.set_xlabel('Equivalent Plastic Strain (ε̄ᵖ)')
    ax1.set_ylabel('Stress (MPa)')
    ax1.set_title('Combined Hardening Decomposition')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Right: Backstress components
    ax2.plot(eps_p, alpha_total, 'r-', linewidth=2.5, label='α (total)')
    ax2.plot(eps_p, alpha1, 'g--', linewidth=2, label='α₁ (C₁/γ₁)')
    ax2.plot(eps_p, alpha2, 'm--', linewidth=2, label='α₂ (C₂/γ₂)')
    
    # Saturation values
    sat1 = C1 / gamma1
    sat2 = C2 / gamma2
    ax2.axhline(y=sat1, color='g', linestyle=':', alpha=0.5)
    ax2.axhline(y=sat2, color='m', linestyle=':', alpha=0.5)
    ax2.axhline(y=sat1+sat2, color='r', linestyle=':', alpha=0.5)
    
    ax2.text(0.25, sat1+2, 'C₁/γ₁ = %.1f' % sat1, color='g', fontsize=10)
    ax2.text(0.25, sat2+2, 'C₂/γ₂ = %.1f' % sat2, color='m', fontsize=10)
    
    ax2.set_xlabel('Equivalent Plastic Strain (ε̄ᵖ)')
    ax2.set_ylabel('Backstress (MPa)')
    ax2.set_title('Backstress Evolution (2 Components)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_combined_hardening_components.png'))
    plt.close()
    print("  Saved: fig_combined_hardening_components.png")


def plot_convergence_history():
    """
    Plot optimization convergence from history CSV.
    """
    hist_file = os.path.join(OPT_DIR, 'convergence_history.csv')
    if not os.path.exists(hist_file):
        print("  No convergence history found. Delegating to generate_fig_convergence.py")
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'generate_fig_convergence.py')])
        return
    
    df = pd.read_csv(hist_file)
    success = df[df['Status'] == 'ok']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Cost convergence
    ax1 = axes[0]
    ax1.semilogy(success['Iteration'], success['NRMSE'], 'b.-', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost Function (NRMSE)')
    ax1.set_title('Optimization Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Parameter evolution
    ax2 = axes[1]
    params = ['sigma0', 'C1', 'gamma1', 'C2', 'gamma2']
    for p in params:
        ax2.plot(success['Iteration'], success[p], '.-', markersize=3, label=p)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_convergence.png'))
    plt.close()
    print("  Saved: fig_convergence.png")


def plot_optimization_comparison(sigma0, C1, gamma1, C2, gamma2,
                                  Q_inf, b_iso):
    """
    Plot experimental vs optimized simulation comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    dir_colors = {'00': '#1f77b4', '45': '#ff7f0e', '90': '#2ca02c'}
    
    for idx, direction in enumerate(['00', '45', '90']):
        ax = axes[idx]
        
        # Load all experimental specimens for this direction
        for specimen in ['01', '02', '03']:
            filepath = os.path.join(DATA_DIR, f'stress-{direction}-{specimen}.csv')
            if os.path.exists(filepath):
                df = load_stress_data(filepath)
                eng_s = df['Stress'].values
                eng_e = df['Strain'].values
                
                uts_idx = np.argmax(eng_s)
                eng_s = eng_s[1:uts_idx+1]
                eng_e = eng_e[1:uts_idx+1]
                
                true_strain = np.log(1 + eng_e)
                true_stress = eng_s * (1 + eng_e)
                
                plastic_strain = true_strain - true_stress / E_YOUNG
                mask = plastic_strain > 0
                
                if specimen == '01':
                    label = f'Experimental ({int(direction)}°)'
                else:
                    label = None
                ax.plot(true_strain, true_stress, color=dir_colors[direction],
                       alpha=0.4, linewidth=1, label=label)
        
        # Model prediction
        eps_p = np.linspace(0, 0.25, 300)
        model_stress = combined_hardening_stress(
            eps_p, sigma0, C1, gamma1, C2, gamma2, Q_inf, b_iso)
        model_total_strain = eps_p + model_stress / E_YOUNG
        
        ax.plot(model_total_strain, model_stress, 'k--', linewidth=2.5,
               label='Combined Hardening Model')
        
        ax.set_xlabel('True Strain')
        ax.set_ylabel('True Stress (MPa)')
        ax.set_title(f'{int(direction)}° Rolling Direction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_optimization_comparison.png'))
    plt.close()
    print("  Saved: fig_optimization_comparison.png")


def plot_fe_model_schematic():
    """
    Create a schematic of the FE model with BC annotations.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    W = 10   # half-width
    L = 40   # half-length
    
    # Mesh grid
    nx, ny = 10, 40
    for i in range(nx + 1):
        x = i * W / nx
        ax.plot([x, x], [0, L], 'b-', linewidth=0.5, alpha=0.3)
    for j in range(ny + 1):
        y = j * L / ny
        ax.plot([0, W], [y, y], 'b-', linewidth=0.5, alpha=0.3)
    
    # Boundary
    ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], 'k-', linewidth=2)
    
    # Symmetry BC - X (left edge)
    for y in np.linspace(2, L-2, 8):
        ax.annotate('', xy=(-2, y), xytext=(0, y),
                   arrowprops=dict(arrowstyle='-|>', color='red', lw=1))
    ax.text(-3, L/2, 'U₁=0\n(Sym X)', ha='right', va='center',
           fontsize=10, color='red', fontweight='bold')
    
    # Symmetry BC - Y (bottom edge)
    for x in np.linspace(1, W-1, 5):
        ax.annotate('', xy=(x, -2), xytext=(x, 0),
                   arrowprops=dict(arrowstyle='-|>', color='green', lw=1))
    ax.text(W/2, -4, 'U₂=0 (Sym Y)', ha='center', va='top',
           fontsize=10, color='green', fontweight='bold')
    
    # Displacement BC (top edge)
    for x in np.linspace(1, W-1, 5):
        ax.annotate('', xy=(x, L+3), xytext=(x, L),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(W/2, L+5, 'U₂ = δ (Displacement)', ha='center',
           fontsize=11, color='blue', fontweight='bold')
    
    # Center element marker
    cx, cy = W/2, L/2
    ax.plot(cx, cy, 'r*', markersize=15, zorder=10)
    ax.text(cx + 1.5, cy, 'Center\nElement', fontsize=9, color='red')
    
    # Dimensions
    ax.annotate('', xy=(W+2, 0), xytext=(W+2, L),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(W+3, L/2, f'{L} mm', va='center', fontsize=10)
    
    ax.annotate('', xy=(0, -7), xytext=(W, -7),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(W/2, -9, f'{W} mm', ha='center', fontsize=10)
    
    ax.set_xlim(-8, W+8)
    ax.set_ylim(-12, L+8)
    ax.set_aspect('equal')
    ax.set_title('FE Model: Quarter Symmetry\nCPS4R Elements, Plane Stress',
                fontsize=13)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_fe_model_schematic.png'))
    plt.close()
    print("  Saved: fig_fe_model_schematic.png")


def plot_inverse_method_flowchart():
    """
    Create a flowchart of the inverse identification procedure.
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    
    boxes = [
        (5, 13, 'Experimental Data\n(UTM Stress + DIC Strain)', '#AED6F1'),
        (5, 11.5, 'Engineering → True Stress-Strain\nConversion', '#A9DFBF'),
        (5, 10, 'Fit Swift/Voce Hardening Law\n(Isotropic Parameters)', '#A9DFBF'),
        (5, 8.5, 'Initial Guess\n[σ₀, C₁, γ₁, C₂, γ₂]', '#F9E79F'),
        (5, 7, 'Abaqus FE Simulation\n(2D Plane Stress, CPS4R)', '#F5B7B1'),
        (5, 5.5, 'Extract σ-ε from\nCenter Element', '#F5B7B1'),
        (5, 4, 'Compute NRMSE\n(Simulation vs Experiment)', '#D7BDE2'),
        (5, 2.5, 'Converged?\n(NRMSE < tolerance)', '#F9E79F'),
        (5, 1, 'Optimized Parameters\n[σ₀, C₁, γ₁, C₂, γ₂]', '#AED6F1'),
        (8.5, 4, 'Nelder-Mead\nUpdate Parameters', '#FADBD8'),
    ]
    
    for (x, y, text, color) in boxes:
        w, h = 3.5, 1.0
        if 'Converged' in text:
            # Diamond shape
            diamond = plt.Polygon([(x, y+0.6), (x+1.5, y), (x, y-0.6), (x-1.5, y)],
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            from matplotlib.patches import FancyBboxPatch
            box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrows = [
        (5, 12.5, 5, 12.0),    # Data → Convert
        (5, 11.0, 5, 10.5),    # Convert → Fit
        (5, 9.5, 5, 9.0),      # Fit → Initial
        (5, 8.0, 5, 7.5),      # Initial → Abaqus
        (5, 6.5, 5, 6.0),      # Abaqus → Extract
        (5, 5.0, 5, 4.5),      # Extract → NRMSE
        (5, 3.4, 5, 2.1),      # NRMSE → Converged  
        (5, 1.9, 5, 1.5),      # Converged → Result (Yes)
        (6.5, 2.5, 7.0, 3.5),  # Converged → NM (No)
        (8.5, 4.5, 8.5, 7.0),  # NM → back to Abaqus
        (8.5, 7.0, 6.75, 7.0), # -> Abaqus input
    ]
    
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Labels
    ax.text(4.5, 1.7, 'Yes', fontsize=10, color='green', fontweight='bold')
    ax.text(6.8, 2.8, 'No', fontsize=10, color='red', fontweight='bold')
    
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 14.5)
    ax.set_title('Inverse Parameter Identification Flowchart\n'
                'Nelder-Mead + Abaqus FEA', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_inverse_flowchart.png'))
    plt.close()
    print("  Saved: fig_inverse_flowchart.png")


def plot_r_value_polar():
    """
    Plot r-value variation with rolling direction (polar plot).
    """
    # From multi-zone DIC extraction, weighted pooling (r0=0.712, r45=0.800, r90=0.742)
    r_values = {'0': 0.712, '45': 0.800, '90': 0.742}
    
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    # Symmetric: r(180-theta) = r(theta) for orthotropy
    r_vals = [r_values['0'], r_values['45'], r_values['90'],
              r_values['45'], r_values['0'], r_values['45'],
              r_values['90'], r_values['45'], r_values['0']]
    
    angles_rad = np.deg2rad(angles_deg)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.plot(angles_rad, r_vals, 'b-o', linewidth=2, markersize=8)
    ax.fill(angles_rad, r_vals, alpha=0.1, color='blue')
    
    # Reference (isotropic r=1)
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100), 'r--', 
           linewidth=1, label='Isotropic (r=1)')
    
    ax.set_title('Lankford r-value vs Rolling Direction\nSGCC JIS G 3302',
                fontsize=13, pad=20)
    ax.set_rlabel_position(22.5)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_r_value_polar.png'))
    plt.close()
    print("  Saved: fig_r_value_polar.png")


def generate_all_thesis_figures():
    """
    Generate all figures needed for the PhD thesis.
    """
    print("=" * 70)
    print("GENERATING THESIS FIGURES")
    print("=" * 70)
    
    # Figure 1: Specimen geometry
    print("\n[1] Specimen geometry...")
    plot_specimen_geometry()
    
    # Figure 2: FE model schematic
    print("\n[2] FE model schematic...")
    plot_fe_model_schematic()
    
    # Figure 3: Inverse method flowchart
    print("\n[3] Inverse identification flowchart...")
    plot_inverse_method_flowchart()
    
    # Figure 4: r-value polar plot
    print("\n[4] r-value polar plot...")
    plot_r_value_polar()
    
    # Figure 5: Combined hardening components
    # Using representative parameters (will be updated after optimization)
    print("\n[5] Combined hardening components...")
    plot_combined_hardening_components(
        sigma0=312.35, C1=502.71, gamma1=499.72,
        C2=100.37, gamma2=199.44,
        Q_inf=335.16, b_iso=3.95
    )
    
    # Figure 6: Optimization comparison
    print("\n[6] Experimental vs model comparison...")
    plot_optimization_comparison(
        sigma0=312.35, C1=502.71, gamma1=499.72,
        C2=100.37, gamma2=199.44,
        Q_inf=335.16, b_iso=3.95
    )
    
    # Figure 7: Convergence
    print("\n[7] Convergence history...")
    plot_convergence_history()

    # -------------------------------------------------------------------
    # Run supplementary analysis scripts (generate additional figures)
    # -------------------------------------------------------------------
    print("\n--- Running supplementary analysis scripts ---")

    # Barlat Yld2000-2d yield surface comparison
    print("\n[8] Barlat Yld2000-2d analysis...")
    try:
        import barlat_yld2000
        barlat_yld2000.main()
        print("    (3 figures + parameter files)")
    except Exception as e:
        print(f"    Skipped: {e}")

    # Multi-direction optimization & algorithm comparison
    print("\n[9] Multi-direction optimization & algorithm comparison...")
    try:
        import multi_objective_optimization
        multi_objective_optimization.main()
        print("    (4 figures + result files)")
    except Exception as e:
        print(f"    Skipped: {e}")

    # Monte Carlo uncertainty analysis
    print("\n[10] Monte Carlo uncertainty analysis...")
    try:
        import sensitivity_analysis
        sensitivity_analysis.main()
        print("    (5 figures + result files)")
    except Exception as e:
        print(f"    Skipped: {e}")

    # Exy validation & alignment quality
    print("\n[11] Exy validation & alignment quality...")
    try:
        import exy_validation
        exy_validation.main()
        print("    (4 figures)")
    except Exception as e:
        print(f"    Skipped: {e}")

    print("\n" + "=" * 70)
    print("ALL THESIS FIGURES GENERATED")
    print(f"Output directory: {OUTPUT_DIR}")
    n_figs = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    print(f"Total figures: {n_figs}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    generate_all_thesis_figures()
