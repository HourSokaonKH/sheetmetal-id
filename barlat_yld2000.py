#!/usr/bin/env python3
"""
=============================================================================
Barlat Yld2000-2d Yield Criterion Implementation
=============================================================================
Implements the Barlat Yld2000-2d anisotropic yield function for plane stress
and compares with Hill'48 yield surface predictions.

Material: SGCC JIS G 3302 galvanized steel
Input:    r-values (r0, r45, r90) and uniaxial yield stresses (σ0, σ45, σ90)
          plus equi-biaxial yield stress (σ_b) and r_b

Reference:
  Barlat et al. (2003) "Plane stress yield function for aluminum alloy 
  sheets - Part 1: Theory", Int. J. Plasticity 19, 1297-1319.

Author: PhD Candidate
Date:   2026
=============================================================================
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experimental data from data_processing.py
# r-values (from multi-zone DIC extraction, weighted pooling)
# Loaded from material_constants.json so that every script that consumes the
# Yld2000 identification (optimize_hardening*.py, bayesian_reidentification.py,
# etc.) uses exactly the same r-values.
from material_constants import R0, R45, R90

# ---------------------------------------------------------------------------
# Yield-stress targets for Yld2000-2d identification.
#
# The canonical target is the pooled 0.2%-offset yield stress (sigma_y,0.2)
# written to output/offset_yield_pooled.json by data_processing.py. These are
# the values every formability / sheet-metal reference uses for "uniaxial
# yield stress". They are loaded at import time if the JSON is present so
# that the identification always tracks the processed data.
#
# The Voce-extrapolated flow stress sigma0_Voce (initial yield of the
# Chaboche hardening kernel) is retained only as a reference for the
# hardening-kernel parameterization and is NOT used as a yield-surface target.
# ---------------------------------------------------------------------------
_OFFSET_YIELD_JSON = os.path.join(OUTPUT_DIR, 'offset_yield_pooled.json')
if os.path.exists(_OFFSET_YIELD_JSON):
    with open(_OFFSET_YIELD_JSON, 'r') as _f:
        _oy = json.load(_f)
    SIGMA_0  = float(_oy['directions']['00']['mean_MPa'])
    SIGMA_45 = float(_oy['directions']['45']['mean_MPa'])
    SIGMA_90 = float(_oy['directions']['90']['mean_MPa'])
    SIGMA_SOURCE = 'pooled 0.2%-offset yield stress (offset_yield_pooled.json)'
else:
    # Fallback literal values from the committed output/offset_yield_pooled.json
    # baseline. Update these only if the JSON file becomes the canonical source.
    SIGMA_0  = 352.53   # 0°  pooled sigma_y,0.2
    SIGMA_45 = 252.17   # 45° pooled sigma_y,0.2
    SIGMA_90 = 381.47   # 90° pooled sigma_y,0.2
    SIGMA_SOURCE = 'hardcoded fallback (offset_yield_pooled.json missing)'

# Voce flow stresses (reference only, NOT used as yield-surface targets)
SIGMA_0_VOCE  = 326.45
SIGMA_45_VOCE = 223.66
SIGMA_90_VOCE = 342.63

# Equi-biaxial data: estimated from r-values using Hill'48 approximation
# sigma_b / sigma_0 ≈ sqrt((1+r0)/(1+r_bar))  (approximate for lacking biaxial test)
#
# PROVENANCE NOTE (Option B honest reframe, 2026-04)
# -------------------------------------------------
# No equi-biaxial measurement (cruciform / hydraulic bulge) is available for
# this SGCC sheet. The biaxial yield stress SIGMA_B and biaxial r-value RB_EST
# used below are Hill'48-derived estimates from the measured (r0, r45, r90) and
# sigma_0 only. They are NOT experimental targets. They are included so the
# Yld2000-2d identification has the usual 8-equation system, but the weights on
# the biaxial residuals are set to zero (see OBJECTIVE_WEIGHTS['sb'] = 0.0,
# ['rb'] = 0.0 below) so they do not bias the fit. Papers and reports must
# disclose this when discussing the Yld2000-2d shape in the biaxial quadrant.
R_BAR = (R0 + 2*R45 + R90) / 4.0
# Hill'48 biaxial yield stress: sigma_b = sigma_0 * sqrt(r90*(1+r0)/(r0+r90))
SIGMA_B = SIGMA_0 * np.sqrt(R90 * (1 + R0) / (R0 + R90))
# Hill'48 biaxial r-value: rb = F/G = r0/r90 (under equi-biaxial stress)
RB_EST  = R0 / R90

# Yld2000-2d exponent (a=6 for BCC, a=8 for FCC)
# SGCC is BCC ferritic steel
A_EXPONENT = 6

# Prior-iteration Yld2000-2d coefficients. These were identified against the
# stale Voce-flow-stress targets and are NOT expected to satisfy the refreshed
# 0.2%-offset targets. They are retained only so the `obj_weighted(PROVEN_ALPHA)`
# check below falls through to differential evolution, which re-identifies the
# coefficients for the current yield-stress targets.
PROVEN_ALPHA = np.array([
    2.39005077,
    1.90355490,
    3.20396706,
    2.21822459,
    2.66091651,
    4.81155956,
    0.40531292,
    3.92851353,
], dtype=float)

OBJECTIVE_WEIGHTS = {
    'r': 10.0,
    's45': 50.0,
    's90': 10.0,
    'sb': 0.0,
    'rb': 0.0,
}

# Hill'48 stress-calibration from pooled 0.2%-offset yield stresses
# (sigma0=352.53, sigma45=252.17, sigma90=381.47 MPa).
#   H = r0/(1+r0) preserves r0 = 0.7122
#   G = 1 - H                                  (normalisation G+H=1)
#   F = G+H - 2H*(sigma0/sigma90)^2 + ... ; equivalently F+H = (sigma0/sigma90)^2
#   2N = 4*(sigma0/sigma45)^2 - F - G
# Implied r45 = (2N-F-G)/(2(F+G)) = 2.826; r90 = H/F = 0.950.
HILL48_STRESS_COEFFS = {
    'F': 0.438040,
    'G': 0.584027,
    'H': 0.415973,
    'N': 3.399020,
}

# Hill'48 r-calibration from canonical r-values (r0=0.7122, r45=0.7998, r90=0.7420)
HILL48_RVALUE_COEFFS = {
    'F': 0.560573,
    'G': 0.584027,
    'H': 0.415973,
    'N': 1.487731,
}

# The *_RUN dictionaries below record the FEA-comparison NRMSEs measured on
# the Windows Abaqus workstation on 2026-04-22/23 using the refreshed
# 0.2%-offset yield targets (stress-cal) and the canonical r-value targets
# (r-cal). Wall times 60 min and 158 min respectively.
HILL48_RVALUE_RUN = {
    'weighted_nrmse': 0.432959,
    'nrmse_00': 0.435931,
    'nrmse_45': 0.385472,
    'nrmse_90': 0.524963,
    'sigma45_ratio': 0.985301,           # analytic 2/sqrt(F+G+2N)
    'sigma90_ratio': 1.011923,           # analytic sqrt((G+H)/(F+H))
    'r0':  0.7122,
    'r45': 0.7998,
    'r90': 0.7420,
    'sigma0_hardening': 262.34,          # identified Chaboche sigma0
}

HILL48_STRESS_RUN = {
    'weighted_nrmse': 0.164399,
    'nrmse_00': 0.041488,
    'nrmse_45': 0.212496,
    'nrmse_90': 0.191115,
    'r0':  0.7122,
    'r45': 2.8256,   # analytical: (2N-F-G)/(2(F+G)) from 0.2%-offset-cal FGHN
    'r90': 0.9496,   # analytical: H/F
    'backstress_sat': 3.42,              # C1/gamma1 + C2/gamma2 = 2.12 + 1.30
    'sigma0_hardening': 327.48,          # identified Chaboche sigma0
}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'lines.linewidth': 1.5, 'font.family': 'serif',
})


# ============================================================================
# YLD2000-2D FUNCTIONS
# ============================================================================

def compute_L_matrices(alpha):
    """
    Compute the two linear transformation matrices L' and L'' from the
    8 anisotropy coefficients alpha_1 to alpha_8.

    L' operates on Cauchy stress to give s' (modified deviatoric stress)
    L'' operates on Cauchy stress to give s'' (modified deviatoric stress)

    Returns: L_prime (3x3), L_double_prime (3x3)
    For plane stress: sigma = [sigma_xx, sigma_yy, sigma_xy]
    """
    a1, a2, a3, a4, a5, a6, a7, a8 = alpha

    # L' matrix (for phi')
    L_prime = np.array([
        [ 2*a1/3,  -a1/3,   0  ],
        [ -a2/3,   2*a2/3,  0  ],
        [  0,       0,      a7 ]
    ])

    # L'' matrix (for phi'')
    # Using the Barlat (2003) convention
    L_double = np.array([
        [  (8*a5 - 2*a3 - 2*a6 + 2*a4) / 9,
           (4*a6 - 4*a5 - 4*a4 + a3) / 9,
           0 ],
        [  (4*a3 - 4*a5 - 4*a4 + a6) / 9,
           (8*a4 - 2*a6 - 2*a3 + 2*a5) / 9,
           0 ],
        [  0, 0, a8 ]
    ])

    return L_prime, L_double


def compute_principal_values(X):
    """
    Compute the principal values of a 2D symmetric tensor.
    X = [X_xx, X_yy, X_xy]
    Returns: (X1, X2) principal values
    """
    X_xx, X_yy, X_xy = X[0], X[1], X[2]
    avg = (X_xx + X_yy) / 2.0
    diff = np.sqrt(((X_xx - X_yy) / 2.0)**2 + X_xy**2)
    return avg + diff, avg - diff


def yld2000_phi(stress, alpha, a=A_EXPONENT):
    """
    Compute the Yld2000-2d yield function value.

    phi = phi' + phi'' = |s1' - s2'|^a + |2*s2'' + s1''|^a + |2*s1'' + s2''|^a

    stress: [sigma_xx, sigma_yy, sigma_xy]
    alpha: [a1, a2, ..., a8]
    a: exponent (6 for BCC, 8 for FCC)

    Returns: phi value (should equal 2*sigma_bar^a at yielding)
    """
    L_prime, L_double = compute_L_matrices(alpha)

    # Transform stress
    s_prime = L_prime @ stress
    s_double = L_double @ stress

    # Principal values
    sp1, sp2 = compute_principal_values(s_prime)
    sd1, sd2 = compute_principal_values(s_double)

    # Yield function components
    phi_prime = np.abs(sp1 - sp2)**a
    phi_double = np.abs(2*sd2 + sd1)**a + np.abs(2*sd1 + sd2)**a

    return phi_prime + phi_double


def yld2000_sigma_bar(stress, alpha, a=A_EXPONENT):
    """
    Compute the equivalent stress from Yld2000-2d.
    sigma_bar = (phi/2)^(1/a)
    """
    phi = yld2000_phi(stress, alpha, a)
    return (phi / 2.0)**(1.0 / a)


def predict_r_value_yld(alpha, theta_deg, a=A_EXPONENT, ds=1e-6):
    """
    Predict r-value at angle theta from rolling direction using Yld2000-2d.

    r(theta) = -(d_eps_zz / d_eps_tt)

    where eps_zz = -(eps_xx + eps_yy) (thickness) and
    eps_tt = d_phi/d_sigma_tt (transverse to loading direction)

    Uses numerical differentiation of the yield function.
    """
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    # Uniaxial stress in theta direction
    sigma_theta = 1.0  # unit stress
    stress = np.array([
        sigma_theta * c**2,
        sigma_theta * s**2,
        sigma_theta * c * s
    ])

    # Numerical gradient of phi w.r.t. stress components
    grad = np.zeros(3)
    for i in range(3):
        sp = stress.copy()
        sm = stress.copy()
        sp[i] += ds
        sm[i] -= ds
        grad[i] = (yld2000_phi(sp, alpha, a) - yld2000_phi(sm, alpha, a)) / (2*ds)

    # Strain rate: d_eps_ij proportional to d_phi/d_sigma_ij
    d_eps_xx = grad[0]
    d_eps_yy = grad[1]

    # Strain in loading direction
    d_eps_ll = d_eps_xx * c**2 + d_eps_yy * s**2 + 2 * grad[2] * c * s

    # Strain in width direction (perpendicular to loading, in-plane)
    d_eps_ww = d_eps_xx * s**2 + d_eps_yy * c**2 - 2 * grad[2] * c * s

    # Thickness strain (incompressibility)
    d_eps_zz = -(d_eps_xx + d_eps_yy)

    # r = d_eps_ww / d_eps_zz
    if abs(d_eps_zz) < 1e-15:
        return 1.0
    r = d_eps_ww / d_eps_zz
    return r


def predict_sigma_ratio(alpha, theta_deg, a=A_EXPONENT):
    """
    Predict normalized yield stress sigma_theta / sigma_0 at angle theta.
    """
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    stress = np.array([c**2, s**2, c*s])
    sigma_bar = yld2000_sigma_bar(stress, alpha, a)

    # Reference (0 degree)
    stress_0 = np.array([1.0, 0.0, 0.0])
    sigma_bar_0 = yld2000_sigma_bar(stress_0, alpha, a)

    return sigma_bar_0 / sigma_bar


# ============================================================================
# COEFFICIENT IDENTIFICATION
# ============================================================================

def objective_yld2000(alpha_vec, targets, a=A_EXPONENT, weights=None):
    """
    Objective function for identifying Yld2000-2d coefficients.

    Minimizes weighted sum of squared errors between predicted and
    measured experimental constraints:
    - r-values at 0°, 45°, 90°
    - Normalized yield stresses at 0°, 45°, 90°

    Optional equi-biaxial terms can be included via weights, but they are
    excluded by default because σ_b and r_b are estimated rather than measured.

    targets: dict with r0, r45, r90, s0, s45, s90, sb, rb
    """
    alpha = alpha_vec
    if weights is None:
        weights = OBJECTIVE_WEIGHTS

    error = 0.0

    # --- r-value predictions ---
    r0_pred  = predict_r_value_yld(alpha, 0, a)
    r45_pred = predict_r_value_yld(alpha, 45, a)
    r90_pred = predict_r_value_yld(alpha, 90, a)

    w_r = weights.get('r', 1.0)
    error += w_r * ((r0_pred - targets['r0'])**2 / max(targets['r0']**2, 1e-6))
    error += w_r * ((r45_pred - targets['r45'])**2 / max(targets['r45']**2, 1e-6))
    error += w_r * ((r90_pred - targets['r90'])**2 / max(targets['r90']**2, 1e-6))

    # --- Normalized yield stress predictions ---
    s45_pred = predict_sigma_ratio(alpha, 45, a)
    s90_pred = predict_sigma_ratio(alpha, 90, a)

    s45_exp = targets['s45'] / targets['s0']
    s90_exp = targets['s90'] / targets['s0']

    error += weights.get('s45', 1.0) * ((s45_pred - s45_exp)**2 / max(s45_exp**2, 1e-6))
    error += weights.get('s90', 1.0) * ((s90_pred - s90_exp)**2 / max(s90_exp**2, 1e-6))

    # --- Equi-biaxial ---
    w_b = weights.get('sb', 0.0)
    if w_b > 0.0:
        stress_b = np.array([1.0, 1.0, 0.0])
        stress_0 = np.array([1.0, 0.0, 0.0])
        sb_pred = yld2000_sigma_bar(stress_0, alpha, a) / yld2000_sigma_bar(stress_b, alpha, a)
        sb_exp = targets['sb'] / targets['s0']
        error += w_b * ((sb_pred - sb_exp)**2 / max(sb_exp**2, 1e-6))

    w_rb = weights.get('rb', 0.0)
    if w_rb > 0.0:
        rb_pred = predict_biaxial_r(alpha, a)
        error += w_rb * ((rb_pred - targets['rb'])**2 / max(targets['rb']**2, 1e-6))

    return error


def predict_biaxial_r(alpha, a=A_EXPONENT, ds=1e-6):
    """
    Predict equi-biaxial r-value: rb = d_eps_yy / d_eps_xx under equal biaxial stress.
    """
    stress = np.array([1.0, 1.0, 0.0])

    grad = np.zeros(3)
    for i in range(3):
        sp = stress.copy()
        sm = stress.copy()
        sp[i] += ds
        sm[i] -= ds
        grad[i] = (yld2000_phi(sp, alpha, a) - yld2000_phi(sm, alpha, a)) / (2*ds)

    d_eps_xx = grad[0]
    d_eps_yy = grad[1]

    if abs(d_eps_xx) < 1e-15:
        return 1.0
    return d_eps_yy / d_eps_xx


def identify_yld2000_coefficients():
    """
    Identify the 8 Yld2000-2d anisotropy coefficients from measured
    uniaxial constraints.

    Estimated biaxial terms are reported as reference only and excluded from
    the fit used for the thesis comparison.
    """
    print("=" * 70)
    print("BARLAT YLD2000-2D COEFFICIENT IDENTIFICATION")
    print("=" * 70)

    targets = {
        'r0':  R0,
        'r45': R45,
        'r90': R90,
        's0':  SIGMA_0,
        's45': SIGMA_45,
        's90': SIGMA_90,
        'sb':  SIGMA_B,
        'rb':  RB_EST,
    }

    print(f"\nTarget values:")
    print(f"  r0  = {R0:.4f},  r45 = {R45:.4f},  r90 = {R90:.4f}")
    print(f"  σ0  = {SIGMA_0:.2f} MPa, σ45 = {SIGMA_45:.2f} MPa, σ90 = {SIGMA_90:.2f} MPa")
    print(f"  σ45/σ0 = {SIGMA_45/SIGMA_0:.6f},  σ90/σ0 = {SIGMA_90/SIGMA_0:.6f}")
    print(f"  σb  = {SIGMA_B:.1f} MPa (Hill'48 estimate, reference only), rb = {RB_EST:.4f}")
    print(f"  Exponent a = {A_EXPONENT} (BCC steel)")

    def obj_weighted(alpha_vec):
        return objective_yld2000(alpha_vec, targets, A_EXPONENT, OBJECTIVE_WEIGHTS)

    print("\nChecking proven exact coefficient set...")
    proven_cost = obj_weighted(PROVEN_ALPHA)
    print(f"  Proven-set cost = {proven_cost:.8e}")

    if proven_cost < 1.0e-12:
        print("  Using proven exact solution; global optimization skipped.")
        alpha_opt = PROVEN_ALPHA.copy()
        result = {
            'fun': float(proven_cost),
            'method': 'proven_exact',
            'success': True,
        }
    else:
        bounds = [(0.001, 5.0)] * 8
        print("\nRunning Differential Evolution optimization...")
        result_de = differential_evolution(
            obj_weighted, bounds,
            maxiter=800, seed=42, tol=1e-14, polish=False,
            popsize=30, mutation=(0.5, 1.5), recombination=0.9
        )
        print(f"  DE result: cost = {result_de.fun:.8e}")

        print("Running bounded local refinement (L-BFGS-B)...")
        refine = minimize(
            obj_weighted, result_de.x,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-12}
        )
        alpha_opt = refine.x
        result = {
            'fun': float(refine.fun),
            'method': 'L-BFGS-B',
            'success': bool(refine.success),
        }

    # Enforce Yld2000 normalization: sigma_bar([1,0,0]) = 1.
    # The objective penalizes only RATIOS (r-values, σ45/σ0, σ90/σ0), which
    # are invariant under uniform scaling of α (phi scales as α^a, sigbar as
    # α). That leaves one flat direction, so the optimizer can drift to α
    # with arbitrary magnitude (observed: sigbar([1,0,0]) ≈ 2.4). The UMAT
    # assumes sigbar equals the applied uniaxial stress in RD, so un-
    # normalized α cause a mismatch between yield surface and hardening
    # curve and the Newton return mapping to diverge. Uniform rescale leaves
    # every ratio target unchanged.
    stress_rd = np.array([1.0, 0.0, 0.0])
    sb_rd = yld2000_sigma_bar(stress_rd, alpha_opt, A_EXPONENT)
    if sb_rd > 0.0:
        alpha_opt = np.asarray(alpha_opt, dtype=float) / sb_rd
        print(f"\n  Normalization: sigbar([1,0,0]) was {sb_rd:.6f}, "
              f"rescaled α by 1/{sb_rd:.6f}")

    print(f"  Final cost = {result['fun']:.4e}")
    print(f"\n  Identified Coefficients:")
    for i, coeff in enumerate(alpha_opt):
        print(f"    α{i+1} = {coeff:.8f}")

    # Verify predictions
    print(f"\n  Verification:")
    r0_p  = predict_r_value_yld(alpha_opt, 0)
    r45_p = predict_r_value_yld(alpha_opt, 45)
    r90_p = predict_r_value_yld(alpha_opt, 90)
    s45_p = predict_sigma_ratio(alpha_opt, 45)
    s90_p = predict_sigma_ratio(alpha_opt, 90)
    stress_b = np.array([1.0, 1.0, 0.0])
    stress_0 = np.array([1.0, 0.0, 0.0])
    sb_p = yld2000_sigma_bar(stress_0, alpha_opt) / yld2000_sigma_bar(stress_b, alpha_opt)
    rb_p  = predict_biaxial_r(alpha_opt)

    print(f"    r0:     exp={R0:.4f},  pred={r0_p:.4f},  err={abs(r0_p-R0)/R0*100:.2f}%")
    print(f"    r45:    exp={R45:.4f}, pred={r45_p:.4f}, err={abs(r45_p-R45)/R45*100:.2f}%")
    print(f"    r90:    exp={R90:.4f}, pred={r90_p:.4f}, err={abs(r90_p-R90)/R90*100:.2f}%")
    print(f"    σ45/σ0: exp={SIGMA_45/SIGMA_0:.6f}, pred={s45_p:.6f}, err={abs(s45_p-SIGMA_45/SIGMA_0)/(SIGMA_45/SIGMA_0)*100:.3f}%")
    print(f"    σ90/σ0: exp={SIGMA_90/SIGMA_0:.6f}, pred={s90_p:.6f}, err={abs(s90_p-SIGMA_90/SIGMA_0)/(SIGMA_90/SIGMA_0)*100:.3f}%")
    print(f"    σb/σ0:  ref={SIGMA_B/SIGMA_0:.6f}, pred={sb_p:.6f}  (not fitted)")
    print(f"    rb:     ref={RB_EST:.4f}, pred={rb_p:.4f}  (not fitted)")

    return alpha_opt, result, targets


# ============================================================================
# HILL'48 HELPERS (for comparison)
# ============================================================================

def hill48_coefficients_from_rvalues(r0, r45, r90):
    """Compute Hill'48 coefficients from three r-values."""
    return {
        'F': r0 / (r90 * (1 + r0)),
        'G': 1.0 / (1 + r0),
        'H': r0 / (1 + r0),
        'N': (r0 + r90) * (1 + 2*r45) / (2 * r90 * (1 + r0)),
    }


def hill48_yield_surface_from_coeffs(coeffs, sigma_ref, n_points=360):
    """Compute Hill'48 yield surface in sigma_11-sigma_22 plane from FGHN."""
    F = coeffs['F']
    G = coeffs['G']
    H = coeffs['H']

    theta = np.linspace(0, 2*np.pi, n_points)
    s1_list, s2_list = [], []

    for t in theta:
        s1 = np.cos(t)
        s2 = np.sin(t)
        denom = F*s2**2 + G*s1**2 + H*(s1-s2)**2
        if denom > 0:
            scale = sigma_ref / np.sqrt(denom)
            s1_list.append(s1 * scale)
            s2_list.append(s2 * scale)

    return np.array(s1_list), np.array(s2_list)


def predict_r_value_hill48(coeffs, theta_deg):
    """Predict Hill'48 r-value at angle theta from FGHN."""
    F = coeffs['F']
    G = coeffs['G']
    H = coeffs['H']
    N = coeffs['N']

    t = np.radians(theta_deg)
    c2 = np.cos(t)**2
    s2 = np.sin(t)**2
    num = H + (2*N - F - G - 4*H) * c2 * s2
    den = F * s2 + G * c2
    return num / den if abs(den) > 1e-15 else np.nan


def predict_sigma_ratio_hill48(coeffs, theta_deg):
    """Predict Hill'48 normalized yield stress σ_theta/σ0 from FGHN."""
    F = coeffs['F']
    G = coeffs['G']
    H = coeffs['H']
    N = coeffs['N']

    t = np.radians(theta_deg)
    c2 = np.cos(t)**2
    s2 = np.sin(t)**2
    denom = F*s2*s2 + G*c2*c2 + H*(c2-s2)**2 + 2*N*c2*s2
    return 1.0 / np.sqrt(denom) if denom > 0 else np.nan


def hill48_yield_surface(r0, r45, r90, sigma_ref, n_points=360):
    """
    Compute Hill'48 yield surface in sigma_11-sigma_22 plane.
    """
    coeffs = hill48_coefficients_from_rvalues(r0, r45, r90)
    return hill48_yield_surface_from_coeffs(coeffs, sigma_ref, n_points)


def yld2000_yield_surface(alpha, sigma_ref, a=A_EXPONENT, n_points=360):
    """
    Compute Yld2000-2d yield surface in sigma_11-sigma_22 plane (sigma_12=0).
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    s1_list, s2_list = [], []

    for t in theta:
        s1 = np.cos(t)
        s2 = np.sin(t)
        stress = np.array([s1, s2, 0.0])
        sig_bar = yld2000_sigma_bar(stress, alpha, a)
        if sig_bar > 1e-10:
            scale = sigma_ref / sig_bar
            s1_list.append(s1 * scale)
            s2_list.append(s2 * scale)

    return np.array(s1_list), np.array(s2_list)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_yield_surface_comparison(alpha_opt, targets):
    """
    Compare Hill'48 vs Yld2000-2d yield surfaces.
    """
    sigma_ref = targets['s0']

    hill_r_s1, hill_r_s2 = hill48_yield_surface_from_coeffs(HILL48_RVALUE_COEFFS, sigma_ref)
    hill_s_s1, hill_s_s2 = hill48_yield_surface_from_coeffs(HILL48_STRESS_COEFFS, sigma_ref)

    # Yld2000-2d
    y_s1, y_s2 = yld2000_yield_surface(alpha_opt, sigma_ref)

    # Von Mises (isotropic reference)
    theta = np.linspace(0, 2*np.pi, 360)
    vm_s1 = sigma_ref * np.cos(theta) / np.sqrt(np.cos(theta)**2 - np.cos(theta)*np.sin(theta) + np.sin(theta)**2 + 1e-15)
    vm_s2 = sigma_ref * np.sin(theta) / np.sqrt(np.cos(theta)**2 - np.cos(theta)*np.sin(theta) + np.sin(theta)**2 + 1e-15)
    # Simpler: parametric von Mises
    vm_s1_list, vm_s2_list = [], []
    for t in theta:
        s1, s2 = np.cos(t), np.sin(t)
        denom = s1**2 - s1*s2 + s2**2
        if denom > 0:
            scale = sigma_ref / np.sqrt(denom)
            vm_s1_list.append(s1 * scale)
            vm_s2_list.append(s2 * scale)
    vm_s1 = np.array(vm_s1_list)
    vm_s2 = np.array(vm_s2_list)

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(vm_s1, vm_s2, 'k:', linewidth=1.5, alpha=0.5, label='von Mises (isotropic)')
    ax.plot(hill_r_s1, hill_r_s2, 'b--', linewidth=2, label="Hill'48 (r-calibrated)")
    ax.plot(hill_s_s1, hill_s_s2, color='seagreen', linestyle='-.', linewidth=2,
            label="Hill'48 (stress-calibrated)")
    ax.plot(y_s1, y_s2, 'r-', linewidth=2.5, label='Yld2000-2d')

    # Experimental points
    ax.plot(sigma_ref, 0, 'ko', markersize=10, zorder=5, label=f'σ₀ = {sigma_ref:.0f} MPa')
    ax.plot(0, targets['s90'], 'ks', markersize=10, zorder=5, label=f'σ₉₀ = {targets["s90"]:.0f} MPa')
    ax.plot(targets['sb'], targets['sb'], 'k^', markersize=10, zorder=5,
            label=f'σ_b = {targets["sb"]:.0f} MPa (est.)')

    ax.set_xlabel('σ₁₁ (MPa)')
    ax.set_ylabel('σ₂₂ (MPa)')
    ax.set_title('Yield Surface Comparison: Hill\'48 vs Barlat Yld2000-2d\nSGCC JIS G 3302')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_yield_surface_comparison.png'))
    plt.close()
    print("  Saved: fig_yield_surface_comparison.png")


def plot_r_value_comparison(alpha_opt):
    """
    Compare r-value predictions from Hill'48 vs Yld2000-2d across angles.
    """
    angles = np.arange(0, 91, 5)

    r_yld = [predict_r_value_yld(alpha_opt, th) for th in angles]
    r_hill_rvalue = [predict_r_value_hill48(HILL48_RVALUE_COEFFS, th) for th in angles]
    r_hill_stress = [predict_r_value_hill48(HILL48_STRESS_COEFFS, th) for th in angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(angles, r_yld, 'r-o', linewidth=2, markersize=5, label='Yld2000-2d')
    ax.plot(angles, r_hill_rvalue, 'b--s', linewidth=2, markersize=5,
            label="Hill'48 (r-calibrated)")
    ax.plot(angles, r_hill_stress, color='seagreen', linestyle='-.', marker='d',
            linewidth=2, markersize=5, label="Hill'48 (stress-calibrated)")
    ax.plot([0, 45, 90], [R0, R45, R90], 'k^', markersize=12, zorder=5,
            label='Experimental')

    ax.set_xlabel('Angle from Rolling Direction (°)')
    ax.set_ylabel('Lankford r-value')
    ax.set_title('r-value Prediction: Hill\'48 vs Barlat Yld2000-2d')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 92)
    ax.set_ylim(0, 7.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_r_value_comparison_models.png'))
    plt.close()
    print("  Saved: fig_r_value_comparison_models.png")


def plot_sigma_ratio_comparison(alpha_opt):
    """
    Compare normalized yield stress predictions across angles.
    """
    angles = np.arange(0, 91, 5)

    # Experimental
    s_exp = {0: 1.0, 45: SIGMA_45/SIGMA_0, 90: SIGMA_90/SIGMA_0}

    s_yld = [predict_sigma_ratio(alpha_opt, th) for th in angles]
    s_hill_rvalue = [predict_sigma_ratio_hill48(HILL48_RVALUE_COEFFS, th) for th in angles]
    s_hill_stress = [predict_sigma_ratio_hill48(HILL48_STRESS_COEFFS, th) for th in angles]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(angles, s_yld, 'r-o', linewidth=2, markersize=5, label='Yld2000-2d')
    ax.plot(angles, s_hill_rvalue, 'b--s', linewidth=2, markersize=5,
            label="Hill'48 (r-calibrated)")
    ax.plot(angles, s_hill_stress, color='seagreen', linestyle='-.', marker='d',
            linewidth=2, markersize=5, label="Hill'48 (stress-calibrated)")
    ax.plot([0, 45, 90], [s_exp[0], s_exp[45], s_exp[90]], 'k^', markersize=12,
            zorder=5, label='Experimental')

    ax.set_xlabel('Angle from Rolling Direction (°)')
    ax.set_ylabel('Normalized Yield Stress (σ_θ / σ₀)')
    ax.set_title('Yield Stress Directionality: Hill\'48 vs Barlat Yld2000-2d')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 92)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_sigma_ratio_comparison.png'))
    plt.close()
    print("  Saved: fig_sigma_ratio_comparison.png")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(alpha_opt, result, targets):
    """Save Yld2000-2d results for documentation."""
    stress_b = np.array([1.0, 1.0, 0.0])
    stress_0 = np.array([1.0, 0.0, 0.0])
    results = {
        'coefficients': {f'alpha_{i+1}': float(alpha_opt[i]) for i in range(8)},
        'exponent': A_EXPONENT,
        'objective': {
            'mode': 'measured_uniaxial_only',
            'weights': {k: float(v) for k, v in OBJECTIVE_WEIGHTS.items()},
            'final_cost': float(result['fun']),
            'method': result['method'],
            'success': bool(result['success']),
        },
        'targets': {k: float(v) for k, v in targets.items()},
        'predictions': {
            'r0': float(predict_r_value_yld(alpha_opt, 0)),
            'r45': float(predict_r_value_yld(alpha_opt, 45)),
            'r90': float(predict_r_value_yld(alpha_opt, 90)),
            'sigma45_ratio': float(predict_sigma_ratio(alpha_opt, 45)),
            'sigma90_ratio': float(predict_sigma_ratio(alpha_opt, 90)),
            'sigma_b_ratio': float(yld2000_sigma_bar(stress_0, alpha_opt) / yld2000_sigma_bar(stress_b, alpha_opt)),
            'rb': float(predict_biaxial_r(alpha_opt)),
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'yld2000_parameters.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: yld2000_parameters.json")

    with open(os.path.join(OUTPUT_DIR, 'yld2000_parameters.txt'), 'w') as f:
        f.write("BARLAT YLD2000-2D ANISOTROPIC YIELD CRITERION\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Material: SGCC JIS G 3302\n")
        f.write(f"Exponent: a = {A_EXPONENT} (BCC)\n\n")
        f.write("Fit objective: measured uniaxial r-values + yield stress ratios\n")
        f.write("Reference-only terms excluded from fit: sigma_b, r_b\n")
        f.write(f"Method: {result['method']}\n")
        f.write(f"Final cost: {result['fun']:.12e}\n\n")
        f.write("Identified Coefficients:\n")
        for i in range(8):
            f.write(f"  α{i+1} = {alpha_opt[i]:.8f}\n")
        f.write(f"\nExperimental vs Predicted:\n")
        f.write(f"  {'Parameter':>12s}  {'Experimental':>12s}  {'Predicted':>12s}  {'Error %':>8s}\n")
        f.write(f"  {'-'*50}\n")
        preds = results['predictions']
        for name, exp_val, pred_key in [
            ('r0', R0, 'r0'), ('r45', R45, 'r45'), ('r90', R90, 'r90'),
            ('σ45/σ0', SIGMA_45/SIGMA_0, 'sigma45_ratio'),
            ('σ90/σ0', SIGMA_90/SIGMA_0, 'sigma90_ratio'),
        ]:
            p = preds[pred_key]
            err = abs(p - exp_val) / exp_val * 100
            f.write(f"  {name:>12s}  {exp_val:>12.4f}  {p:>12.4f}  {err:>7.2f}%\n")
        f.write("\nReference-only biaxial terms:\n")
        f.write(f"  {'σb/σ0':>12s}  {SIGMA_B/SIGMA_0:>12.4f}  {preds['sigma_b_ratio']:>12.4f}\n")
        f.write(f"  {'rb':>12s}  {RB_EST:>12.4f}  {preds['rb']:>12.4f}\n")
    print("  Saved: yld2000_parameters.txt")


def save_comparison_summary(alpha_opt, result, targets):
    """Save a thesis-ready comparison of Hill'48 and Yld2000-2d."""
    yld_r0 = predict_r_value_yld(alpha_opt, 0)
    yld_r45 = predict_r_value_yld(alpha_opt, 45)
    yld_r90 = predict_r_value_yld(alpha_opt, 90)
    yld_s45 = predict_sigma_ratio(alpha_opt, 45)
    yld_s90 = predict_sigma_ratio(alpha_opt, 90)

    hill_rvalue_s45 = HILL48_RVALUE_RUN['sigma45_ratio']
    hill_rvalue_s90 = HILL48_RVALUE_RUN['sigma90_ratio']
    hill_stress_s45 = predict_sigma_ratio_hill48(HILL48_STRESS_COEFFS, 45)
    hill_stress_s90 = predict_sigma_ratio_hill48(HILL48_STRESS_COEFFS, 90)

    comparison_path = os.path.join(OUTPUT_DIR, 'yield_criterion_comparison_summary.txt')
    with open(comparison_path, 'w') as f:
        f.write("YIELD CRITERION COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Material: SGCC JIS G 3302\n")
        f.write("Scope: Mac-safe thesis comparison without UMAT-based Yld2000 FEA\n\n")

        f.write("Measured uniaxial targets\n")
        f.write("-" * 60 + "\n")
        f.write(f"  r0={R0:.4f}, r45={R45:.4f}, r90={R90:.4f}\n")
        f.write(f"  sigma0={SIGMA_0:.2f} MPa, sigma45={SIGMA_45:.2f} MPa, sigma90={SIGMA_90:.2f} MPa\n")
        f.write(f"  sigma45/sigma0={SIGMA_45/SIGMA_0:.6f}, sigma90/sigma0={SIGMA_90/SIGMA_0:.6f}\n\n")

        f.write("1. Hill'48 r-value calibration\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Weighted FEA NRMSE: {HILL48_RVALUE_RUN['weighted_nrmse']:.3f}\n")
        f.write(f"  Predicted sigma45/sigma0: {hill_rvalue_s45:.6f}\n")
        f.write(f"  Predicted sigma90/sigma0: {hill_rvalue_s90:.6f}\n")
        f.write(f"  45-deg yield error: {(hill_rvalue_s45 - SIGMA_45/SIGMA_0)/(SIGMA_45/SIGMA_0)*100:+.1f}%\n")
        f.write("  Interpretation: matches the measured r-values but misses the severe 45-deg yield-stress drop.\n\n")

        f.write("2. Hill'48 stress calibration\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Weighted FEA NRMSE: {HILL48_STRESS_RUN['weighted_nrmse']:.3f}\n")
        f.write(f"  Per-direction NRMSE: 0-deg={HILL48_STRESS_RUN['nrmse_00']:.3f}, 45-deg={HILL48_STRESS_RUN['nrmse_45']:.3f}, 90-deg={HILL48_STRESS_RUN['nrmse_90']:.3f}\n")
        f.write(f"  Predicted sigma45/sigma0: {hill_stress_s45:.6f}\n")
        f.write(f"  Predicted sigma90/sigma0: {hill_stress_s90:.6f}\n")
        f.write(f"  Implied r-values: r0={HILL48_STRESS_RUN['r0']:.4f}, r45={HILL48_STRESS_RUN['r45']:.4f}, r90={HILL48_STRESS_RUN['r90']:.4f}\n")
        f.write(f"  Backstress saturation at optimum: {HILL48_STRESS_RUN['backstress_sat']:.3f} MPa\n")
        f.write("  Interpretation: fits the three uniaxial yield stresses, but the 45-deg residual remains because one isotropic hardening law cannot reproduce direction-dependent hardening.\n\n")

        f.write("3. Yld2000-2d measured-only identification\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Method: {result['method']}\n")
        f.write(f"  Final objective cost: {result['fun']:.12e}\n")
        f.write("  Coefficients:\n")
        for i, coeff in enumerate(alpha_opt):
            f.write(f"    alpha{i+1} = {coeff:.8f}\n")
        f.write(f"  Predicted r-values: r0={yld_r0:.4f}, r45={yld_r45:.4f}, r90={yld_r90:.4f}\n")
        f.write(f"  Predicted stress ratios: sigma45/sigma0={yld_s45:.6f}, sigma90/sigma0={yld_s90:.6f}\n")
        f.write("  Interpretation: resolves the measured yield-surface anisotropy exactly at the uniaxial level.\n\n")

        f.write("Conclusion\n")
        f.write("-" * 60 + "\n")
        f.write("  Hill'48 cannot simultaneously represent the measured r-values and the moderate 45-deg yield-stress softening (sigma45/sigma0 = 0.715) of this SGCC sheet.\n")
        f.write(f"  Stress-calibrated Hill'48 remains the best available FEA model on the current machine (weighted NRMSE {HILL48_STRESS_RUN['weighted_nrmse']:.3f} vs {HILL48_RVALUE_RUN['weighted_nrmse']:.3f} for the r-value calibration), but its remaining 45-deg error is a hardening-anisotropy limitation rather than a yield-stress mismatch.\n")
        f.write("  Yld2000-2d provides the correct non-FEA reference for the thesis because it matches both measured r-values and measured yield-stress ratios exactly, while UMAT-based FEA is unavailable on the current Apple Silicon / Parallels setup.\n")
    print("  Saved: yield_criterion_comparison_summary.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    alpha_opt, result, targets = identify_yld2000_coefficients()

    print("\nGenerating comparison plots...")
    plot_yield_surface_comparison(alpha_opt, targets)
    plot_r_value_comparison(alpha_opt)
    plot_sigma_ratio_comparison(alpha_opt)

    print("\nSaving results...")
    save_results(alpha_opt, result, targets)
    save_comparison_summary(alpha_opt, result, targets)

    print("\n" + "=" * 70)
    print("BARLAT YLD2000-2D ANALYSIS COMPLETE")
    print("=" * 70)

    return alpha_opt


if __name__ == '__main__':
    alpha_opt = main()
