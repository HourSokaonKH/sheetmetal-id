# -*- coding: utf-8 -*-
"""
=============================================================================
Multi-Direction Inverse Parameter Identification — FEA-Based
Combined Isotropic-Kinematic Hardening, 2 Backstresses
Nelder-Mead Simplex (built-in, no scipy)
=============================================================================
Compatible with: Abaqus 2024 Python (Windows)

Usage:
    abaqus python optimize_hardening_multidir.py

This script extends optimize_hardening_standalone.py to use THREE tensile
directions (0°, 45°, 90°) simultaneously. For each parameter evaluation:
  1. Generates 3 .inp files with different *Orientation angles
  2. Submits 3 Abaqus jobs (sequential)
  3. Extracts global RF2/U2 from loaded boundary (mimics UTM measurement)
  4. Computes weighted NRMSE across all 3 directions
  5. Uses Nelder-Mead to optimize [sigma0, C1, gamma1, C2, gamma2]

Extraction method: Global force-displacement (RF2 summed at loaded nodes,
U2 at corner node) converted to true stress-strain. This directly
corresponds to the UTM measurement, unlike single-element extraction
which is sensitive to mesh artifacts and hourglass modes (CPS4R).

Key novelty: The material orientation in the .inp file is rotated to match
each specimen cutting angle. Hill'48 R-values remain fixed (measured from
DIC), so the anisotropic yield surface is identical — only the loading
direction relative to rolling direction changes.

Parameters to optimize: [sigma0, C1, gamma1, C2, gamma2]
Fixed: Q_inf=335.16, b=3.95 (from Voce fit of 0° data)
       Hill'48 R-values from multi-zone DIC

Author: HOUR Sokaon
Date:   2026
=============================================================================
"""

import numpy as np
import os
import sys
import csv
import subprocess
import time
import math
import shutil
import platform

# Try odbAccess — available under 'abaqus python'
try:
    from odbAccess import openOdb
    HAS_ODB = True
except ImportError:
    HAS_ODB = False
    print("WARNING: odbAccess not available. Will use CSV fallback.")

# ============================================================================
# CONFIGURATION
# ============================================================================

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

# Directions and experimental data
# This Abaqus-side weighting intentionally emphasizes 45° because that
# direction is the most discriminating for anisotropy in the validated
# three-direction FEA comparison. It is not the same weighted study used in
# multi_objective_optimization.py, where the analytical surrogate prioritizes
# the 0° rolling direction.
DIRECTIONS = {
    0: {
        'angle_deg': 0,
        'files': ['stress-00-01.csv', 'stress-00-02.csv', 'stress-00-03.csv'],
        'weight': 1.0,
    },
    45: {
        'angle_deg': 45,
        'files': ['stress-45-01.csv', 'stress-45-02.csv', 'stress-45-03.csv'],
        'weight': 2.0,  # 2× weight: 45° is most sensitive to anisotropy
    },
    90: {
        'angle_deg': 90,
        'files': ['stress-90-01.csv', 'stress-90-02.csv', 'stress-90-03.csv'],
        'weight': 1.0,
    },
}

# Geometry (quarter model)
HALF_WIDTH = 10.0      # mm (full gauge = 20 mm)
HALF_LENGTH = 40.0     # mm (full gauge = 80 mm)
THICKNESS = 1.5        # mm
MESH_SIZE = 1.0        # mm

# Fixed material
E_YOUNG = 200000.0     # MPa
NU = 0.3
DENSITY = 7.85e-9      # tonne/mm^3

# ─── Yield criterion mode ────────────────────────────────────────────────────
# MODEL_TYPE = 'hill48'  : Abaqus built-in Hill'48 (*Plastic + *Potential)
#                          Sub-mode controlled by HILL48_MODE ('stress'/'r_value')
# MODEL_TYPE = 'yld2000' : Barlat Yld2000-2d via UMAT (umat_yld2000.f)
#                          Simultaneously matches r-values AND yield stresses.
#                          Requires Intel Fortran on Windows for compilation.
#                          Run: abaqus job=... user=umat_yld2000.f interactive
# ─────────────────────────────────────────────────────────────────────────────
MODEL_TYPE = 'yld2000'   # 'hill48' or 'yld2000'

# ─── Hill'48 calibration mode ────────────────────────────────────────────────
# Only used when MODEL_TYPE = 'hill48'
# HILL48_MODE = 'r_value': Matches canonical r0/r45/r90 from MATLAB multi-zone DIC
# HILL48_MODE = 'stress' : Matches the pooled 0.2% offset yield stresses
# ─────────────────────────────────────────────────────────────────────────────
HILL48_MODE = 'stress'

# R-value calibration (canonical r0=0.7122, r45=0.7998, r90=0.7420)
# F=0.5606, G=0.5840, H=0.4160, N=1.4877
# Abaqus ratios: R11=1, R22=1.0119, R33=0.9347, R12=1.0041
_R_RVALUE = (1.0, 1.0119, 0.9347, 1.0041, 1.0, 1.0)

# Stress calibration: pooled 0.2% offset yield stresses (mean of 3 specimens)
# sigma0 = 352.53 MPa, sigma45 = 252.17 MPa, sigma90 = 381.47 MPa
# sigma45/sigma0 = 0.7154, sigma90/sigma0 = 1.0821
# r0 preserved = 0.7122; implied r45 = 2.826 (Hill'48 structural limit for this
# material — unusual but finite, versus the 6.41 obtained when Voce-extrapolated
# flow stresses were used as targets).
# F=0.4380, G=0.5840, H=0.4160, N=3.3990
# Abaqus ratios: R11=1, R22=1.0821, R33=0.9891, R12=0.6643
_R_STRESS  = (1.0, 1.0821, 0.9891, 0.6643, 1.0, 1.0)

R11, R22, R33, R12, R13, R23 = _R_STRESS if HILL48_MODE == 'stress' else _R_RVALUE

# ─── Yld2000-2d coefficients ─────────────────────────────────────────────────
# Identified from: r0=0.712, r45=0.800, r90=0.742 (DIC multi-zone)
#             and: sigma45/sigma0 = 0.715251, sigma90/sigma0 = 1.082025
#                  (pooled 0.2% offset yield stresses: 352.55 / 252.16 / 381.47 MPa).
# Re-identified by barlat_yld2000.py (cost 3.06e-20, machine precision).
# UMAT: umat_yld2000.f  NPROPS=13  NSTATV=1
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: The hardcoded list below is a placeholder.  At import time we try to
# load the authoritative coefficients from output/yld2000_parameters.json
# (the same file run_yld2000_umat.py reads).  This prevents the optimizer
# and the standalone runner from using different material cards — a bug
# that made the optimizer report NRMSE ~10x larger than preflight for
# identical parameters because the yield-surface scale differed.
YLD2000_ALPHA = [
    2.78216474,   # α1  (fallback; overridden from JSON below if available)
    1.99615102,   # α2
    3.86202580,   # α3
    2.58000953,   # α4
    2.87101512,   # α5
    4.79371206,   # α6
    1.18982067,   # α7
    4.25577549,   # α8
]

try:
    import json as _json
    _alpha_json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'output', 'yld2000_parameters.json',
    )
    if os.path.exists(_alpha_json_path):
        with open(_alpha_json_path, 'r') as _fh:
            _alpha_data = _json.load(_fh)
        _coeffs = _alpha_data.get('coefficients') or _alpha_data
        YLD2000_ALPHA = [
            float(_coeffs['alpha_1']),
            float(_coeffs['alpha_2']),
            float(_coeffs['alpha_3']),
            float(_coeffs['alpha_4']),
            float(_coeffs['alpha_5']),
            float(_coeffs['alpha_6']),
            float(_coeffs['alpha_7']),
            float(_coeffs['alpha_8']),
        ]
        print("  Loaded Yld2000 alpha from %s" % _alpha_json_path)
except Exception as _e:
    print("  Warning: could not load Yld2000 alpha from JSON (%s); "
          "using hardcoded fallback." % _e)

# Fixed isotropic hardening (Voce from 0° data) - loaded from material_constants.json
from material_constants import Q_INF, B_ISO

# Loading
# DISPLACEMENT must stay below the onset of geometric instability (necking).
# Experimental data extends to ~20% true strain (~22% engineering strain on
# a 40 mm gauge half-length = 8.8 mm displacement).  Driving past UTS makes
# the FE response unstable: 0 deg often tolerates it, but 45 deg develops
# a localization band that cascades into cutback exhaustion.  NRMSE is
# computed on the overlap region only, so a shorter run loses no fidelity.
DISPLACEMENT = 12.0    # mm  (~30% eng strain on 40 mm half-length)

# Optimization
MAX_ITER = 150
FTOL = 1e-5
JOB_TIMEOUT = 300
JOB_POLL_INTERVAL = 2.0

# Initial guess [sigma0, C1, gamma1, C2, gamma2]
# sigma0 ~ 326 MPa (0-deg Voce initial yield) for both Hill48-stress and Yld2000
X0 = [326.0, 800.0, 400.0, 200.0, 150.0]

# Bounds
BOUNDS_LOW  = [200.0, 100.0,  10.0,  10.0,   5.0]
BOUNDS_HIGH = [400.0, 5000.0, 2000.0, 2000.0, 500.0]

# Job naming — prefix encodes model and calibration to avoid overwriting runs
if MODEL_TYPE == 'yld2000':
    JOB_PREFIX = 'mopt_y'
elif MODEL_TYPE == 'yld2000_table':
    JOB_PREFIX = 'mopt_yt'
elif HILL48_MODE == 'stress':
    JOB_PREFIX = 'mopt_s'
else:
    JOB_PREFIX = 'mopt_r'

# ============================================================================
# GLOBAL TRACKING
# ============================================================================
ITER_COUNT = [0]
BEST_COST = [1e20]
BEST_PARAMS = [None]
HISTORY = []


# ============================================================================
# EXPERIMENTAL DATA
# ============================================================================

def load_experimental_data():
    """Load and average experimental true stress-strain for all 3 directions."""
    exp_data = {}

    for angle, cfg in DIRECTIONS.items():
        all_curves = []
        for fname in cfg['files']:
            filepath = os.path.join(WORK_DIR, fname)
            if not os.path.exists(filepath):
                print("  WARNING: %s not found, skipping" % filepath)
                continue

            stress_list = []
            strain_list = []
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 4:
                        try:
                            s = float(row[2])
                            e = float(row[3])
                            stress_list.append(s)
                            strain_list.append(e)
                        except ValueError:
                            continue

            eng_stress = np.array(stress_list)
            eng_strain = np.array(strain_list)

            # Truncate at UTS
            uts_idx = np.argmax(eng_stress)
            eng_stress = eng_stress[:uts_idx + 1]
            eng_strain = eng_strain[:uts_idx + 1]

            mask = (eng_strain > 0) & (eng_stress > 0)
            true_strain = np.log(1.0 + eng_strain[mask])
            true_stress = eng_stress[mask] * (1.0 + eng_strain[mask])
            all_curves.append((true_strain, true_stress))

        if not all_curves:
            raise RuntimeError("No data for %d-degree direction!" % angle)

        # Interpolate to common grid and average
        strain_min = max(c[0].min() for c in all_curves)
        strain_max = min(c[0].max() for c in all_curves)
        common_strain = np.linspace(strain_min, strain_max, 200)

        stress_interp = []
        for ts, ss in all_curves:
            stress_interp.append(np.interp(common_strain, ts, ss))

        mean_stress = np.mean(stress_interp, axis=0)
        exp_data[angle] = (common_strain, mean_stress)

        print("  %2d-deg: %d specimens, strain [%.4f, %.4f], "
              "stress [%.1f, %.1f] MPa" % (
                  angle, len(all_curves),
                  common_strain.min(), common_strain.max(),
                  mean_stress.min(), mean_stress.max()))

    return exp_data


# ============================================================================
# INP FILE GENERATION (with rotated material orientation)
# ============================================================================

def generate_inp(job_name, sigma0, C1, gamma1, C2, gamma2, angle_deg):
    """
    Generate Abaqus .inp for monotonic tensile test at a given angle.

    The material *Orientation is rotated so that the rolling direction (RD)
    makes angle_deg with the tensile axis (Y-axis of the model).

    For a specimen cut at angle theta from RD:
      - RD direction in model coords: a = (sin(theta), cos(theta), 0)
      - TD direction in model coords: b = (-cos(theta), sin(theta), 0)

    This means Hill'48 anisotropy is correctly applied relative to RD
    regardless of the specimen cutting angle.
    """
    filepath = job_name + '.inp'
    theta = math.radians(angle_deg)

    # Orientation vectors: RD in model coordinate system
    # Model Y = tensile axis, Model X = width
    # For 0° specimen: RD = Y → a = (0, 1, 0)
    # For 45° specimen: RD at 45° to tensile → a = (sin45, cos45, 0)
    # For 90° specimen: RD perpendicular to tensile → a = (1, 0, 0)
    ax = math.sin(theta)
    ay = math.cos(theta)
    bx = -math.cos(theta)
    by = math.sin(theta)

    # Mesh
    nx = int(HALF_WIDTH / MESH_SIZE)
    ny = int(HALF_LENGTH / MESH_SIZE)

    # Nodes
    nodes = []
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append((nid, i * MESH_SIZE, j * MESH_SIZE))
            nid += 1

    # Elements (CPS4R)
    elements = []
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i + 1
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            elements.append((eid, n1, n2, n3, n4))
            eid += 1

    # Node sets
    sym_x = [n[0] for n in nodes if abs(n[1]) < 1e-10]
    sym_y = [n[0] for n in nodes if abs(n[2]) < 1e-10]
    load_n = [n[0] for n in nodes if abs(n[2] - HALF_LENGTH) < 1e-10]
    corner = [n[0] for n in nodes if abs(n[1]) < 1e-10
              and abs(n[2] - HALF_LENGTH) < 1e-10]

    # Hardening table: monotonic equivalent
    n_pts = 50
    eps_p = np.linspace(0, 0.5, n_pts)
    sigma = sigma0 + Q_INF * (1.0 - np.exp(-B_ISO * eps_p))
    sigma += (C1 / gamma1) * (1.0 - np.exp(-gamma1 * eps_p))
    sigma += (C2 / gamma2) * (1.0 - np.exp(-gamma2 * eps_p))
    hardening = list(zip(sigma, eps_p))

    def write_nset(f, nids):
        for i, n in enumerate(nids):
            if i > 0 and i % 16 == 0:
                f.write('\n')
            elif i > 0:
                f.write(', ')
            f.write('%d' % n)
        f.write('\n')

    with open(filepath, 'w') as f:
        f.write('*Heading\n')
        f.write('** Multi-direction optimization: %s (angle=%d deg)\n' % (
            job_name, angle_deg))
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n')

        # Part
        f.write('*Part, name=SPECIMEN\n')
        f.write('*Node\n')
        for nid, x, y in nodes:
            f.write('%d, %.6f, %.6f\n' % (nid, x, y))
        # Elements: CPS4R for Hill'48 (matches thesis mesh-convergence
        # study); CPS4 (full integration) for Yld2000 UMAT. Abaqus
        # refuses CPS4R + UMAT without an explicit *Hourglass Stiffness
        # because it cannot infer a default shear modulus from a user
        # subroutine ("400 elements have been defined with zero hour
        # glass stiffness"). Full integration sidesteps the issue with a
        # 4× cost at 400 elements, which is negligible here. The UMAT
        # itself returns the plane-stress elastic tangent as DDSDDE so
        # the assembled K is well conditioned.
        if MODEL_TYPE == 'yld2000':
            f.write('*Element, type=CPS4\n')
        elif MODEL_TYPE == 'yld2000_table':
            f.write('*Element, type=CPS4\n')
        else:
            f.write('*Element, type=CPS4R\n')
        for eid, n1, n2, n3, n4 in elements:
            f.write('%d, %d, %d, %d, %d\n' % (eid, n1, n2, n3, n4))
        f.write('*Elset, elset=ALL, generate\n')
        f.write('1, %d, 1\n' % len(elements))

        f.write('*Nset, nset=SYM_X\n')
        write_nset(f, sym_x)
        f.write('*Nset, nset=SYM_Y\n')
        write_nset(f, sym_y)
        f.write('*Nset, nset=LOAD\n')
        write_nset(f, load_n)
        if corner:
            f.write('*Nset, nset=CORNER\n')
            f.write('%d\n' % corner[0])

        # Orientation: rotate RD into the model frame
        f.write('*Orientation, name=ORI_ROLL, system=RECTANGULAR\n')
        f.write('%.6f, %.6f, 0.0, %.6f, %.6f, 0.0\n' % (ax, ay, bx, by))
        f.write('1, 0.0\n')

        # Section with orientation. For CPS4R + UMAT Abaqus falls back to
        # a default hourglass stiffness (0.5 % G) because the UMAT does
        # not supply one. This is adequate here since the UMAT returns
        # the elastic tangent as DDSDDE (positive definite, no near-zero
        # eigenvalues to interact with hourglass modes).
        f.write('*Solid Section, elset=ALL, material=MAT, '
                'orientation=ORI_ROLL\n')
        f.write('%.1f,\n' % THICKNESS)
        f.write('*End Part\n')
        f.write('**\n')

        # Assembly
        f.write('*Assembly, name=Assembly\n')
        f.write('*Instance, name=SPEC-1, part=SPECIMEN\n')
        f.write('*End Instance\n')
        f.write('*Nset, nset=SYM_X, instance=SPEC-1\n')
        write_nset(f, sym_x)
        f.write('*Nset, nset=SYM_Y, instance=SPEC-1\n')
        write_nset(f, sym_y)
        f.write('*Nset, nset=LOAD, instance=SPEC-1\n')
        write_nset(f, load_n)
        if corner:
            f.write('*Nset, nset=CORNER, instance=SPEC-1\n')
            f.write('%d\n' % corner[0])
        f.write('*End Assembly\n')
        f.write('**\n')

        # Material — Hill'48 or Yld2000-2d
        f.write('*Material, name=MAT\n')
        f.write('*Density\n%.2e,\n' % DENSITY)
        if MODEL_TYPE == 'yld2000':
            # Yld2000-2d UMAT: *User Material + *Depvar only.
            # *Elastic must NOT appear — elasticity is handled inside the UMAT.
            # *Depvar must come BEFORE *User Material per Abaqus convention.
            # PROPS(1..13): E, NU, sigma0, Q_INF, B_ISO, alpha1..alpha8
            props = [E_YOUNG, NU, sigma0, Q_INF, B_ISO] + list(YLD2000_ALPHA)
            f.write('*Depvar\n1,\n')   # NSTATV=1 (kappa) — before *User Material
            f.write('*User Material, constants=13\n')
            line_vals = []
            for v in props:
                line_vals.append('%.8g' % v)
                if len(line_vals) == 8:
                    f.write(', '.join(line_vals) + '\n')
                    line_vals = []
            if line_vals:
                f.write(', '.join(line_vals) + '\n')
        elif MODEL_TYPE == 'yld2000_table':
            # Yld2000-2d UMAT with tabulated flow curve (umat_yld2000_table.f).
            # Bake (sigma0, C1, gamma1, C2, gamma2) together with the fixed
            # Voce (Q_INF, B_ISO) into a monotonic-equivalent flow-stress
            # table so all five parameters genuinely reach Abaqus.
            # PROPS layout: [E, NU, alpha1..alpha8, NTAB, kappa*NTAB, sigy*NTAB]
            from hardening_table import build_flow_curve, build_umat_props
            kappa_tab, sigy_tab = build_flow_curve(
                sigma_0=sigma0,
                Q_inf=Q_INF, b=B_ISO,
                C1=C1, gamma1=gamma1, C2=C2, gamma2=gamma2,
                n_points=50, eps_max=0.40,
            )
            props = build_umat_props(
                E=E_YOUNG, nu=NU, alpha=list(YLD2000_ALPHA),
                kappa=kappa_tab, sigma_y=sigy_tab,
            )
            f.write('*Depvar\n1,\n')
            f.write('*User Material, constants=%d\n' % len(props))
            line_vals = []
            for v in props:
                line_vals.append('%.8g' % v)
                if len(line_vals) == 8:
                    f.write(', '.join(line_vals) + '\n')
                    line_vals = []
            if line_vals:
                f.write(', '.join(line_vals) + '\n')
        else:
            # Hill'48: *Elastic + tabulated *Plastic + *Potential R-values.
            # *Elastic must precede *Plastic; omitting it causes Abaqus to
            # reject the material at the Input File Processor stage.
            f.write('*Elastic\n')
            f.write('%.1f, %.3f\n' % (E_YOUNG, NU))
            f.write('*Plastic\n')
            for s, ep in hardening:
                f.write('%.4f, %.6f\n' % (s, ep))
            f.write('*Potential\n')
            f.write('%.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n' % (
                R11, R22, R33, R12, R13, R23))
        f.write('**\n')

        # Step
        # The Yld2000-2d UMAT uses a 4×4 Newton return mapping with a
        # finite-difference Hessian. A large elastic-predictor overshoot on
        # the first plastic increment (default 0.01 → ε ≈ 0.6% → σ_trial ≈
        # 4× yield) causes Newton to diverge and PNEWDT cutbacks to exhaust.
        # A much smaller initial increment keeps the predictor close to
        # yield and lets the return mapping converge reliably. Hill'48 uses
        # Abaqus' built-in closed-form return and does not need this.
        if MODEL_TYPE == 'yld2000' or MODEL_TYPE == 'yld2000_table':
            f.write('*Step, name=Tensile, nlgeom=YES, inc=10000\n')
            f.write('*Static\n')
            f.write('1e-4, 1.0, 1e-10, 5e-3\n')
        else:
            f.write('*Step, name=Tensile, nlgeom=YES, inc=1000\n')
            f.write('*Static\n')
            f.write('0.01, 1.0, 1e-08, 0.02\n')
        f.write('**\n')

        # BCs
        f.write('*Boundary\nSYM_X, 1, 1, 0.0\n')
        f.write('*Boundary\nSYM_Y, 2, 2, 0.0\n')
        f.write('*Boundary\nLOAD, 2, 2, %.1f\n' % DISPLACEMENT)
        f.write('**\n')

        # Output
        f.write('*Output, field, frequency=1\n')
        f.write('*Node Output\nU, RF\n')
        f.write('*Element Output, directions=YES\nS, LE, PE, PEEQ\n')
        f.write('*Output, history, frequency=1\n')
        f.write('*Node Output, nset=CORNER\nU2, RF2\n')
        f.write('*End Step\n')

    return filepath


# ============================================================================
# JOB SUBMISSION AND EXTRACTION
# ============================================================================

def run_abaqus_job(job_name, timeout=300):
    """Submit Abaqus job, poll real job files, and report failures."""

    def read_text(path):
        try:
            with open(path, 'rb') as f:
                data = f.read()
            return data.decode('utf-8', 'ignore')
        except Exception:
            return ''

    def tail_text(path, max_lines=20):
        text = read_text(path)
        if not text:
            return ''
        lines = text.splitlines()
        return '\n'.join(lines[-max_lines:])

    def job_has_activity():
        for ext in ['.run', '.log', '.sta', '.msg', '.dat', '.lck', '.odb']:
            if os.path.exists(job_name + ext):
                return True
        return False

    def job_completed():
        sta_text = read_text(job_name + '.sta').upper()
        msg_text = read_text(job_name + '.msg').upper()
        if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta_text:
            return True
        if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in msg_text:
            return True
        # Do NOT treat the mere presence of an .odb as success: a failed
        # Input-File Processor run or a diverged analysis can leave a
        # partial/empty .odb with no .lck, and relying on it would make
        # run_abaqus_job return True for a broken job.
        return False

    def job_failed():
        keywords = [
            'EXITED WITH ERRORS',
            'HAS NOT BEEN COMPLETED',
            'HAS BEEN TERMINATED',
            'PROBLEM DURING COMPILATION',
            'NOT RECOGNIZED AS AN INTERNAL OR EXTERNAL COMMAND',
            'COMMAND NOT FOUND',
            'LINK : FATAL ERROR',
            'LNK',
            '***ERROR',
            'ABAQUS/MAKE',
        ]
        for ext in ['.run', '.log', '.msg', '.dat', '.sta']:
            text = read_text(job_name + ext).upper()
            if not text:
                continue
            for key in keywords:
                if key in text:
                    return True
        return False

    def summarize_failure():
        summary = []
        for ext in ['.run', '.log', '.msg', '.dat', '.sta']:
            path = job_name + ext
            text = read_text(path)
            if not text:
                continue
            hits = []
            for line in text.splitlines():
                upper = line.upper()
                if ('ERROR' in upper or
                        'EXITED WITH ERRORS' in upper or
                        'HAS NOT BEEN COMPLETED' in upper or
                        'HAS BEEN TERMINATED' in upper or
                        'PROBLEM DURING COMPILATION' in upper or
                        'NOT RECOGNIZED AS AN INTERNAL OR EXTERNAL COMMAND' in upper or
                        'COMMAND NOT FOUND' in upper or
                        'LNK' in upper or
                        'USER SUBROUTINE' in upper):
                    hits.append(line.strip())
            if hits:
                summary.append('  %s:' % os.path.basename(path))
                for line in hits[-8:]:
                    summary.append('    %s' % line)
        if not summary:
            for ext in ['.run', '.log', '.msg', '.dat', '.sta']:
                path = job_name + ext
                tail = tail_text(path)
                if tail:
                    summary.append('  %s (tail):' % os.path.basename(path))
                    for line in tail.splitlines()[-10:]:
                        summary.append('    %s' % line)
                    break
        if not summary:
            summary.append('  No Abaqus diagnostic text was found.')
        return '\n'.join(summary)

    def save_failure_artifacts():
        fail_dir = os.path.join(WORK_DIR, 'abaqus_failures', job_name)
        if not os.path.exists(fail_dir):
            os.makedirs(fail_dir)
        for ext in ['.inp', '.run', '.log', '.msg', '.dat', '.sta', '.com',
                    '.prt', '.sim', '.res', '.mdl', '.stt', '.odb']:
            src = job_name + ext
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(fail_dir,
                                                   os.path.basename(src)))
                except Exception:
                    pass
        return fail_dir

    if MODEL_TYPE == 'yld2000':
        cmd_core = 'abaqus job=%s user=umat_yld2000.f interactive ask_delete=OFF' % job_name
    elif MODEL_TYPE == 'yld2000_table':
        cmd_core = 'abaqus job=%s user=umat_yld2000_table.f interactive ask_delete=OFF' % job_name
    else:
        cmd_core = 'abaqus job=%s interactive ask_delete=OFF' % job_name
    cmd = '%s > %s.run 2>&1' % (cmd_core, job_name)

    try:
        proc = subprocess.Popen(cmd, shell=True)
    except Exception as e:
        print("  Error launching Abaqus: %s" % str(e))
        return False

    t0 = time.time()
    saw_activity = False

    while time.time() - t0 < timeout:
        if job_has_activity():
            saw_activity = True

        if job_completed():
            return True

        if saw_activity and job_failed() and not os.path.exists(job_name + '.lck'):
            fail_dir = save_failure_artifacts()
            print("  Abaqus job failed. Diagnostics:")
            print(summarize_failure())
            print("  Failure files saved to: %s" % fail_dir)
            return False

        if proc.poll() is not None and not os.path.exists(job_name + '.lck'):
            if job_completed():
                return True
            if saw_activity and job_failed():
                fail_dir = save_failure_artifacts()
                print("  Abaqus job failed. Diagnostics:")
                print(summarize_failure())
                print("  Failure files saved to: %s" % fail_dir)
                return False
            if not saw_activity and time.time() - t0 > 15.0:
                break

        time.sleep(JOB_POLL_INTERVAL)

    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass

    fail_dir = save_failure_artifacts()
    print("  Abaqus job did not complete successfully within %d seconds." % timeout)
    print(summarize_failure())
    print("  Failure files saved to: %s" % fail_dir)
    return False


def extract_results_odb(job_name):
    """
    Extract global force-displacement from ODB and convert to true
    stress-strain.

    Method: Sum RF2 at all loaded nodes (LOAD nset) for quarter-model
    reaction force. Get U2 from corner node for displacement. Convert:
      eng_stress = (2 * RF2_quarter) / A0   [x2: width symmetry only]
      eng_strain = U2 / L0
      true_stress = eng_stress * (1 + eng_strain)
      true_strain = ln(1 + eng_strain)

    This directly mimics the UTM measurement (force / cross-section),
    making the comparison with experimental data physically consistent.
    """
    odb_path = job_name + '.odb'
    if not os.path.exists(odb_path):
        return None

    try:
        odb = openOdb(path=odb_path, readOnly=True)
    except Exception as e:
        print("  Cannot open ODB: %s" % str(e))
        return None

    step = odb.steps['Tensile']
    asm = odb.rootAssembly

    # Identify loaded nodes (top edge: y = HALF_LENGTH)
    inst = asm.instances['SPEC-1']
    load_labels = set()
    for node in inst.nodes:
        if abs(node.coordinates[1] - HALF_LENGTH) < 1e-6:
            load_labels.add(node.label)

    # Find the corner node (x=0, y=HALF_LENGTH) for displacement
    corner_label = None
    for node in inst.nodes:
        if (abs(node.coordinates[0]) < 1e-6 and
                abs(node.coordinates[1] - HALF_LENGTH) < 1e-6):
            corner_label = node.label
            break

    if not load_labels or corner_label is None:
        odb.close()
        return None

    # Cross-section area (full specimen)
    A0 = 2.0 * HALF_WIDTH * THICKNESS  # mm^2 (full width × thickness)
    L0 = HALF_LENGTH  # mm (gauge half-length)

    true_strain_list = []
    true_stress_list = []

    for frame in step.frames:
        rf_field = frame.fieldOutputs['RF']
        u_field = frame.fieldOutputs['U']

        # Sum RF2 over all loaded nodes (quarter-model reaction)
        rf2_quarter = 0.0
        for val in rf_field.values:
            if val.nodeLabel in load_labels:
                rf2_quarter += val.data[1]  # RF2 component

        # Get U2 at corner node
        u2 = None
        for val in u_field.values:
            if val.nodeLabel == corner_label:
                u2 = val.data[1]  # U2 component
                break

        if u2 is None:
            continue

        # Only the width is halved by symmetry; tensile force doubles, not quadruples.
        rf2_full = 2.0 * rf2_quarter
        eng_stress = rf2_full / A0
        eng_strain = u2 / L0

        # Skip zero/negative strain points
        if eng_strain <= 0 or eng_stress <= 0:
            continue

        # Convert to true stress-strain
        true_strain = math.log(1.0 + eng_strain)
        true_stress = eng_stress * (1.0 + eng_strain)

        true_strain_list.append(true_strain)
        true_stress_list.append(true_stress)

    odb.close()

    if len(true_strain_list) < 5:
        return None

    return np.array(true_strain_list), np.array(true_stress_list)


# ============================================================================
# NELDER-MEAD IMPLEMENTATION
# ============================================================================

def nelder_mead(func, x0, args=(), maxiter=200, ftol=1e-5,
                initial_step=None):
    """
    Nelder-Mead simplex optimization (no scipy required).

    Parameters:
        func: objective function f(x, *args) -> scalar
        x0: initial guess (1D array)
        args: extra arguments passed to func
        maxiter: max iterations
        ftol: convergence tolerance on function value spread
        initial_step: step sizes for initial simplex
    Returns:
        dict with 'x', 'fun', 'nit', 'nfev'
    """
    n = len(x0)
    x0 = np.array(x0, dtype=float)

    # Standard coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    # Build initial simplex
    if initial_step is None:
        initial_step = np.where(np.abs(x0) > 1e-10, 0.15 * np.abs(x0), 1.0)
    elif np.isscalar(initial_step):
        initial_step = np.full(n, initial_step)
    else:
        initial_step = np.array(initial_step, dtype=float)

    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += initial_step[i]

    # Evaluate all vertices
    f_vals = np.array([func(simplex[i], *args) for i in range(n + 1)])
    nfev = n + 1

    for iteration in range(maxiter):
        # Sort
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        # Check convergence
        f_spread = np.abs(f_vals[-1] - f_vals[0])
        if f_spread < ftol:
            print("\n  Converged at iteration %d (spread=%.2e)" % (
                iteration, f_spread))
            break

        # Centroid of all except worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        f_worst = f_vals[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        f_r = func(x_r, *args)
        nfev += 1

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = func(x_e, *args)
            nfev += 1
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        # Contraction
        if f_r < f_worst:
            x_c = centroid + rho * (x_r - centroid)
            f_c = func(x_c, *args)
            nfev += 1
            if f_c <= f_r:
                simplex[-1] = x_c
                f_vals[-1] = f_c
                continue
        else:
            x_c = centroid - rho * (centroid - worst)
            f_c = func(x_c, *args)
            nfev += 1
            if f_c < f_worst:
                simplex[-1] = x_c
                f_vals[-1] = f_c
                continue

        # Shrink
        best = simplex[0].copy()
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            f_vals[i] = func(simplex[i], *args)
            nfev += 1

    # Final sort
    order = np.argsort(f_vals)
    best_x = simplex[order[0]]
    best_f = f_vals[order[0]]

    return {
        'x': best_x,
        'fun': best_f,
        'nit': min(iteration + 1, maxiter),
        'nfev': nfev,
    }


# ============================================================================
# OBJECTIVE FUNCTION (multi-direction)
# ============================================================================

def objective(params, exp_data):
    """
    Run THREE Abaqus simulations (0°, 45°, 90°) and compute
    weighted NRMSE across all directions.
    """
    sigma0, C1, gamma1, C2, gamma2 = params
    ITER_COUNT[0] += 1
    it = ITER_COUNT[0]

    # Bounds penalty
    for val, lo, hi in zip(params, BOUNDS_LOW, BOUNDS_HIGH):
        if val < lo or val > hi:
            cost = 10.0
            print("  Iter %3d: OUT OF BOUNDS -> cost=%.4f" % (it, cost))
            HISTORY.append((it, list(params), cost, 'bounds',
                            {0: None, 45: None, 90: None}))
            return cost

    # Positivity
    if sigma0 <= 0 or C1 <= 0 or gamma1 <= 0 or C2 <= 0 or gamma2 <= 0:
        print("  Iter %3d: NEGATIVE PARAMS" % it)
        HISTORY.append((it, list(params), 10.0, 'negative',
                        {0: None, 45: None, 90: None}))
        return 10.0

    print("\n  Iter %3d: s0=%.1f C1=%.1f g1=%.1f C2=%.1f g2=%.1f" % (
        it, sigma0, C1, gamma1, C2, gamma2))

    # Run all 3 directions
    nrmse_per_dir = {}
    total_weight = 0.0
    weighted_nrmse = 0.0
    any_failed = False

    for angle, cfg in DIRECTIONS.items():
        job_name = '%s_%04d_%02d' % (JOB_PREFIX, it, angle)
        exp_strain, exp_stress = exp_data[angle]
        weight = cfg['weight']

        # Generate .inp with rotated orientation
        generate_inp(job_name, sigma0, C1, gamma1, C2, gamma2, angle)

        # Run simulation
        success = run_abaqus_job(job_name)
        if not success:
            print("           %2d-deg FAILED" % angle)
            nrmse_per_dir[angle] = None
            any_failed = True
            cleanup_job(job_name, keep_odb=False)
            continue

        # Extract results
        result = extract_results_odb(job_name)
        if result is None:
            print("           %2d-deg no data" % angle)
            nrmse_per_dir[angle] = None
            any_failed = True
            cleanup_job(job_name, keep_odb=False)
            continue

        sim_strain, sim_stress = result

        # Compute NRMSE on overlapping range
        strain_min = max(sim_strain.min(), exp_strain.min())
        strain_max = min(sim_strain.max(), exp_strain.max())
        if strain_max <= strain_min + 0.01:
            print("           %2d-deg no overlap" % angle)
            nrmse_per_dir[angle] = None
            any_failed = True
            cleanup_job(job_name, keep_odb=False)
            continue

        common = np.linspace(strain_min, strain_max, 150)
        sim_interp = np.interp(common, sim_strain, sim_stress)
        exp_interp = np.interp(common, exp_strain, exp_stress)

        rmse = np.sqrt(np.mean((sim_interp - exp_interp) ** 2))
        stress_range = exp_interp.max() - exp_interp.min()
        nrmse = rmse / stress_range if stress_range > 0 else 10.0

        nrmse_per_dir[angle] = nrmse
        weighted_nrmse += weight * nrmse
        total_weight += weight

        print("           %2d-deg NRMSE=%.5f (w=%.1f)" % (
            angle, nrmse, weight))

        # Cleanup (only keep best ODB)
        cleanup_job(job_name, keep_odb=False)

    # If any direction failed, penalize
    if any_failed:
        cost = 5.0
        print("           COMBINED: PARTIAL FAILURE -> cost=%.4f" % cost)
        HISTORY.append((it, list(params), cost, 'partial_fail',
                        nrmse_per_dir))
        return cost

    # Weighted average NRMSE
    cost = weighted_nrmse / total_weight if total_weight > 0 else 10.0

    print("           COMBINED NRMSE=%.5f (weighted)" % cost)

    if cost < BEST_COST[0]:
        BEST_COST[0] = cost
        BEST_PARAMS[0] = list(params)
        print("           *** NEW BEST ***")
        # Re-run and keep best ODBs
        for angle in DIRECTIONS:
            job_best = '%s_best_%02d' % (JOB_PREFIX, angle)
            generate_inp(job_best, sigma0, C1, gamma1, C2, gamma2, angle)
            run_abaqus_job(job_best)
            # Keep this ODB

    HISTORY.append((it, list(params), cost, 'ok', nrmse_per_dir))

    return cost


def cleanup_job(job_name, keep_odb=False):
    """Remove intermediate Abaqus files to save disk space."""
    exts_remove = ['.dat', '.msg', '.sta', '.com', '.prt', '.sim',
                   '.log', '.inp', '.mdl', '.stt', '.res', '.abq',
                   '.pac', '.sel', '.run', '.lck']
    if not keep_odb:
        exts_remove.append('.odb')

    for ext in exts_remove:
        fpath = job_name + ext
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass


def check_yld2000_compiler_environment():
    """Validate that Intel Fortran is available before launching Abaqus."""
    if MODEL_TYPE not in ('yld2000', 'yld2000_table'):
        return True

    arch_tokens = [
        platform.machine(),
        os.environ.get('PROCESSOR_ARCHITECTURE', ''),
        os.environ.get('PROCESSOR_ARCHITEW6432', ''),
    ]
    arch_info = ' | '.join([token for token in arch_tokens if token])
    if sys.platform.startswith('win') and 'ARM' in arch_info.upper():
        print("\nERROR: Detected Windows ARM environment: %s" % arch_info)
        print("  Abaqus 2024 Windows requirements specify a 64-bit Windows")
        print("  x86_64 platform. Windows on ARM under Parallels on Apple Silicon")
        print("  is outside the supported configuration for Abaqus user")
        print("  subroutine compilation/linking.")
        print("")
        print("  Practical implication:")
        print("  - Built-in Abaqus material models may still run.")
        print("  - UMAT/VUMAT workflows should be moved to a native x86_64")
        print("    Windows or Linux machine.")
        return False

    ifort_path = shutil.which('ifort')
    if ifort_path:
        print("Compiler check: ifort found at %s" % ifort_path)
        return True

    ifx_path = shutil.which('ifx')
    if ifx_path:
        print("Compiler check: ifx found at %s" % ifx_path)
        print("  (classic ifort was retired in oneAPI 2025+; Abaqus 2024")
        print("   win86_64.env has been patched to use ifx.)")
        return True

    print("\nERROR: Intel Fortran compiler (ifort or ifx) is not available in PATH.")
    print("  Abaqus/Standard cannot compile umat_yld2000.f without the Intel")
    print("  compiler environment active in the SAME shell that launches Abaqus.")
    print("")
    print("  Fix on Windows:")
    print("  1. Open 'Intel oneAPI command prompt for Intel 64' (or cmd.exe).")
    print("  2. Activate: call \"C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat\" intel64 vs2022 --force")
    print("  3. cd to the working directory.")
    print("  4. Run: where ifx   (should print a path)")
    print("  5. Run: abaqus python optimize_hardening_multidir.py")
    return False


def run_preflight_check():
    """Run one Abaqus job before optimization to catch setup issues early."""
    if MODEL_TYPE not in ('yld2000', 'yld2000_table'):
        return True

    umat_name = ('umat_yld2000_table.f' if MODEL_TYPE == 'yld2000_table'
                 else 'umat_yld2000.f')
    umat_path = os.path.join(WORK_DIR, umat_name)
    if not os.path.exists(umat_path):
        print("\nERROR: UMAT file not found: %s" % umat_path)
        return False

    if not check_yld2000_compiler_environment():
        return False

    print("Preflight: single 0-deg Yld2000 job...")
    job_name = '%s_preflight_00' % JOB_PREFIX
    generate_inp(job_name, X0[0], X0[1], X0[2], X0[3], X0[4], 0)
    success = run_abaqus_job(job_name, timeout=JOB_TIMEOUT)
    if not success:
        print("  Preflight failed. Optimization aborted before simplex iterations.")
        print("  Inspect abaqus_failures/%s for the full .inp/.log/.msg/.dat files." % job_name)
        cleanup_job(job_name, keep_odb=False)
        return False

    cleanup_job(job_name, keep_odb=False)
    print("  Preflight passed.")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MULTI-DIRECTION INVERSE PARAMETER IDENTIFICATION")
    print("Combined Isotropic-Kinematic Hardening (2 Backstresses)")
    print("Directions: 0deg, 45deg, 90deg (weighted NRMSE)")
    print("Method: Nelder-Mead Simplex (built-in, no scipy)")
    print("=" * 70)

    # Load experimental data for all directions
    print("\nLoading experimental data...")
    exp_data = load_experimental_data()
    print("  Loaded %d directions" % len(exp_data))

    # Direction weights
    for angle, cfg in DIRECTIONS.items():
        print("  %2d-deg: weight=%.1f" % (angle, cfg['weight']))
    total_w = sum(c['weight'] for c in DIRECTIONS.values())
    print("  Total weight: %.1f (normalized)" % total_w)

    # Initial guess
    x0 = np.array(X0, dtype=float)
    print("\nInitial guess:")
    print("  sigma0=%.1f  C1=%.1f  gamma1=%.1f  C2=%.1f  gamma2=%.1f" % (
        tuple(x0)))
    print("Fixed: Q_inf=%.2f, b=%.2f" % (Q_INF, B_ISO))
    if MODEL_TYPE == 'yld2000':
        print("Yield criterion: Yld2000-2d UMAT  (a=6, alpha=[%.4f,%.4f,...,%.4f])" % (
            YLD2000_ALPHA[0], YLD2000_ALPHA[1], YLD2000_ALPHA[-1]))
        print("  r0=0.712, r45=0.800, r90=0.742  sigma45/sigma0=0.685  (exact fit)")
        print("  Run command: abaqus job=<name> user=umat_yld2000.f interactive")
    else:
        print("Hill48 calibration: %s  R11=%.4f R22=%.4f R33=%.4f R12=%.4f" % (
            HILL48_MODE, R11, R22, R33, R12))
    print("\nEach iteration runs 3 Abaqus jobs (~90s total per eval)")
    print("Starting optimization (max %d iterations)...\n" % MAX_ITER)

    if not run_preflight_check():
        return

    # Time tracking
    t0 = time.time()

    # Run optimization
    result = nelder_mead(
        objective, x0,
        args=(exp_data,),
        maxiter=MAX_ITER,
        ftol=FTOL,
    )

    elapsed = time.time() - t0

    # Report
    opt = result['x']
    print("\n" + "=" * 70)
    print("MULTI-DIRECTION OPTIMIZATION RESULTS")
    print("=" * 70)
    print("  sigma0 = %.4f MPa" % opt[0])
    print("  C1     = %.4f" % opt[1])
    print("  gamma1 = %.4f" % opt[2])
    print("  C2     = %.4f" % opt[3])
    print("  gamma2 = %.4f" % opt[4])
    print("  Weighted NRMSE = %.6f" % result['fun'])
    print("  Iterations: %d" % result['nit'])
    print("  Function evals: %d (×3 directions = %d Abaqus jobs)" % (
        result['nfev'], result['nfev'] * 3))
    print("  Wall time: %.1f min" % (elapsed / 60.0))

    # Per-direction NRMSE at optimum
    print("\nPer-direction NRMSE at optimum:")
    best_hist = [h for h in HISTORY if h[3] == 'ok']
    if best_hist:
        last_ok = min(best_hist, key=lambda h: h[2])
        for angle in [0, 45, 90]:
            val = last_ok[4].get(angle, None)
            if val is not None:
                print("  %2d-deg: NRMSE = %.6f" % (angle, val))
    else:
        print("  No successful Abaqus evaluation completed.")
        print("  Inspect: %s" % os.path.join(WORK_DIR, 'abaqus_failures'))

    # Save results
    out_dir = os.path.join(WORK_DIR, 'optimization_results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_tag = MODEL_TYPE if MODEL_TYPE == 'yld2000' else HILL48_MODE
    with open(os.path.join(out_dir,
                           'optimized_parameters_multidir_%s.txt' % out_tag), 'w') as f:
        f.write("MULTI-DIRECTION OPTIMIZED COMBINED HARDENING PARAMETERS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Yield criterion: %s\n" % MODEL_TYPE)
        if MODEL_TYPE == 'hill48':
            f.write("Hill48 calibration: %s\n" % HILL48_MODE)
        else:
            f.write("Yld2000-2d alpha: %s\n" % YLD2000_ALPHA)
        f.write("Method: Nelder-Mead (built-in, FEA-based)\n")
        f.write("Directions: 0-deg (w=1), 45-deg (w=2), 90-deg (w=1)\n")
        f.write("Total weight: %.1f\n\n" % total_w)
        f.write("Optimized:\n")
        f.write("  sigma0 = %.6f MPa\n" % opt[0])
        f.write("  C1     = %.6f\n" % opt[1])
        f.write("  gamma1 = %.6f\n" % opt[2])
        f.write("  C2     = %.6f\n" % opt[3])
        f.write("  gamma2 = %.6f\n" % opt[4])
        f.write("\nFixed:\n")
        f.write("  Q_inf  = %.6f MPa\n" % Q_INF)
        f.write("  b      = %.6f\n" % B_ISO)
        if MODEL_TYPE == 'hill48':
            f.write("\nHill48: R11=%.4f R22=%.4f R33=%.4f R12=%.4f\n" % (
                R11, R22, R33, R12))
        else:
            f.write("\nYld2000-2d alpha: %s\n" % YLD2000_ALPHA)
        f.write("\nFit:\n")
        f.write("  Weighted NRMSE = %.8f\n" % result['fun'])
        f.write("  Iterations = %d\n" % result['nit'])
        f.write("  Evaluations = %d\n" % result['nfev'])
        f.write("  Abaqus jobs = %d\n" % (result['nfev'] * 3))
        f.write("  Wall time = %.1f min\n" % (elapsed / 60.0))

        if best_hist:
            f.write("\nPer-direction NRMSE:\n")
            for angle in [0, 45, 90]:
                val = last_ok[4].get(angle, None)
                if val is not None:
                    f.write("  %2d-deg: %.8f\n" % (angle, val))

    with open(os.path.join(out_dir,
                           'convergence_history_multidir_%s.csv' % out_tag), 'w') as f:
        f.write('Iteration,sigma0,C1,gamma1,C2,gamma2,'
                'NRMSE_weighted,NRMSE_00,NRMSE_45,NRMSE_90,Status\n')
        for it, p, cost, status, per_dir in HISTORY:
            n00 = per_dir.get(0, None)
            n45 = per_dir.get(45, None)
            n90 = per_dir.get(90, None)
            f.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%s,%s,%s,%s\n' % (
                it, p[0], p[1], p[2], p[3], p[4], cost,
                '%.8f' % n00 if n00 is not None else '',
                '%.8f' % n45 if n45 is not None else '',
                '%.8f' % n90 if n90 is not None else '',
                status))

    print("\nResults saved to: %s" % out_dir)
    print("Output tag: '%s'" % out_tag)
    if best_hist:
        print("Best ODBs: %s_best_00.odb, %s_best_45.odb, %s_best_90.odb" % (
            JOB_PREFIX, JOB_PREFIX, JOB_PREFIX))
    else:
        print("Failure diagnostics: %s" % os.path.join(WORK_DIR, 'abaqus_failures'))


if __name__ == '__main__':
    main()
