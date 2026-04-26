# -*- coding: utf-8 -*-
"""
=============================================================================
Inverse Parameter Identification — Standalone (no Abaqus CAE required)
Combined Isotropic-Kinematic Hardening, 2 Backstresses
Nelder-Mead Simplex (implemented from scratch — no scipy needed)
=============================================================================
Compatible with: Abaqus 2024 Python (Windows)

Usage:
    abaqus python optimize_hardening.py

This script:
  1. Generates .inp files directly (no CAE)
  2. Submits jobs via subprocess
  3. Extracts results via odbAccess
  4. Optimizes with a built-in Nelder-Mead implementation

For monotonic tensile loading, combined hardening is equivalent to:
  sigma(ep) = sigma0 + Q_inf*(1-exp(-b*ep)) + sum(Ck/gk*(1-exp(-gk*ep)))
so we generate a single isotropic hardening table + Hill'48 *Potential.

Parameters to optimize: [sigma0, C1, gamma1, C2, gamma2]
Fixed: Q_inf=335.16, b=3.95 (from Voce fit of 0-deg data)

Author: PhD Candidate
Date:   2026
=============================================================================
"""

import numpy as np
import os
import sys
import csv
import subprocess
import time
import shutil

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

# Experimental data (0-degree, all 3 specimens averaged)
EXP_FILES = [
    os.path.join(WORK_DIR, 'stress-00-01.csv'),
    os.path.join(WORK_DIR, 'stress-00-02.csv'),
    os.path.join(WORK_DIR, 'stress-00-03.csv'),
]

# Geometry (quarter model)
HALF_WIDTH = 10.0      # mm (full gauge = 20 mm)
HALF_LENGTH = 40.0     # mm (full gauge = 80 mm)
THICKNESS = 1.5        # mm
MESH_SIZE = 1.0        # mm

# Fixed material
E_YOUNG = 200000.0     # MPa
NU = 0.3
DENSITY = 7.85e-9      # tonne/mm^3

# Hill'48 R-values (MATLAB-validated)
R11, R22, R33, R12, R13, R23 = 1.0, 1.0119, 0.9347, 1.0041, 1.0, 1.0

# Fixed isotropic hardening (Voce from 0-deg)
Q_INF = 335.16         # MPa
B_ISO = 3.95

# Loading
DISPLACEMENT = 25.0    # mm

# Optimization
MAX_ITER = 150
FTOL = 1e-5

# Initial guess [sigma0, C1, gamma1, C2, gamma2]
X0 = [300.0, 800.0, 400.0, 200.0, 150.0]

# Bounds
BOUNDS_LOW  = [200.0, 100.0,  10.0,  10.0,   5.0]
BOUNDS_HIGH = [400.0, 5000.0, 2000.0, 2000.0, 500.0]

# Job naming
JOB_PREFIX = 'opt'

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
    """Load and average experimental true stress-strain from all 0-deg specimens."""
    all_true_stress = []
    all_true_strain = []

    for filepath in EXP_FILES:
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

        all_true_stress.append((true_strain, true_stress))

    if not all_true_stress:
        raise RuntimeError("No experimental data loaded!")

    # Interpolate all to common strain grid
    strain_min = max(ts[0].min() for ts in all_true_stress)
    strain_max = min(ts[0].max() for ts in all_true_stress)
    common_strain = np.linspace(strain_min, strain_max, 200)

    stress_interp = []
    for ts, ss in all_true_stress:
        interp = np.interp(common_strain, ts, ss)
        stress_interp.append(interp)

    mean_stress = np.mean(stress_interp, axis=0)

    return common_strain, mean_stress


# ============================================================================
# INP FILE GENERATION
# ============================================================================

def generate_inp(job_name, sigma0, C1, gamma1, C2, gamma2):
    """
    Generate Abaqus .inp for monotonic tensile test.
    Combined hardening is baked into a single isotropic table for monotonic loading.
    """
    filepath = job_name + '.inp'

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
    corner = [n[0] for n in nodes if abs(n[1]) < 1e-10 and abs(n[2] - HALF_LENGTH) < 1e-10]

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
        f.write('** Optimization iteration: %s\n' % job_name)
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n')

        # Part
        f.write('*Part, name=SPECIMEN\n')
        f.write('*Node\n')
        for nid, x, y in nodes:
            f.write('%d, %.6f, %.6f\n' % (nid, x, y))
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

        # Orientation + section
        f.write('*Orientation, name=ORI, system=RECTANGULAR\n')
        f.write('1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n')
        f.write('1, 0.0\n')
        f.write('*Solid Section, elset=ALL, material=MAT, orientation=ORI\n')
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

        # Material
        f.write('*Material, name=MAT\n')
        f.write('*Density\n%.2e,\n' % DENSITY)
        f.write('*Elastic\n%.1f, %.1f\n' % (E_YOUNG, NU))
        f.write('*Plastic\n')
        for s, ep in hardening:
            f.write('%.4f, %.6f\n' % (s, ep))
        f.write('*Potential\n')
        f.write('%.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n' % (R11, R22, R33, R12, R13, R23))
        f.write('**\n')

        # Step
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
    """Submit Abaqus job and wait for completion."""
    cmd = 'abaqus job=%s interactive' % job_name
    try:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        if rc != 0:
            print("  Abaqus returned code %d" % rc)
            return False

        # Check for .odb
        if os.path.exists(job_name + '.odb'):
            return True

        # Check .sta for completion
        sta_file = job_name + '.sta'
        if os.path.exists(sta_file):
            with open(sta_file, 'r') as f:
                content = f.read()
            if 'COMPLETED' in content.upper():
                return True

        return False

    except subprocess.TimeoutExpired:
        proc.kill()
        print("  Job timed out after %d seconds" % timeout)
        return False
    except Exception as e:
        print("  Error running job: %s" % str(e))
        return False


def extract_results_odb(job_name):
    """
    Extract global force-displacement from ODB and convert to true
    stress-strain.

    Method: Sum RF2 at all loaded nodes for quarter-model reaction force.
    Get U2 from corner node for displacement. Convert to true stress-strain.
    This directly mimics the UTM measurement (force / cross-section).
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
    inst = odb.rootAssembly.instances['SPEC-1']

    # Identify loaded nodes (top edge: y = HALF_LENGTH)
    load_labels = set()
    for node in inst.nodes:
        if abs(node.coordinates[1] - HALF_LENGTH) < 1e-6:
            load_labels.add(node.label)

    # Find corner node (x=0, y=HALF_LENGTH) for displacement
    corner_label = None
    for node in inst.nodes:
        if (abs(node.coordinates[0]) < 1e-6 and
                abs(node.coordinates[1] - HALF_LENGTH) < 1e-6):
            corner_label = node.label
            break

    if not load_labels or corner_label is None:
        odb.close()
        return None

    import math as _math
    A0 = 2.0 * HALF_WIDTH * THICKNESS  # full cross-section area (mm^2)
    L0 = HALF_LENGTH  # gauge half-length (mm)

    true_strain_list = []
    true_stress_list = []

    for frame in step.frames:
        rf_field = frame.fieldOutputs['RF']
        u_field = frame.fieldOutputs['U']

        # Sum RF2 over loaded nodes (quarter-model reaction)
        rf2_quarter = 0.0
        for val in rf_field.values:
            if val.nodeLabel in load_labels:
                rf2_quarter += val.data[1]

        # Get U2 at corner node
        u2 = None
        for val in u_field.values:
            if val.nodeLabel == corner_label:
                u2 = val.data[1]
                break

        if u2 is None:
            continue

        # Only the width is halved by symmetry; tensile force doubles, not quadruples.
        rf2_full = 2.0 * rf2_quarter
        eng_stress = rf2_full / A0
        eng_strain = u2 / L0

        if eng_strain <= 0 or eng_stress <= 0:
            continue

        # Convert to true stress-strain
        true_strain = _math.log(1.0 + eng_strain)
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
                initial_step=None, bounds_low=None, bounds_high=None):
    """
    Nelder-Mead simplex optimization (no scipy required).

    Parameters:
        func: objective function f(x, *args) -> scalar
        x0: initial guess (1D array)
        args: extra arguments passed to func
        maxiter: max iterations
        ftol: convergence tolerance on function value spread
        initial_step: step sizes for initial simplex (array or scalar)
        bounds_low/high: optional bounds (enforced by penalty in func)
    Returns:
        dict with 'x' (best params), 'fun' (best cost), 'nit', 'nfev'
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
            print("\n  Converged at iteration %d (spread=%.2e)" % (iteration, f_spread))
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
            # Accept reflection
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
            # Outside contraction
            x_c = centroid + rho * (x_r - centroid)
            f_c = func(x_c, *args)
            nfev += 1
            if f_c <= f_r:
                simplex[-1] = x_c
                f_vals[-1] = f_c
                continue
        else:
            # Inside contraction
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
# OBJECTIVE FUNCTION
# ============================================================================

def objective(params, exp_strain, exp_stress):
    """
    Run one Abaqus simulation and compute NRMSE vs experiment.
    """
    sigma0, C1, gamma1, C2, gamma2 = params
    ITER_COUNT[0] += 1
    it = ITER_COUNT[0]

    # Bounds penalty
    for val, lo, hi in zip(params, BOUNDS_LOW, BOUNDS_HIGH):
        if val < lo or val > hi:
            cost = 10.0
            print("  Iter %3d: OUT OF BOUNDS -> cost=%.4f" % (it, cost))
            HISTORY.append((it, list(params), cost, 'bounds'))
            return cost

    # Positivity
    if sigma0 <= 0 or C1 <= 0 or gamma1 <= 0 or C2 <= 0 or gamma2 <= 0:
        print("  Iter %3d: NEGATIVE PARAMS" % it)
        HISTORY.append((it, list(params), 10.0, 'negative'))
        return 10.0

    job_name = '%s_%04d' % (JOB_PREFIX, it)

    print("\n  Iter %3d: s0=%.1f C1=%.1f g1=%.1f C2=%.1f g2=%.1f" % (
        it, sigma0, C1, gamma1, C2, gamma2))

    # Generate .inp
    generate_inp(job_name, sigma0, C1, gamma1, C2, gamma2)

    # Run
    success = run_abaqus_job(job_name)
    if not success:
        print("           FAILED to complete")
        HISTORY.append((it, list(params), 5.0, 'failed'))
        cleanup_job(job_name, keep_odb=False)
        return 5.0

    # Extract
    result = extract_results_odb(job_name)
    if result is None:
        print("           No results extracted")
        HISTORY.append((it, list(params), 5.0, 'no_data'))
        cleanup_job(job_name, keep_odb=False)
        return 5.0

    sim_strain, sim_stress = result

    # Compute NRMSE on overlapping range
    strain_min = max(sim_strain.min(), exp_strain.min())
    strain_max = min(sim_strain.max(), exp_strain.max())
    if strain_max <= strain_min + 0.01:
        print("           No overlap")
        HISTORY.append((it, list(params), 5.0, 'no_overlap'))
        cleanup_job(job_name, keep_odb=False)
        return 5.0

    common = np.linspace(strain_min, strain_max, 150)
    sim_interp = np.interp(common, sim_strain, sim_stress)
    exp_interp = np.interp(common, exp_strain, exp_stress)

    rmse = np.sqrt(np.mean((sim_interp - exp_interp) ** 2))
    stress_range = exp_interp.max() - exp_interp.min()
    nrmse = rmse / stress_range if stress_range > 0 else 10.0

    print("           NRMSE=%.5f  RMSE=%.2f MPa" % (nrmse, rmse))

    if nrmse < BEST_COST[0]:
        BEST_COST[0] = nrmse
        BEST_PARAMS[0] = list(params)
        print("           *** NEW BEST ***")

    HISTORY.append((it, list(params), nrmse, 'ok'))

    # Cleanup (keep best ODB)
    is_best = (nrmse <= BEST_COST[0])
    cleanup_job(job_name, keep_odb=is_best)

    return nrmse


def cleanup_job(job_name, keep_odb=False):
    """Remove intermediate Abaqus files to save disk space."""
    exts_remove = ['.dat', '.msg', '.sta', '.com', '.prt', '.sim',
                   '.log', '.inp', '.mdl', '.stt', '.res', '.abq',
                   '.pac', '.sel']
    if not keep_odb:
        exts_remove.append('.odb')

    for ext in exts_remove:
        fpath = job_name + ext
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("INVERSE PARAMETER IDENTIFICATION")
    print("Combined Isotropic-Kinematic Hardening (2 Backstresses)")
    print("Method: Nelder-Mead Simplex (built-in, no scipy)")
    print("=" * 70)

    # Load experimental data
    print("\nLoading experimental data...")
    exp_strain, exp_stress = load_experimental_data()
    print("  %d points, strain [%.4f, %.4f], stress [%.1f, %.1f] MPa" % (
        len(exp_strain), exp_strain.min(), exp_strain.max(),
        exp_stress.min(), exp_stress.max()))

    # Initial guess
    x0 = np.array(X0, dtype=float)
    print("\nInitial guess:")
    print("  sigma0=%.1f  C1=%.1f  gamma1=%.1f  C2=%.1f  gamma2=%.1f" % tuple(x0))
    print("Fixed: Q_inf=%.2f, b=%.2f" % (Q_INF, B_ISO))
    print("\nStarting optimization (max %d iterations)...\n" % MAX_ITER)

    # Run optimization
    result = nelder_mead(
        objective, x0,
        args=(exp_strain, exp_stress),
        maxiter=MAX_ITER,
        ftol=FTOL,
    )

    # Report
    opt = result['x']
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print("  sigma0 = %.4f MPa" % opt[0])
    print("  C1     = %.4f" % opt[1])
    print("  gamma1 = %.4f" % opt[2])
    print("  C2     = %.4f" % opt[3])
    print("  gamma2 = %.4f" % opt[4])
    print("  NRMSE  = %.6f" % result['fun'])
    print("  Iterations: %d" % result['nit'])
    print("  Function evals: %d" % result['nfev'])

    # Save results
    out_dir = os.path.join(WORK_DIR, 'optimization_results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'optimized_parameters.txt'), 'w') as f:
        f.write("OPTIMIZED COMBINED HARDENING PARAMETERS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Method: Nelder-Mead (built-in)\n")
        f.write("Experimental: 0-deg mean of 3 specimens\n\n")
        f.write("Optimized:\n")
        f.write("  sigma0 = %.6f MPa\n" % opt[0])
        f.write("  C1     = %.6f\n" % opt[1])
        f.write("  gamma1 = %.6f\n" % opt[2])
        f.write("  C2     = %.6f\n" % opt[3])
        f.write("  gamma2 = %.6f\n" % opt[4])
        f.write("\nFixed:\n")
        f.write("  Q_inf  = %.6f MPa\n" % Q_INF)
        f.write("  b      = %.6f\n" % B_ISO)
        f.write("\nHill48: R11=%.4f R22=%.4f R33=%.4f R12=%.4f\n" % (R11, R22, R33, R12))
        f.write("\nFit:\n")
        f.write("  NRMSE = %.8f\n" % result['fun'])
        f.write("  Iterations = %d\n" % result['nit'])
        f.write("  Evaluations = %d\n" % result['nfev'])

    with open(os.path.join(out_dir, 'convergence_history.csv'), 'w') as f:
        f.write('Iteration,sigma0,C1,gamma1,C2,gamma2,NRMSE,Status\n')
        for it, p, cost, status in HISTORY:
            f.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%s\n' % (
                it, p[0], p[1], p[2], p[3], p[4], cost, status))

    print("\nResults saved to: %s" % out_dir)
    print("Best ODB kept as: %s_%04d.odb" % (
        JOB_PREFIX, next((h[0] for h in HISTORY if abs(h[2] - BEST_COST[0]) < 1e-10), 0)))


if __name__ == '__main__':
    main()
