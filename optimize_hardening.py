# -*- coding: utf-8 -*-
"""
=============================================================================
Abaqus Python Script: Inverse Parameter Identification
Combined Kinematic Hardening with 2 Backstresses
Using scipy.optimize.minimize (Nelder-Mead)
=============================================================================
Compatible with: Abaqus 2024 (Windows)

Usage:
  abaqus cae noGUI=optimize_hardening.py

This script performs inverse identification of combined isotropic-kinematic
hardening parameters (sigma0, C1, gamma1, C2, gamma2) by:
  1. Running Abaqus FE simulation of uniaxial tensile test
  2. Extracting stress-strain from center element
  3. Comparing with experimental data (UTM stress + DIC strain)
  4. Minimizing error using Nelder-Mead simplex algorithm

Parameters to identify:
  sigma0  - Initial yield stress (MPa)
  C1      - First backstress kinematic modulus
  gamma1  - First backstress rate parameter
  C2      - Second backstress kinematic modulus
  gamma2  - Second backstress rate parameter

The isotropic hardening is defined from Swift/Voce fitting of experimental
data (kept fixed), while kinematic parameters are optimized.

Author: PhD Candidate
Date:   2026
=============================================================================
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh
import material
import section
import assembly
import load
import job
import odbAccess
import numpy as np
import os
import sys
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Working directory
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

# Experimental data file (stress-strain from UTM+DIC)
# The validated Abaqus baseline uses 0-degree specimen 1 as the objective
# target. Post-hoc comparison against the 0-degree experimental mean is
# performed separately in compare_sim_exp.py.
EXP_DATA_FILE = os.path.join(WORK_DIR, 'stress-00-01.csv')
EXP_DATA_DESCRIPTION = (
    '0-degree specimen 01 reference curve; mean-curve comparison is evaluated '
    'separately in compare_sim_exp.py.'
)

# Geometry
GAUGE_LENGTH = 80.0     # mm
GAUGE_WIDTH  = 20.0     # mm
THICKNESS    = 1.5       # mm
HALF_LENGTH  = GAUGE_LENGTH / 2.0
HALF_WIDTH   = GAUGE_WIDTH / 2.0

# Fixed material properties (loaded from material_constants.json)
from material_constants import (
    E_MPA as E_YOUNG, NU, DENSITY,
    Q_INF as Q_INF_FIXED, B_ISO as B_ISO_FIXED,
)

# Hill'48 R-values (from multi-zone DIC extraction)
R11 = 1.0000
R22 = 1.0119
R33 = 0.9347
R12 = 1.0041
R13 = 1.0
R23 = 1.0

# Mesh size
MESH_SIZE = 1.0           # mm

# Applied displacement (enough to reach ~30% strain on half-model)
DISPLACEMENT = 15.0       # mm

# Optimization settings
MAX_ITER = 200
FTOL = 1e-4
XTOL = 1e-4

# Initial guess for parameters [sigma0, C1, gamma1, C2, gamma2]
X0 = [270.0, 5000.0, 50.0, 1000.0, 10.0]

# Parameter bounds (for Nelder-Mead, enforced via penalty)
BOUNDS_LOW  = [150.0, 500.0,  5.0,  100.0,  1.0]
BOUNDS_HIGH = [400.0, 50000.0, 500.0, 20000.0, 200.0]

# Iteration counter
ITER_COUNT = [0]
BEST_COST = [1e20]
BEST_PARAMS = [None]
HISTORY = []


# ============================================================================
# LOAD EXPERIMENTAL DATA
# ============================================================================

def load_experimental_data(filepath):
    """
    Load experimental stress-strain data from CSV.
    Format: Time, Load(kN), Stress(MPa), Strain
    
    Returns: (strain_array, stress_array) - engineering values
    Converts to true stress-strain internally.
    """
    time_data = []
    stress_data = []
    strain_data = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 4:
                try:
                    t = float(row[0])
                    s = float(row[2])   # Stress (MPa)
                    e = float(row[3])   # Strain
                    time_data.append(t)
                    stress_data.append(s)
                    strain_data.append(e)
                except ValueError:
                    continue
    
    eng_stress = np.array(stress_data)
    eng_strain = np.array(strain_data)
    
    # Find UTS index (valid range for true conversion)
    uts_idx = np.argmax(eng_stress)
    
    # Truncate at UTS
    eng_stress = eng_stress[:uts_idx+1]
    eng_strain = eng_strain[:uts_idx+1]
    
    # Convert to true stress-strain
    true_strain = np.log(1 + eng_strain)
    true_stress = eng_stress * (1 + eng_strain)
    
    # Compute plastic strain
    plastic_strain = true_strain - true_stress / E_YOUNG
    
    # Keep only positive plastic strain
    mask = plastic_strain > 0
    
    return {
        'eng_stress': eng_stress,
        'eng_strain': eng_strain,
        'true_stress': true_stress[mask],
        'true_strain': true_strain[mask],
        'plastic_strain': plastic_strain[mask]
    }


# ============================================================================
# ABAQUS MODEL CREATION
# ============================================================================

def create_tensile_model(sigma0, C1, gamma1, C2, gamma2,
                         Q_inf, b_iso, job_name='opt_iter'):
    """
    Create 2D plane-stress uniaxial tensile model with combined hardening.
    Uses quarter symmetry.
    """
    model_name = 'OptModel'
    
    # Clean up
    if model_name in mdb.models:
        del mdb.models[model_name]
    
    myModel = mdb.Model(name=model_name)
    
    # ---- PART ----
    s = myModel.ConstrainedSketch(name='sketch', sheetSize=200.0)
    s.rectangle(point1=(0.0, 0.0), point2=(HALF_WIDTH, HALF_LENGTH))
    
    part = myModel.Part(name='Specimen', dimensionality=TWO_D_PLANAR,
                        type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del s
    
    # ---- MATERIAL ----
    mat = myModel.Material(name='SGCC')
    mat.Elastic(table=((E_YOUNG, NU),))
    mat.Density(table=((DENSITY,),))
    
    # Isotropic hardening table (Voce: sigma = sigma0 + Q*(1-exp(-b*ep)))
    n_pts = 50
    ep_vals = np.linspace(0, 0.5, n_pts)
    sigma_vals = sigma0 + Q_inf * (1.0 - np.exp(-b_iso * ep_vals))
    iso_table = tuple([(float(sigma_vals[i]), float(ep_vals[i])) for i in range(n_pts)])
    
    # Use isotropic hardening for the base; input file will be modified
    mat.Plastic(hardening=ISOTROPIC, table=iso_table)
    
    # Hill'48
    mat.plastic.Potential(
        table=((R11, R22, R33, R12, R13, R23),)
    )
    
    # ---- SECTION ----
    myModel.HomogeneousSolidSection(name='Section', material='SGCC',
                                     thickness=THICKNESS)
    region = part.Set(faces=part.faces[:], name='All')
    part.SectionAssignment(region=region, sectionName='Section')
    
    # ---- ASSEMBLY ----
    a = myModel.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst = a.Instance(name='Specimen-1', part=part, dependent=ON)
    
    # ---- STEP ----
    myModel.StaticStep(name='Tensile', previous='Initial',
                       timePeriod=1.0, initialInc=0.01,
                       minInc=1e-8, maxInc=0.05, maxNumInc=5000,
                       nlgeom=ON)
    
    # ---- FIELD OUTPUT ----
    myModel.FieldOutputRequest(name='F-Output-1', createStepName='Tensile',
                               variables=('S', 'LE', 'PE', 'PEEQ', 'U', 'RF'),
                               frequency=5)
    
    # ---- BOUNDARY CONDITIONS ----
    # Symmetry X (left edge, X=0)
    left_edges = inst.edges.getByBoundingBox(xMin=-0.01, xMax=0.01,
                                              yMin=-0.01, yMax=HALF_LENGTH+0.01)
    left_set = a.Set(edges=left_edges, name='Left')
    myModel.XsymmBC(name='SymX', createStepName='Initial', region=left_set)
    
    # Symmetry Y (bottom edge, Y=0)
    bot_edges = inst.edges.getByBoundingBox(xMin=-0.01, xMax=HALF_WIDTH+0.01,
                                             yMin=-0.01, yMax=0.01)
    bot_set = a.Set(edges=bot_edges, name='Bottom')
    myModel.YsymmBC(name='SymY', createStepName='Initial', region=bot_set)
    
    # Displacement (top edge)
    top_edges = inst.edges.getByBoundingBox(xMin=-0.01, xMax=HALF_WIDTH+0.01,
                                             yMin=HALF_LENGTH-0.01,
                                             yMax=HALF_LENGTH+0.01)
    top_set = a.Set(edges=top_edges, name='Top')
    myModel.DisplacementBC(name='Disp', createStepName='Tensile',
                           region=top_set, u2=DISPLACEMENT)
    
    # ---- MESH ----
    part.setElementType(
        regions=(part.faces[:],),
        elemTypes=(
            mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD,
                         hourglassControl=DEFAULT),
            mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD)
        )
    )
    part.seedPart(size=MESH_SIZE)
    part.generateMesh()
    
    # ---- CENTER ELEMENT SET ----
    cx = HALF_WIDTH / 2.0
    cy = HALF_LENGTH / 2.0
    center_elems = part.elements.getByBoundingBox(
        xMin=cx-MESH_SIZE, xMax=cx+MESH_SIZE,
        yMin=cy-MESH_SIZE, yMax=cy+MESH_SIZE)
    if len(center_elems) > 0:
        part.Set(elements=center_elems[:1], name='CenterElem')
    
    a.regenerate()
    
    # ---- JOB ----
    myJob = mdb.Job(name=job_name, model=model_name,
                    numCpus=4, numDomains=4,
                    memory=90, memoryUnits=PERCENTAGE)
    
    return myModel, myJob


def modify_input_combined(job_name, sigma0, C1, gamma1, C2, gamma2,
                           Q_inf, b_iso):
    """
    Replace the single isotropic *Plastic block produced by CAE with a
    tabulated *Plastic curve that bakes the monotonic combined hardening
    response (Eq. 8) into an isotropic flow stress table. This is the exact
    monotonic equivalent of the Chaboche model under proportional loading
    and is required because Abaqus/Standard does not accept
    *Potential (Hill'48) together with *Plastic, hardening=COMBINED.

    The generated table is:
        sigma(eps_p) = sigma0
                     + Q_inf * (1 - exp(-b_iso * eps_p))
                     + sum_{i=1,2} (C_i / gamma_i) * (1 - exp(-gamma_i * eps_p))
    evaluated on a dense plastic-strain grid.

    Notes:
      - This approximation is exact for monotonic uniaxial loading.
      - For cyclic or load-reversal simulations the backstress evolution
        must be retained explicitly; this monotonic-equivalent path must
        not be used in those cases.
    """
    inp_path = job_name + '.inp'

    with open(inp_path, 'r') as f:
        content = f.read()

    # Pre-compute the monotonic equivalent table
    n_pts = 50
    ep_vals = np.linspace(0.0, 0.5, n_pts)
    sig_vals = (
        sigma0
        + Q_inf * (1.0 - np.exp(-b_iso * ep_vals))
        + (C1 / max(gamma1, 1e-8)) * (1.0 - np.exp(-gamma1 * ep_vals))
        + (C2 / max(gamma2, 1e-8)) * (1.0 - np.exp(-gamma2 * ep_vals))
    )
    table_lines = ['%.6f, %.6f' % (float(sig_vals[i]), float(ep_vals[i]))
                   for i in range(n_pts)]

    lines = content.split('\n')
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.strip().upper().startswith('*PLASTIC'):
            # Replace the plastic block with a tabulated isotropic flow stress
            new_lines.append('*Plastic')
            new_lines.extend(table_lines)

            # Skip the original data lines belonging to *Plastic
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith('*'):
                    break
                i += 1
            continue
        else:
            new_lines.append(line)

        i += 1

    with open(inp_path, 'w') as f:
        f.write('\n'.join(new_lines))

    return True


def extract_sim_results(job_name):
    """
    Extract loading-direction stress (S22) and log strain (LE22) from the
    center element of the completed simulation. The tensile axis is Y, so
    component index 1 of the plane-stress tensor is the loading direction.
    """
    odb_path = job_name + '.odb'
    
    if not os.path.exists(odb_path):
        return None
    
    try:
        odb = odbAccess.openOdb(path=odb_path, readOnly=True)
    except:
        return None
    
    step_obj = odb.steps['Tensile']
    inst_name = 'SPECIMEN-1'
    
    s22_list = []
    le22_list = []
    peeq_list = []
    
    for frame in step_obj.frames:
        s_field = frame.fieldOutputs['S']
        e_field = frame.fieldOutputs['LE']
        peeq_field = frame.fieldOutputs['PEEQ']
        
        # Try center element set
        try:
            region = odb.rootAssembly.instances[inst_name].elementSets['CENTERELEM']
            s_sub = s_field.getSubset(region=region)
            e_sub = e_field.getSubset(region=region)
            p_sub = peeq_field.getSubset(region=region)
        except:
            # Fallback: use all elements, pick middle
            s_sub = s_field
            e_sub = e_field
            p_sub = peeq_field
        
        if len(s_sub.values) > 0:
            # Plane-stress tensor components: [S11, S22, S12].
            # Loading axis is Y, so component index 1 is S22 (loading stress).
            s22 = s_sub.values[0].data[1]
            le22 = e_sub.values[0].data[1]
            pq = p_sub.values[0].data
            
            s22_list.append(s22)
            le22_list.append(le22)
            if hasattr(pq, '__len__'):
                peeq_list.append(pq[0] if len(pq) > 0 else float(pq))
            else:
                peeq_list.append(float(pq))
    
    odb.close()
    
    return {
        'stress_22': np.array(s22_list),
        'strain_22': np.array(le22_list),
        'peeq': np.array(peeq_list)
    }


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective_function(params, exp_data):
    """
    Objective function for Nelder-Mead optimization.
    
    Runs Abaqus simulation with given parameters, extracts stress-strain
    from center element, and computes normalized mean squared error
    against experimental data.
    
    Parameters:
        params: [sigma0, C1, gamma1, C2, gamma2]
        exp_data: dict with experimental true_stress, true_strain
    
    Returns:
        cost: normalized root mean squared error
    """
    sigma0, C1, gamma1, C2, gamma2 = params
    
    ITER_COUNT[0] += 1
    iter_num = ITER_COUNT[0]
    
    # Enforce bounds via penalty
    penalty = 0.0
    for val, lo, hi in zip(params, BOUNDS_LOW, BOUNDS_HIGH):
        if val < lo:
            penalty += 1000 * (lo - val)**2
        if val > hi:
            penalty += 1000 * (val - hi)**2
    
    if penalty > 0:
        cost = 1e10 + penalty
        print("Iter %d: OUT OF BOUNDS, penalty=%.2f" % (iter_num, penalty))
        HISTORY.append({'iter': iter_num, 'params': list(params), 
                       'cost': cost, 'status': 'bounds'})
        return cost
    
    # Ensure positive values
    if sigma0 <= 0 or C1 <= 0 or gamma1 <= 0 or C2 <= 0 or gamma2 <= 0:
        print("Iter %d: NEGATIVE PARAMS" % iter_num)
        return 1e10
    
    job_name = 'opt_iter_%04d' % iter_num
    
    print("\n--- Iteration %d ---" % iter_num)
    print("  sigma0=%.2f, C1=%.2f, gamma1=%.2f, C2=%.2f, gamma2=%.2f" % (
        sigma0, C1, gamma1, C2, gamma2))
    
    try:
        # Create model
        myModel, myJob = create_tensile_model(
            sigma0, C1, gamma1, C2, gamma2,
            Q_INF_FIXED, B_ISO_FIXED, job_name=job_name
        )
        
        # Write input file
        myJob.writeInput()
        
        # Modify for combined hardening
        modify_input_combined(job_name, sigma0, C1, gamma1, C2, gamma2,
                              Q_INF_FIXED, B_ISO_FIXED)
        
        # Submit from input file
        run_name = job_name
        run_job = mdb.JobFromInputFile(
            name=run_name,
            inputFileName=job_name + '.inp',
            numCpus=4, numDomains=4,
            memory=90, memoryUnits=PERCENTAGE
        )
        run_job.submit()
        run_job.waitForCompletion()
        
        # Extract results
        sim_results = extract_sim_results(run_name)
        
        if sim_results is None or len(sim_results['stress_22']) < 5:
            print("  FAILED: No results extracted")
            HISTORY.append({'iter': iter_num, 'params': list(params),
                           'cost': 1e8, 'status': 'no_results'})
            return 1e8
        
        # Interpolate simulation to experimental strain points
        sim_strain = sim_results['strain_22']
        sim_stress = sim_results['stress_22']
        
        exp_strain = exp_data['true_strain']
        exp_stress = exp_data['true_stress']
        
        # Common strain range
        strain_min = max(sim_strain.min(), exp_strain.min())
        strain_max = min(sim_strain.max(), exp_strain.max())
        
        if strain_max <= strain_min:
            print("  FAILED: No overlapping strain range")
            return 1e8
        
        # Interpolate both to common points
        n_compare = 100
        common_strain = np.linspace(strain_min, strain_max, n_compare)
        
        sim_interp = np.interp(common_strain, sim_strain, sim_stress)
        exp_interp = np.interp(common_strain, exp_strain, exp_stress)
        
        # Normalized Root Mean Square Error
        nrmse = np.sqrt(np.mean((sim_interp - exp_interp)**2)) / \
                (exp_interp.max() - exp_interp.min())
        
        # Also compute R-squared
        ss_res = np.sum((sim_interp - exp_interp)**2)
        ss_tot = np.sum((exp_interp - np.mean(exp_interp))**2)
        r_squared = 1 - ss_res / ss_tot
        
        cost = nrmse
        
        print("  NRMSE = %.6f, R² = %.6f" % (nrmse, r_squared))
        
        # Track best
        if cost < BEST_COST[0]:
            BEST_COST[0] = cost
            BEST_PARAMS[0] = list(params)
            print("  *** NEW BEST ***")
        
        HISTORY.append({
            'iter': iter_num, 'params': list(params),
            'cost': cost, 'r_squared': r_squared,
            'status': 'success'
        })
        
        return cost
        
    except Exception as e:
        print("  EXCEPTION: %s" % str(e))
        HISTORY.append({'iter': iter_num, 'params': list(params),
                       'cost': 1e8, 'status': 'exception'})
        return 1e8
    
    finally:
        # Clean up files to save disk space (keep ODB of best)
        for ext in ['.dat', '.msg', '.sta', '.com', '.prt', '.sim', '.log']:
            fpath = job_name + ext
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except:
                    pass


# ============================================================================
# OPTIMIZATION DRIVER
# ============================================================================

def run_optimization():
    """
    Main optimization routine using Nelder-Mead simplex.
    """
    print("=" * 70)
    print("INVERSE PARAMETER IDENTIFICATION")
    print("Combined Isotropic-Kinematic Hardening (2 Backstresses)")
    print("Method: Nelder-Mead Simplex")
    print("=" * 70)
    
    # Load experimental data
    print("\nLoading experimental data...")
    exp_data = load_experimental_data(EXP_DATA_FILE)
    print("  Source: %s" % EXP_DATA_DESCRIPTION)
    print("  Loaded %d data points" % len(exp_data['true_stress']))
    print("  Stress range: %.1f - %.1f MPa" % (
        exp_data['true_stress'].min(), exp_data['true_stress'].max()))
    print("  Strain range: %.6f - %.6f" % (
        exp_data['true_strain'].min(), exp_data['true_strain'].max()))
    
    # Initial guess
    x0 = np.array(X0)
    print("\nInitial parameters:")
    print("  sigma0 = %.2f MPa" % x0[0])
    print("  C1     = %.2f" % x0[1])
    print("  gamma1 = %.2f" % x0[2])
    print("  C2     = %.2f" % x0[3])
    print("  gamma2 = %.2f" % x0[4])
    print("\nFixed isotropic: Q_inf = %.2f, b = %.2f" % (Q_INF_FIXED, B_ISO_FIXED))
    print("Starting optimization...\n")
    
    # Run Nelder-Mead
    from scipy.optimize import minimize
    
    result = minimize(
        objective_function,
        x0,
        args=(exp_data,),
        method='Nelder-Mead',
        options={
            'maxiter': MAX_ITER,
            'maxfev': MAX_ITER * 5,
            'fatol': FTOL,
            'xatol': XTOL,
            'disp': True,
            'adaptive': True
        }
    )
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    opt_params = result.x
    print("\nOptimized Parameters:")
    print("  sigma0 = %.4f MPa" % opt_params[0])
    print("  C1     = %.4f" % opt_params[1])
    print("  gamma1 = %.4f" % opt_params[2])
    print("  C2     = %.4f" % opt_params[3])
    print("  gamma2 = %.4f" % opt_params[4])
    print("\nFinal cost (NRMSE): %.6f" % result.fun)
    print("Number of iterations: %d" % result.nit)
    print("Number of function evaluations: %d" % result.nfev)
    print("Success: %s" % result.success)
    print("Message: %s" % result.message)
    
    # Save results
    save_results(opt_params, result, exp_data)
    
    return result, exp_data


def save_results(opt_params, opt_result, exp_data):
    """
    Save optimization results and convergence history.
    """
    output_dir = os.path.join(WORK_DIR, 'optimization_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save parameters
    with open(os.path.join(output_dir, 'optimized_parameters.txt'), 'w') as f:
        f.write("OPTIMIZED COMBINED HARDENING PARAMETERS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Method: Nelder-Mead Simplex\n")
        f.write("Experimental data: %s\n\n" % EXP_DATA_FILE)
        f.write("Experimental data description: %s\n\n" % EXP_DATA_DESCRIPTION)
        f.write("Optimized Parameters:\n")
        f.write("  sigma0 = %.6f MPa\n" % opt_params[0])
        f.write("  C1     = %.6f\n" % opt_params[1])
        f.write("  gamma1 = %.6f\n" % opt_params[2])
        f.write("  C2     = %.6f\n" % opt_params[3])
        f.write("  gamma2 = %.6f\n" % opt_params[4])
        f.write("\nFixed Isotropic Parameters:\n")
        f.write("  Q_inf  = %.6f MPa\n" % Q_INF_FIXED)
        f.write("  b      = %.6f\n" % B_ISO_FIXED)
        f.write("\nHill'48 R-values:\n")
        f.write("  R11 = %.6f\n" % R11)
        f.write("  R22 = %.6f\n" % R22)
        f.write("  R33 = %.6f\n" % R33)
        f.write("  R12 = %.6f\n" % R12)
        f.write("\nOptimization Statistics:\n")
        f.write("  Final cost (NRMSE): %.8f\n" % opt_result.fun)
        f.write("  Iterations: %d\n" % opt_result.nit)
        f.write("  Function evaluations: %d\n" % opt_result.nfev)
        f.write("  Success: %s\n" % opt_result.success)
    
    # Save convergence history
    with open(os.path.join(output_dir, 'convergence_history.csv'), 'w') as f:
        f.write('Iteration,sigma0,C1,gamma1,C2,gamma2,Cost,Status\n')
        for h in HISTORY:
            p = h['params']
            f.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%s\n' % (
                h['iter'], p[0], p[1], p[2], p[3], p[4],
                h['cost'], h['status']))
    
    print("\nResults saved to: %s" % output_dir)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    result, exp_data = run_optimization()
