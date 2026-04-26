# -*- coding: utf-8 -*-
"""
=============================================================================
Abaqus Python Script: 2D Uniaxial Tensile Test Simulation
Combined Isotropic-Kinematic Hardening with Hill'48 Anisotropy
=============================================================================
Compatible with: Abaqus 2024 (run via Abaqus/CAE command line)

Usage:
  abaqus cae noGUI=abaqus_tensile_model.py
  
  OR from Abaqus/CAE:
  File -> Run Script -> abaqus_tensile_model.py

This script creates a 2D plane-stress uniaxial tensile test model with:
  - Combined nonlinear isotropic/kinematic hardening (2 backstresses)
  - Hill'48 anisotropic yield criterion
  - Displacement-controlled loading

Author: PhD Candidate
Date:   2026
=============================================================================
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
import regionToolset
import mesh
import step
import material
import section
import assembly
import interaction
import load
import job
import visualization
import odbAccess
import numpy as np
import os
import sys

# ============================================================================
# MODEL PARAMETERS (to be modified by optimization script)
# ============================================================================

# Geometry (mm) - Half model due to symmetry
GAUGE_LENGTH = 80.0    # Full gauge length
GAUGE_WIDTH  = 20.0    # Full gauge width  
THICKNESS    = 1.5     # Sheet thickness

# Use quarter model (symmetry in both X and Y)
HALF_LENGTH = GAUGE_LENGTH / 2.0   # 40 mm
HALF_WIDTH  = GAUGE_WIDTH / 2.0    # 10 mm

# Material properties
E_YOUNG     = 200000.0   # MPa (200 GPa)
NU_POISSON  = 0.3
DENSITY     = 7.85e-9    # tonne/mm^3

# Hill'48 Anisotropy R-values (from multi-zone DIC extraction)
R11 = 1.0000
R22 = 1.0119
R33 = 0.9347
R12 = 1.0041
R13 = 1.0
R23 = 1.0

# Combined Hardening Parameters (initial guess, to be optimized)
# Isotropic: sigma_y = sigma0 + Q_inf*(1 - exp(-b*eps_p))
SIGMA0 = 270.0     # Initial yield stress (MPa)
Q_INF  = 150.0     # Isotropic hardening saturation
B_ISO  = 10.0      # Isotropic hardening rate

# Kinematic (2 backstresses): alpha_i = C_i/gamma_i * (1 - exp(-gamma_i*eps_p))
C1     = 5000.0    # Kinematic hardening modulus 1
GAMMA1 = 50.0      # Kinematic hardening rate 1
C2     = 1000.0    # Kinematic hardening modulus 2
GAMMA2 = 10.0      # Kinematic hardening rate 2

# Loading
DISPLACEMENT = 25.0   # mm (total applied displacement on half-model)
# Corresponds to ~62.5% engineering strain on gauge

# Mesh
MESH_SIZE = 1.0    # mm (element size)

# Job
JOB_NAME = 'Tensile_CKH'


def create_isotropic_hardening_table(sigma0, Q_inf, b_iso, max_strain=0.5, n_points=100):
    """
    Generate tabular isotropic hardening data for Abaqus.
    sigma = sigma0 + Q_inf*(1 - exp(-b*eps_p))
    
    Returns: tuple of tuples ((sigma1, eps_p1), (sigma2, eps_p2), ...)
    """
    eps_p = np.linspace(0, max_strain, n_points)
    sigma = sigma0 + Q_inf * (1.0 - np.exp(-b_iso * eps_p))
    
    table = []
    for i in range(n_points):
        table.append((float(sigma[i]), float(eps_p[i])))
    
    return tuple(table)


def create_model(sigma0, C1, gamma1, C2, gamma2, 
                 Q_inf=None, b_iso=None, iso_table=None,
                 hill_R=None, job_name=None, displacement=None):
    """
    Create complete Abaqus model for uniaxial tensile test.
    
    Parameters:
        sigma0:  Initial yield stress (MPa)
        C1, gamma1: First backstress kinematic parameters
        C2, gamma2: Second backstress kinematic parameters
        Q_inf, b_iso: Isotropic hardening parameters (Voce)
        iso_table: Alternative - direct tabular isotropic data
        hill_R: dict with R11, R22, R33, R12, R13, R23
        job_name: Name for the Abaqus job
        displacement: Applied displacement (mm)
    """
    
    if job_name is None:
        job_name = JOB_NAME
    if displacement is None:
        displacement = DISPLACEMENT
    if hill_R is None:
        hill_R = {'R11': R11, 'R22': R22, 'R33': R33, 
                  'R12': R12, 'R13': R13, 'R23': R23}
    
    # ------------------------------------------------------------------
    # Create Model
    # ------------------------------------------------------------------
    model_name = 'Tensile_Model'
    
    # Delete existing model if present
    if model_name in mdb.models:
        del mdb.models[model_name]
    
    myModel = mdb.Model(name=model_name)
    
    # ------------------------------------------------------------------
    # Create Part (Quarter model - 2D Plane Stress)
    # ------------------------------------------------------------------
    mySketch = myModel.ConstrainedSketch(name='TensileSketch',
                                          sheetSize=200.0)
    
    # Rectangle: origin at center, quarter model in +X, +Y quadrant
    mySketch.rectangle(point1=(0.0, 0.0), 
                       point2=(HALF_WIDTH, HALF_LENGTH))
    
    myPart = myModel.Part(name='Specimen',
                          dimensionality=TWO_D_PLANAR,
                          type=DEFORMABLE_BODY)
    myPart.BaseShell(sketch=mySketch)
    del mySketch
    
    # ------------------------------------------------------------------
    # Create Material
    # ------------------------------------------------------------------
    myMaterial = myModel.Material(name='SGCC_Steel')
    
    # Elastic
    myMaterial.Elastic(table=((E_YOUNG, NU_POISSON),))
    
    # Density
    myMaterial.Density(table=((DENSITY,),))
    
    # Plastic: Combined Hardening (nonlinear kinematic + isotropic)
    # In Abaqus, combined hardening needs:
    #   1. Plastic data table (isotropic part): (yield_stress, plastic_strain)
    #   2. Kinematic hardening parameters: backstress data
    
    # Generate isotropic hardening table
    if iso_table is not None:
        hardening_table = iso_table
    elif Q_inf is not None and b_iso is not None:
        hardening_table = create_isotropic_hardening_table(
            sigma0, Q_inf, b_iso, max_strain=0.5, n_points=100)
    else:
        # If no isotropic data, use constant yield (pure kinematic)
        hardening_table = ((float(sigma0), 0.0),)
    
    # Define plastic with combined hardening
    myMaterial.Plastic(
        table=hardening_table,
        hardening=COMBINED,
        dataType=HALF_CYCLE,
        numBackstresses=2
    )
    
    # Kinematic hardening parameters (2 backstresses)
    # Format: ((C1, gamma1, C2, gamma2),)
    myMaterial.plastic.CyclicHardening(
        table=hardening_table
    )
    
    # Set kinematic parameters
    # In Abaqus, for COMBINED hardening with PARAMETERS data type:
    # We need to redefine using parameters approach
    
    # Delete and recreate with proper format
    del myMaterial.plastic
    
    # Method: Use PARAMETERS dataType for kinematic part
    myMaterial.Plastic(
        table=hardening_table,
        hardening=COMBINED, 
        dataType=PARAMETERS,
        numBackstresses=2
    )
    
    # CyclicHardening defines the isotropic part evolution
    myMaterial.plastic.CyclicHardening(
        table=hardening_table,
        parameters=ON
    )
    
    # KinematicHardening backstress parameters
    # For PARAMETERS type with 2 backstresses: ((C1, gamma1, C2, gamma2, temperature),)
    backstress_table = ((float(C1), float(gamma1), float(C2), float(gamma2)),)
    
    # We redefine properly
    del myMaterial.plastic
    
    # ---- CORRECT APPROACH for Abaqus Combined Hardening ----
    # Step 1: Define Plastic with COMBINED + PARAMETERS
    # The plastic table is (yield_stress, 0.0) for initial yield
    myMaterial.Plastic(
        hardening=COMBINED,
        dataType=PARAMETERS,
        numBackstresses=2,
        table=((float(sigma0), 0.0),)
    )
    
    # Step 2: CyclicHardening for isotropic component
    # Parameters ON => (sigma|0, Q_inf, b)
    if Q_inf is not None and b_iso is not None:
        myMaterial.plastic.CyclicHardening(
            parameters=ON,
            table=((float(sigma0), float(Q_inf), float(b_iso)),)
        )
    else:
        # No isotropic evolution
        myMaterial.plastic.CyclicHardening(
            parameters=ON,
            table=((float(sigma0), 0.0, 0.0),)
        )
    
    # Step 3: Kinematic hardening parameters 
    # For 2 backstresses: C1, gamma1, C2, gamma2
    # This is set via the plastic keyword in input file
    # In CAE, we handle this through the Parameters approach
    
    # Note: Abaqus CAE with PARAMETERS + numBackstresses=2
    # automatically expects the table format ((C1, gamma1, C2, gamma2),)
    # This is handled by the Plastic definition above when dataType=PARAMETERS
    # We need to add the backstress data
    
    # Actually in Abaqus CAE Python API:
    # When hardening=COMBINED, dataType=PARAMETERS, numBackstresses=2
    # The main Plastic table should be: ((sigma0, C1, gamma1, C2, gamma2),)
    
    # Let's redo this correctly
    del myMaterial.plastic
    
    # FINAL CORRECT DEFINITION:
    # For Combined hardening with PARAMETERS and 2 backstresses in Abaqus:
    # Plastic table: ((yield_stress, 0.0),)   - just initial yield
    # Then separately set kinematic parameters
    
    # Use the TABULAR approach instead - more reliable in scripting
    # Generate stress-strain data that combines isotropic + kinematic
    
    # Actually, let's use the input file keyword approach via editKeywords
    # This is the most reliable method
    
    # For now, define basic plasticity; we'll modify the input file
    myMaterial.Plastic(
        hardening=ISOTROPIC,
        table=hardening_table if isinstance(hardening_table[0], tuple) and len(hardening_table[0]) == 2 
              else ((float(sigma0), 0.0),)
    )
    
    # Hill'48 Anisotropy
    myMaterial.plastic.Potential(
        table=((float(hill_R['R11']), float(hill_R['R22']), float(hill_R['R33']),
                float(hill_R['R12']), float(hill_R['R13']), float(hill_R['R23'])),)
    )
    
    # ------------------------------------------------------------------
    # Create Section and Assign
    # ------------------------------------------------------------------
    myModel.HomogeneousSolidSection(
        name='SteelSection',
        material='SGCC_Steel',
        thickness=THICKNESS
    )
    
    # Assign section to part
    region = myPart.Set(
        faces=myPart.faces[:],
        name='AllElements'
    )
    myPart.SectionAssignment(
        region=region,
        sectionName='SteelSection'
    )
    
    # ------------------------------------------------------------------
    # Create Assembly
    # ------------------------------------------------------------------
    myAssembly = myModel.rootAssembly
    myAssembly.DatumCsysByDefault(CARTESIAN)
    myInstance = myAssembly.Instance(
        name='Specimen-1',
        part=myPart,
        dependent=ON
    )
    
    # ------------------------------------------------------------------
    # Create Step
    # ------------------------------------------------------------------
    myModel.StaticStep(
        name='Tensile',
        previous='Initial',
        timePeriod=1.0,
        initialInc=0.01,
        minInc=1e-8,
        maxInc=0.05,
        maxNumInc=10000,
        nlgeom=ON
    )
    
    # Field output
    myModel.FieldOutputRequest(
        name='F-Output-1',
        createStepName='Tensile',
        variables=('S', 'E', 'PE', 'PEEQ', 'U', 'RF', 'LE'),
        frequency=10
    )
    
    # History output at center element (set created after meshing below)
    # Placeholder: will be overwritten after CenterElement set exists
    
    # ------------------------------------------------------------------
    # Create Boundary Conditions
    # ------------------------------------------------------------------
    
    # Symmetry BC on left edge (X = 0): U1 = 0
    left_edges = myInstance.edges.getByBoundingBox(
        xMin=-0.01, xMax=0.01, yMin=-0.01, yMax=HALF_LENGTH+0.01)
    left_set = myAssembly.Set(edges=left_edges, name='LeftEdge')
    myModel.XsymmBC(
        name='Symmetry_X',
        createStepName='Initial',
        region=left_set
    )
    
    # Symmetry BC on bottom edge (Y = 0): U2 = 0
    bottom_edges = myInstance.edges.getByBoundingBox(
        xMin=-0.01, xMax=HALF_WIDTH+0.01, yMin=-0.01, yMax=0.01)
    bottom_set = myAssembly.Set(edges=bottom_edges, name='BottomEdge')
    myModel.YsymmBC(
        name='Symmetry_Y',
        createStepName='Initial',
        region=bottom_set
    )
    
    # Displacement BC on top edge (Y = HALF_LENGTH): U2 = displacement
    top_edges = myInstance.edges.getByBoundingBox(
        xMin=-0.01, xMax=HALF_WIDTH+0.01,
        yMin=HALF_LENGTH-0.01, yMax=HALF_LENGTH+0.01)
    top_set = myAssembly.Set(edges=top_edges, name='TopEdge')
    myModel.DisplacementBC(
        name='Applied_Disp',
        createStepName='Tensile',
        region=top_set,
        u2=displacement
    )
    
    # ------------------------------------------------------------------
    # Create Mesh
    # ------------------------------------------------------------------
    myPart.setElementType(
        regions=(myPart.faces[:],),
        elemTypes=(
            mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD,
                         secondOrderAccuracy=OFF,
                         hourglassControl=DEFAULT),
            mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD)
        )
    )
    
    myPart.seedPart(size=MESH_SIZE)
    myPart.generateMesh()
    
    print("Mesh generated: %d elements, %d nodes" % (
        len(myPart.elements), len(myPart.nodes)))
    
    # ------------------------------------------------------------------
    # Create set for center element (for data extraction)
    # ------------------------------------------------------------------
    # Find element closest to center of gauge
    center_x = HALF_WIDTH / 2.0
    center_y = HALF_LENGTH / 2.0
    
    center_elements = myPart.elements.getByBoundingBox(
        xMin=center_x - MESH_SIZE,
        xMax=center_x + MESH_SIZE,
        yMin=center_y - MESH_SIZE,
        yMax=center_y + MESH_SIZE
    )
    
    if len(center_elements) > 0:
        myPart.Set(elements=center_elements[:1], name='CenterElement')
        print("Center element set created")
    
    # Regenerate assembly
    myAssembly.regenerate()
    
    # History output at center element (now that the set exists)
    instance_name = myAssembly.instances.keys()[0]
    center_set_name = instance_name + '.CenterElement'
    if center_set_name in myAssembly.allSets:
        myModel.HistoryOutputRequest(
            name='H-Output-1',
            createStepName='Tensile',
            variables=('S11', 'S22', 'S12', 'E11', 'E22', 'E12',
                       'PE11', 'PE22', 'PE12', 'PEEQ'),
            frequency=1,
            region=myAssembly.allSets[center_set_name]
        )
    
    # ------------------------------------------------------------------
    # Create and Submit Job
    # ------------------------------------------------------------------
    myJob = mdb.Job(
        name=job_name,
        model=model_name,
        type=ANALYSIS,
        numCpus=4,
        numDomains=4,
        memory=90,
        memoryUnits=PERCENTAGE
    )
    
    return myModel, myJob


def modify_input_for_combined_hardening(job_name, sigma0, C1, gamma1, C2, gamma2,
                                         Q_inf=0.0, b_iso=0.0):
    """
    Modify the Abaqus input file to use proper combined hardening keywords.
    This is called AFTER writing the input file but BEFORE submitting.
    
    The input file modification replaces the ISOTROPIC plasticity with
    COMBINED HARDENING (PARAMETERS, 2 backstresses).
    """
    inp_file = job_name + '.inp'
    
    if not os.path.exists(inp_file):
        print("ERROR: Input file %s not found!" % inp_file)
        return False
    
    with open(inp_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    skip_plastic = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Find the *Plastic keyword
        if line.strip().upper().startswith('*PLASTIC'):
            # Replace with combined hardening
            new_lines.append('*Plastic, hardening=COMBINED, dataType=PARAMETERS, '
                           'number backstresses=2\n')
            new_lines.append('%.6f\n' % sigma0)
            
            # Add cyclic hardening (isotropic component)
            new_lines.append('*Cyclic Hardening, parameters\n')
            new_lines.append('%.6f, %.6f, %.6f\n' % (sigma0, Q_inf, b_iso))
            
            # Skip original plastic data lines
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                i += 1
            
            # Now add kinematic parameters
            # They are embedded in the *Plastic definition
            # Actually for input file format:
            # After the yield stress line, we need backstress parameters
            # Format: C1, gamma1, C2, gamma2
            
            # Insert before the next keyword
            # The backstress params go right after the yield stress in *Plastic
            # Let me restructure:
            
            # Remove what we just added and redo properly
            new_lines = new_lines[:-4]  # Remove last 4 lines (Plastic + sigma0 + CyclicHardening + params)
            
            new_lines.append('*Plastic, hardening=COMBINED, dataType=PARAMETERS, '
                           'number backstresses=2\n')
            new_lines.append('%.6f\n' % sigma0)
            new_lines.append('%.6f, %.6f\n' % (C1, gamma1))
            new_lines.append('%.6f, %.6f\n' % (C2, gamma2))
            new_lines.append('*Cyclic Hardening, parameters\n')
            new_lines.append('%.6f, %.6f, %.6f\n' % (sigma0, Q_inf, b_iso))
            
            continue
        else:
            new_lines.append(line)
        
        i += 1
    
    # Write modified input file
    with open(inp_file, 'w') as f:
        f.writelines(new_lines)
    
    print("Input file modified for combined hardening: %s" % inp_file)
    return True


def extract_results(job_name, set_name='CENTERELEMENT'):
    """
    Extract stress-strain results from Abaqus ODB file.
    
    Returns: dict with stress and strain arrays
    """
    odb_file = job_name + '.odb'
    
    if not os.path.exists(odb_file):
        print("ERROR: ODB file %s not found!" % odb_file)
        return None
    
    odb = odbAccess.openOdb(path=odb_file, readOnly=True)
    
    step = odb.steps['Tensile']
    
    # Try to find the element set
    instance_name = 'SPECIMEN-1'
    
    stress_11 = []
    strain_11 = []
    pe_11 = []
    peeq = []
    
    for frame in step.frames:
        # Get stress
        stress_field = frame.fieldOutputs['S']
        strain_field = frame.fieldOutputs['LE']  # Logarithmic strain
        pe_field = frame.fieldOutputs['PE']
        peeq_field = frame.fieldOutputs['PEEQ']
        
        # Try to get from set, otherwise use center element
        try:
            region = odb.rootAssembly.instances[instance_name].elementSets[set_name]
            s_values = stress_field.getSubset(region=region)
            e_values = strain_field.getSubset(region=region)
            pe_values = pe_field.getSubset(region=region)
            peeq_values = peeq_field.getSubset(region=region)
        except:
            # Use all elements and pick center
            s_values = stress_field
            e_values = strain_field
            pe_values = pe_field
            peeq_values = peeq_field
        
        if len(s_values.values) > 0:
            # S22 component (index 1) — loading is in Y-direction
            s11 = s_values.values[0].data[1]  # S22 (axial)
            e11 = e_values.values[0].data[1]  # LE22 (axial)
            p11 = pe_values.values[0].data[1]  # PE22 (axial)
            eq = peeq_values.values[0].data
            
            stress_11.append(s11)
            strain_11.append(e11)
            pe_11.append(p11)
            if hasattr(eq, '__len__'):
                peeq.append(eq[0] if len(eq) > 0 else eq)
            else:
                peeq.append(eq)
    
    odb.close()
    
    results = {
        'stress_11': np.array(stress_11),
        'strain_11': np.array(strain_11),
        'pe_11': np.array(pe_11),
        'peeq': np.array(peeq),
    }
    
    return results


def run_simulation(sigma0, C1, gamma1, C2, gamma2,
                   Q_inf=150.0, b_iso=10.0, iso_table=None,
                   hill_R=None, job_name=None, displacement=None,
                   submit=True, wait=True):
    """
    Complete workflow: create model, write input, modify for combined
    hardening, submit, and extract results.
    """
    if job_name is None:
        job_name = JOB_NAME
    
    print("\n" + "=" * 60)
    print("Running simulation: %s" % job_name)
    print("  sigma0=%.2f, C1=%.2f, gamma1=%.2f, C2=%.2f, gamma2=%.2f" % (
        sigma0, C1, gamma1, C2, gamma2))
    print("  Q_inf=%.2f, b_iso=%.2f" % (Q_inf, b_iso))
    print("=" * 60)
    
    # Create model
    myModel, myJob = create_model(
        sigma0, C1, gamma1, C2, gamma2,
        Q_inf=Q_inf, b_iso=b_iso, iso_table=iso_table,
        hill_R=hill_R, job_name=job_name, displacement=displacement
    )
    
    if submit:
        # Write input file first
        myJob.writeInput()
        
        # Modify input file for proper combined hardening
        modify_input_for_combined_hardening(
            job_name, sigma0, C1, gamma1, C2, gamma2,
            Q_inf=Q_inf, b_iso=b_iso
        )
        
        # Submit modified input file
        myJob2 = mdb.JobFromInputFile(
            name=job_name + '_run',
            inputFileName=job_name + '.inp',
            numCpus=4,
            numDomains=4,
            memory=90,
            memoryUnits=PERCENTAGE
        )
        
        myJob2.submit()
        
        if wait:
            myJob2.waitForCompletion()
            print("Simulation completed: %s" % job_name)
            
            # Extract results
            results = extract_results(job_name + '_run')
            return results
    
    return None


# ============================================================================
# MAIN - Create and run the base model
# ============================================================================
if __name__ == '__main__':
    
    # Default parameters
    results = run_simulation(
        sigma0=SIGMA0, C1=C1, gamma1=GAMMA1, C2=C2, gamma2=GAMMA2,
        Q_inf=Q_INF, b_iso=B_ISO,
        hill_R={'R11': R11, 'R22': R22, 'R33': R33, 
                'R12': R12, 'R13': R13, 'R23': R23},
        submit=True, wait=True
    )
    
    if results:
        print("\nExtracted Results:")
        print("  Stress range: %.1f - %.1f MPa" % (
            results['stress_11'].min(), results['stress_11'].max()))
        print("  Strain range: %.6f - %.6f" % (
            results['strain_11'].min(), results['strain_11'].max()))
