"""
Extract results from Tensile_CKH.odb to CSV files.
Run on Abaqus machine:  abaqus python extract_odb_results.py

Outputs:
  sim_force_disp.csv      — Force-displacement (history at corner + summed RF)
  sim_stress_strain.csv   — S22 vs LE22/PE22 at center element (field output)
"""

from odbAccess import openOdb
import numpy as np
import csv
import os

ODB_NAME = 'Tensile_CKH.odb'
STEP_NAME = 'Tensile'
INSTANCE_NAME = 'SPECIMEN-1'

# Quarter-model geometry
HALF_WIDTH = 10.0   # mm (full gauge = 20 mm)
THICKNESS = 1.5     # mm
HALF_AREA = HALF_WIDTH * THICKNESS  # 15 mm^2
HALF_GAUGE = 40.0   # mm (full gauge length = 80 mm)


def extract_results():
    print("Opening %s ..." % ODB_NAME)
    odb = openOdb(path=ODB_NAME, readOnly=True)
    step = odb.steps[STEP_NAME]
    instance = odb.rootAssembly.instances[INSTANCE_NAME]

    # ------------------------------------------------------------------
    # 1) Field output: S22, LE22, PE22, PEEQ at center element
    #    Center element = middle of gauge (x~5, y~20)
    # ------------------------------------------------------------------
    # Find center element (closest to x=5, y=20)
    n_elems = len(instance.elements)
    print("  Total elements: %d" % n_elems)

    # Compute element centroids
    node_map = {}
    for node in instance.nodes:
        node_map[node.label] = node.coordinates

    best_elem = None
    best_dist = 1e20
    for elem in instance.elements:
        cx = sum(node_map[n][0] for n in elem.connectivity) / len(elem.connectivity)
        cy = sum(node_map[n][1] for n in elem.connectivity) / len(elem.connectivity)
        dist = (cx - 5.0)**2 + (cy - 20.0)**2
        if dist < best_dist:
            best_dist = dist
            best_elem = elem.label

    print("  Center element: %d (dist=%.3f)" % (best_elem, best_dist**0.5))

    frames_data = []
    for i, frame in enumerate(step.frames):
        s_field = frame.fieldOutputs['S']
        # NLGEOM=YES stores logarithmic strain as 'LE'
        if 'LE' in frame.fieldOutputs:
            e_field = frame.fieldOutputs['LE']
        else:
            e_field = frame.fieldOutputs['E']
        pe_field = frame.fieldOutputs['PE']
        peeq_field = frame.fieldOutputs['PEEQ']

        # Find values for center element
        s22 = None
        le22 = None
        pe22 = None
        peeq_val = None

        for val in s_field.values:
            if val.elementLabel == best_elem:
                s22 = val.data[1]  # S22 (axial, Y-direction)
                break

        for val in e_field.values:
            if val.elementLabel == best_elem:
                le22 = val.data[1]  # E22
                break

        for val in pe_field.values:
            if val.elementLabel == best_elem:
                pe22 = val.data[1]  # PE22
                break

        for val in peeq_field.values:
            if val.elementLabel == best_elem:
                if hasattr(val.data, '__len__'):
                    peeq_val = val.data[0]
                else:
                    peeq_val = val.data
                break

        if s22 is not None:
            frames_data.append({
                'frame': i,
                'time': frame.frameValue,
                'S22': s22,
                'E22': le22,
                'PE22': pe22,
                'PEEQ': peeq_val,
            })

    # Write stress-strain CSV
    with open('sim_stress_strain.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Time', 'S22', 'E22', 'PE22', 'PEEQ'])
        for d in frames_data:
            writer.writerow([d['frame'], d['time'],
                             d['S22'], d['E22'], d['PE22'], d['PEEQ']])

    print("  Saved: sim_stress_strain.csv (%d frames)" % len(frames_data))

    # ------------------------------------------------------------------
    # 2) Force-displacement from summing RF2 at top-edge nodes
    # ------------------------------------------------------------------
    # Top-edge nodes: y = 40 (LOAD_NODES)
    top_nodes = set()
    for node in instance.nodes:
        if abs(node.coordinates[1] - HALF_GAUGE) < 0.01:
            top_nodes.add(node.label)

    print("  Top-edge nodes: %d" % len(top_nodes))

    fd_data = []
    for i, frame in enumerate(step.frames):
        rf_field = frame.fieldOutputs['RF']
        u_field = frame.fieldOutputs['U']

        rf2_sum = 0.0
        u2_ref = None

        for val in rf_field.values:
            if val.nodeLabel in top_nodes:
                rf2_sum += val.data[1]  # RF2

        for val in u_field.values:
            if val.nodeLabel in top_nodes:
                u2_ref = val.data[1]  # U2 (same for all top nodes)
                break

        if u2_ref is not None:
            # Quarter model: multiply RF by 2 for full specimen force
            full_force = 2.0 * rf2_sum
            full_area = 2.0 * HALF_AREA  # 30 mm^2
            eng_stress = full_force / full_area
            eng_strain = u2_ref / HALF_GAUGE

            fd_data.append({
                'frame': i,
                'time': frame.frameValue,
                'U2': u2_ref,
                'RF2_quarter': rf2_sum,
                'RF2_full': full_force,
                'EngStress': eng_stress,
                'EngStrain': eng_strain,
            })

    with open('sim_force_disp.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Time', 'U2_mm', 'RF2_quarter_N',
                         'RF2_full_N', 'EngStress_MPa', 'EngStrain'])
        for d in fd_data:
            writer.writerow([d['frame'], d['time'], d['U2'],
                             d['RF2_quarter'], d['RF2_full'],
                             d['EngStress'], d['EngStrain']])

    print("  Saved: sim_force_disp.csv (%d frames)" % len(fd_data))

    odb.close()
    print("Done.")


if __name__ == '__main__':
    extract_results()
