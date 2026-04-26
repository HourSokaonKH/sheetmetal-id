"""
=============================================================================
Extract Abaqus Contour Data — Full Field Output for Plotting
=============================================================================
Run on Abaqus machine:
    abaqus python extract_contour_data.py

Extracts from Tensile_CKH.odb (or Tensile_mesh_1p0.odb):
  - Element centroids (x, y)
  - S_Mises, PEEQ, S22, LE22 at the final frame
  - Node displacements for deformed shape

Outputs:
  contour_field_data.csv   — per-element field values at final frame
  contour_node_disp.csv    — node coordinates + displacements at final frame

Transfer CSVs to local machine, then run: python3 generate_fig_contours.py
=============================================================================
"""

from odbAccess import openOdb
import csv
import os
import math

# Try multiple possible ODB names
ODB_CANDIDATES = ['Tensile_mesh_1p0.odb', 'Tensile_CKH.odb']
STEP_NAME = 'Tensile'
INSTANCE_NAME = 'SPECIMEN-1'


def find_odb():
    for name in ODB_CANDIDATES:
        if os.path.exists(name):
            return name
    print("ERROR: No ODB file found. Tried: %s" % ', '.join(ODB_CANDIDATES))
    return None


def main():
    odb_name = find_odb()
    if odb_name is None:
        return

    print("Opening %s ..." % odb_name)
    odb = openOdb(path=odb_name, readOnly=True)
    step = odb.steps[STEP_NAME]
    instance = odb.rootAssembly.instances[INSTANCE_NAME]

    # Build node coordinate map
    node_map = {}
    for node in instance.nodes:
        node_map[node.label] = (node.coordinates[0], node.coordinates[1])

    # Get final frame
    frame = step.frames[-1]
    print("  Final frame: time = %.4f" % frame.frameValue)

    s_field = frame.fieldOutputs['S']
    peeq_field = frame.fieldOutputs['PEEQ']
    if 'LE' in frame.fieldOutputs:
        e_field = frame.fieldOutputs['LE']
    else:
        e_field = frame.fieldOutputs['E']
    u_field = frame.fieldOutputs['U']

    # ── Extract element field data ──────────────────────────────────────
    # Build element data: centroid + field values
    elem_data = []
    for elem in instance.elements:
        conn = elem.connectivity
        cx = sum(node_map[n][0] for n in conn) / len(conn)
        cy = sum(node_map[n][1] for n in conn) / len(conn)
        elem_data.append({
            'label': elem.label,
            'cx': cx,
            'cy': cy,
        })

    # Map element labels to indices
    elem_idx = {ed['label']: i for i, ed in enumerate(elem_data)}

    # Extract stress values
    for val in s_field.values:
        if val.elementLabel in elem_idx:
            i = elem_idx[val.elementLabel]
            elem_data[i]['S22'] = val.data[1]    # S22
            elem_data[i]['S_Mises'] = val.mises if hasattr(val, 'mises') else 0.0

    # Extract PEEQ
    for val in peeq_field.values:
        if val.elementLabel in elem_idx:
            i = elem_idx[val.elementLabel]
            if hasattr(val.data, '__len__'):
                elem_data[i]['PEEQ'] = val.data[0]
            else:
                elem_data[i]['PEEQ'] = val.data

    # Extract strain
    for val in e_field.values:
        if val.elementLabel in elem_idx:
            i = elem_idx[val.elementLabel]
            elem_data[i]['LE22'] = val.data[1]

    # Write element field CSV
    with open('contour_field_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['elem_label', 'cx', 'cy', 'S_Mises', 'S22', 'PEEQ', 'LE22'])
        for ed in elem_data:
            writer.writerow([
                ed['label'],
                '%.6f' % ed['cx'],
                '%.6f' % ed['cy'],
                '%.4f' % ed.get('S_Mises', 0),
                '%.4f' % ed.get('S22', 0),
                '%.6f' % ed.get('PEEQ', 0),
                '%.6f' % ed.get('LE22', 0),
            ])

    print("  Saved: contour_field_data.csv (%d elements)" % len(elem_data))

    # ── Extract node displacements ──────────────────────────────────────
    node_disp = []
    for val in u_field.values:
        nl = val.nodeLabel
        if nl in node_map:
            x0, y0 = node_map[nl]
            u1 = val.data[0]
            u2 = val.data[1]
            node_disp.append({
                'label': nl,
                'x0': x0, 'y0': y0,
                'u1': u1, 'u2': u2,
                'x_def': x0 + u1,
                'y_def': y0 + u2,
            })

    with open('contour_node_disp.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['node_label', 'x0', 'y0', 'u1', 'u2', 'x_def', 'y_def'])
        for nd in node_disp:
            writer.writerow([
                nd['label'],
                '%.6f' % nd['x0'],
                '%.6f' % nd['y0'],
                '%.6f' % nd['u1'],
                '%.6f' % nd['u2'],
                '%.6f' % nd['x_def'],
                '%.6f' % nd['y_def'],
            ])

    print("  Saved: contour_node_disp.csv (%d nodes)" % len(node_disp))

    # ── Print summary ───────────────────────────────────────────────────
    s_mises_vals = [ed.get('S_Mises', 0) for ed in elem_data]
    peeq_vals = [ed.get('PEEQ', 0) for ed in elem_data]
    s22_vals = [ed.get('S22', 0) for ed in elem_data]

    print("\n  Field data summary (final frame):")
    print("    S_Mises: min=%.2f, max=%.2f, mean=%.2f MPa" % (
        min(s_mises_vals), max(s_mises_vals),
        sum(s_mises_vals) / len(s_mises_vals)))
    print("    S22:     min=%.2f, max=%.2f, mean=%.2f MPa" % (
        min(s22_vals), max(s22_vals), sum(s22_vals) / len(s22_vals)))
    print("    PEEQ:    min=%.4f, max=%.4f, mean=%.4f" % (
        min(peeq_vals), max(peeq_vals), sum(peeq_vals) / len(peeq_vals)))

    odb.close()
    print("\nTransfer contour_field_data.csv and contour_node_disp.csv")
    print("to local machine, then run: python3 generate_fig_contours.py")


if __name__ == '__main__':
    main()
