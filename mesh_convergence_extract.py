"""
=============================================================================
Mesh Convergence Study — Extract ODB Results (runs on Abaqus machine)
=============================================================================
Run AFTER all 3 simulations have completed:
    abaqus python mesh_convergence_extract.py

Extracts from each ODB:
  - S22 vs LE22 at center element (stress-strain)
  - Total RF2 sum at top edge (force-displacement)
  - Maximum PEEQ, von Mises at final frame
  - Wall-clock time (from .sta file)

Outputs:
  mesh_convergence_results.csv  — summary metrics per mesh size
  mesh_0p5_stress_strain.csv    — full S22/LE22 curves
  mesh_1p0_stress_strain.csv
  mesh_2p0_stress_strain.csv
=============================================================================
"""

from odbAccess import openOdb
import csv
import os
import math

MESH_SIZES = [0.5, 1.0, 2.0]
STEP_NAME = 'Tensile'
INSTANCE_NAME = 'SPECIMEN-1'
HALF_WIDTH = 10.0
LENGTH = 40.0
THICKNESS = 1.5


def get_job_name(mesh_size):
    tag = ("%.1f" % mesh_size).replace('.', 'p')
    return "Tensile_mesh_%s" % tag


def extract_one(mesh_size):
    """Extract results from one ODB. Returns dict of metrics + stress-strain list."""
    job = get_job_name(mesh_size)
    odb_path = "%s.odb" % job
    if not os.path.exists(odb_path):
        print("  WARNING: %s not found, skipping" % odb_path)
        return None, None

    print("  Opening %s ..." % odb_path)
    odb = openOdb(path=odb_path, readOnly=True)
    step = odb.steps[STEP_NAME]
    instance = odb.rootAssembly.instances[INSTANCE_NAME]

    n_nodes = len(instance.nodes)
    n_elems = len(instance.elements)
    n_frames = len(step.frames)

    # Build node coordinate map
    node_map = {}
    for node in instance.nodes:
        node_map[node.label] = node.coordinates

    # Find center element (closest to x=5, y=20)
    best_elem = None
    best_dist = 1e20
    for elem in instance.elements:
        cx = sum(node_map[n][0] for n in elem.connectivity) / len(elem.connectivity)
        cy = sum(node_map[n][1] for n in elem.connectivity) / len(elem.connectivity)
        dist = (cx - 5.0)**2 + (cy - 20.0)**2
        if dist < best_dist:
            best_dist = dist
            best_elem = elem.label

    print("    Center element: %d (dist=%.4f mm)" % (best_elem, math.sqrt(best_dist)))

    # Find top-edge nodes (y = LENGTH)
    top_nodes = set()
    for node in instance.nodes:
        if abs(node.coordinates[1] - LENGTH) < 1e-6:
            top_nodes.add(node.label)

    # Extract stress-strain at center element + force-disp at top edge
    ss_data = []  # stress-strain
    for i, frame in enumerate(step.frames):
        s_field = frame.fieldOutputs['S']
        if 'LE' in frame.fieldOutputs:
            e_field = frame.fieldOutputs['LE']
        else:
            e_field = frame.fieldOutputs['E']
        peeq_field = frame.fieldOutputs['PEEQ']
        rf_field = frame.fieldOutputs['RF']
        u_field = frame.fieldOutputs['U']

        # Center element stress/strain
        s22 = None
        le22 = None
        for val in s_field.values:
            if val.elementLabel == best_elem:
                s22 = val.data[1]
                break
        for val in e_field.values:
            if val.elementLabel == best_elem:
                le22 = val.data[1]
                break

        # Sum RF2 at top edge
        rf2_sum = 0.0
        u2_top = 0.0
        n_top_found = 0
        for val in rf_field.values:
            if val.nodeLabel in top_nodes:
                rf2_sum += val.data[1]
        for val in u_field.values:
            if val.nodeLabel in top_nodes:
                u2_top += val.data[1]
                n_top_found += 1
        if n_top_found > 0:
            u2_top /= n_top_found

        if s22 is not None:
            ss_data.append({
                'frame': i,
                'time': frame.frameValue,
                'S22': s22,
                'LE22': le22,
                'RF2_sum': rf2_sum,
                'U2_top': u2_top,
            })

    # Final frame metrics
    last = step.frames[-1]
    peeq_field = last.fieldOutputs['PEEQ']
    s_field = last.fieldOutputs['S']

    max_peeq = 0.0
    max_mises = 0.0
    for val in peeq_field.values:
        if hasattr(val.data, '__len__'):
            v = val.data[0]
        else:
            v = val.data
        if v > max_peeq:
            max_peeq = v
    for val in s_field.values:
        if hasattr(val, 'mises') and val.mises > max_mises:
            max_mises = val.mises

    # Final stress at center
    final_s22 = ss_data[-1]['S22'] if ss_data else 0.0
    final_le22 = ss_data[-1]['LE22'] if ss_data else 0.0

    # Read wall-clock time from .sta file if available
    wall_time = None
    sta_path = "%s.sta" % job
    if os.path.exists(sta_path):
        try:
            with open(sta_path, 'r') as sf:
                lines = sf.readlines()
            for line in reversed(lines):
                if 'TOTAL' in line.upper() or 'WALLCLOCK' in line.upper():
                    parts = line.split()
                    for p in parts:
                        try:
                            wall_time = float(p)
                        except ValueError:
                            pass
                    break
        except Exception:
            pass

    odb.close()

    metrics = {
        'mesh_size': mesh_size,
        'n_nodes': n_nodes,
        'n_elements': n_elems,
        'n_frames': n_frames,
        'final_S22': final_s22,
        'final_LE22': final_le22,
        'max_PEEQ': max_peeq,
        'max_Mises': max_mises,
        'wall_time': wall_time,
    }

    return metrics, ss_data


def main():
    print("=" * 60)
    print("Mesh Convergence Study - Extracting ODB Results")
    print("=" * 60)

    all_metrics = []
    for ms in MESH_SIZES:
        print("\nMesh size: %.1f mm" % ms)
        metrics, ss_data = extract_one(ms)
        if metrics is None:
            continue
        all_metrics.append(metrics)

        # Write per-mesh stress-strain CSV
        tag = ("%.1f" % ms).replace('.', 'p')
        ss_file = "mesh_%s_stress_strain.csv" % tag
        with open(ss_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Time', 'S22', 'LE22', 'RF2_sum', 'U2_top'])
            for d in ss_data:
                writer.writerow([d['frame'], d['time'], d['S22'],
                                 d['LE22'], d['RF2_sum'], d['U2_top']])
        print("    Saved: %s (%d frames)" % (ss_file, len(ss_data)))
        print("    Final S22=%.2f MPa, LE22=%.6f" % (metrics['final_S22'],
                                                       metrics['final_LE22']))

    # Write summary CSV
    if all_metrics:
        with open('mesh_convergence_results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['mesh_size', 'n_nodes', 'n_elements', 'n_frames',
                             'final_S22', 'final_LE22', 'max_PEEQ', 'max_Mises',
                             'wall_time'])
            for m in all_metrics:
                writer.writerow([m['mesh_size'], m['n_nodes'], m['n_elements'],
                                 m['n_frames'], m['final_S22'], m['final_LE22'],
                                 m['max_PEEQ'], m['max_Mises'], m['wall_time']])
        print("\nSaved: mesh_convergence_results.csv")

    # Print summary table
    print("\n" + "=" * 80)
    print("MESH CONVERGENCE SUMMARY")
    print("=" * 80)
    print("%-10s %8s %8s %10s %10s %10s" % (
        'Mesh(mm)', 'Nodes', 'Elems', 'S22_final', 'LE22_final', 'max_PEEQ'))
    print("-" * 80)
    for m in all_metrics:
        print("%-10.1f %8d %8d %10.2f %10.6f %10.6f" % (
            m['mesh_size'], m['n_nodes'], m['n_elements'],
            m['final_S22'], m['final_LE22'], m['max_PEEQ']))

    if len(all_metrics) >= 2:
        # Richardson extrapolation estimate (between two finest meshes)
        s1 = all_metrics[0]['final_S22']  # finest
        s2 = all_metrics[1]['final_S22']  # medium
        h1 = all_metrics[0]['mesh_size']
        h2 = all_metrics[1]['mesh_size']
        r = h2 / h1
        if abs(s1 - s2) > 1e-10:
            p_est = abs(math.log(abs((all_metrics[2]['final_S22'] - s2) /
                                      (s2 - s1)))) / math.log(r) if len(all_metrics) >= 3 else 2.0
            s_exact = s1 + (s1 - s2) / (r**p_est - 1)
            print("\nRichardson extrapolation (p=%.1f): S22_exact ~ %.2f MPa" % (
                p_est, s_exact))
            for m in all_metrics:
                err = abs(m['final_S22'] - s_exact) / abs(s_exact) * 100
                print("  mesh=%.1f: relative error = %.3f%%" % (m['mesh_size'], err))

    print("\nTransfer mesh_*_stress_strain.csv and mesh_convergence_results.csv")
    print("to your local machine, then run: python3 mesh_convergence_plot.py")


if __name__ == '__main__':
    main()
