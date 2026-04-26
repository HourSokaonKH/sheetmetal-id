"""
Run on the lab PC under: abaqus python extract_yld2000_table_best.py

Reads mopt_yt_best_{00,45,90}.odb, extracts global RF2/U2 -> true stress/strain,
writes sim_best_{00,45,90}.csv into optimization_results/.

Mirrors the extraction logic in optimize_hardening_multidir.py.
"""
from __future__ import print_function
import os
import math
import csv

try:
    from odbAccess import openOdb
except ImportError:
    raise SystemExit('Run with: abaqus python extract_yld2000_table_best.py')

# Geometry constants (must match optimize_hardening_multidir.py)
HALF_WIDTH  = 10.0   # mm  (full gauge width = 20 mm)
HALF_LENGTH = 40.0   # mm  (full gauge length = 80 mm)
THICKNESS   = 1.5    # mm

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'optimization_results')
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

A0 = 2.0 * HALF_WIDTH * THICKNESS
L0 = HALF_LENGTH


def extract(odb_path, out_csv):
    odb = openOdb(odb_path, readOnly=True)
    step = odb.steps[odb.steps.keys()[-1]]
    instance = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]

    load_labels = set([n.label for n in instance.nodeSets['LOAD'].nodes])
    corner_label = None
    if 'CORNER' in instance.nodeSets.keys():
        corner_label = instance.nodeSets['CORNER'].nodes[0].label

    rows = [['true_strain', 'true_stress_MPa']]
    for frame in step.frames:
        rf = frame.fieldOutputs['RF']
        u  = frame.fieldOutputs['U']
        rf2 = 0.0
        for v in rf.values:
            if v.nodeLabel in load_labels:
                rf2 += v.data[1]
        u2 = None
        if corner_label is not None:
            for v in u.values:
                if v.nodeLabel == corner_label:
                    u2 = v.data[1]
                    break
        if u2 is None:
            continue
        eng_stress = (2.0 * rf2) / A0
        eng_strain = u2 / L0
        if eng_strain <= 0.0 or eng_stress <= 0.0:
            continue
        ts = math.log(1.0 + eng_strain)
        ss = eng_stress * (1.0 + eng_strain)
        rows.append([ts, ss])
    odb.close()

    with open(out_csv, 'wb' if str is bytes else 'w') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print('  wrote %s (%d rows)' % (out_csv, len(rows) - 1))


for ang in (0, 45, 90):
    odb = 'mopt_yt_best_%02d.odb' % ang
    out = os.path.join(OUT_DIR, 'sim_best_%02d.csv' % ang)
    if not os.path.exists(odb):
        print('SKIP: %s not found' % odb)
        continue
    print('Extracting %s ...' % odb)
    extract(odb, out)

print('\nDone. Copy optimization_results/sim_best_*.csv back to the Mac.')
