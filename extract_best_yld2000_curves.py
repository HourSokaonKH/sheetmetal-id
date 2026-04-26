"""
Extract true stress-strain curves from the three best Yld2000 multi-direction
ODBs produced by optimize_hardening_multidir.py.

Run on the Abaqus machine:
    abaqus python extract_best_yld2000_curves.py

Reads :  mopt_y_best_{00,45,90}.odb
Writes:  output/yld2000_best_{00,45,90}_true_curve.csv

Reuses the same RF2/U2 -> true-stress conversion as
optimize_hardening_multidir.py:extract_results_odb so the curves are
identical to what the optimizer NRMSE was computed on.
"""
import os
import math
import csv

from odbAccess import openOdb

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
if not os.path.isdir(OUT):
    os.makedirs(OUT)

HALF_WIDTH = 10.0
HALF_LENGTH = 40.0
THICKNESS = 1.5
A0 = 2.0 * HALF_WIDTH * THICKNESS
L0 = HALF_LENGTH


def extract(job):
    odb_path = os.path.join(ROOT, "%s.odb" % job)
    if not os.path.exists(odb_path):
        print("MISSING: %s" % odb_path)
        return None
    odb = openOdb(path=odb_path, readOnly=True)
    step = odb.steps["Tensile"]
    inst = odb.rootAssembly.instances["SPEC-1"]

    load_labels = set()
    corner_label = None
    for n in inst.nodes:
        if abs(n.coordinates[1] - HALF_LENGTH) < 1e-6:
            load_labels.add(n.label)
            if abs(n.coordinates[0]) < 1e-6:
                corner_label = n.label

    rows = []
    for f in step.frames:
        rf = f.fieldOutputs["RF"]
        u = f.fieldOutputs["U"]
        rf2_q = 0.0
        for v in rf.values:
            if v.nodeLabel in load_labels:
                rf2_q += v.data[1]
        u2 = None
        for v in u.values:
            if v.nodeLabel == corner_label:
                u2 = v.data[1]
                break
        if u2 is None or u2 <= 0:
            continue
        eng_stress = (2.0 * rf2_q) / A0
        eng_strain = u2 / L0
        if eng_stress <= 0:
            continue
        rows.append((math.log(1.0 + eng_strain),
                     eng_stress * (1.0 + eng_strain)))
    odb.close()
    return rows


def main():
    for angle in (0, 45, 90):
        job = "mopt_y_best_%02d" % angle
        rows = extract(job)
        if not rows:
            print("  No data for %s" % job)
            continue
        out = os.path.join(OUT, "yld2000_best_%02d_true_curve.csv" % angle)
        with open(out, "w") as f:
            w = csv.writer(f)
            w.writerow(["true_strain", "true_stress_mpa"])
            for s, t in rows:
                w.writerow(["%.10f" % s, "%.10f" % t])
        print("Wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
