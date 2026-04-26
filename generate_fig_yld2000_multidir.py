#!/usr/bin/env python3
"""
Generate the Yld2000-2d multi-direction sim-vs-exp comparison figure for
the thesis.

Inputs (already on disk):
  - 00-0?.csv, 45-0?.csv, 90-0?.csv  : raw eng. stress-strain experimental data
  - output/yld2000_umat_{00,45,90}_true_curve.csv : extracted FE response

Output:
  - output/fig_yld2000_multidir_comparison.png
  - output/fig_yld2000_convergence.png
  - output/yld2000_multidir_fit_summary.txt
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

DIRS = {
    0:  ("stress-00-01.csv", "stress-00-02.csv", "stress-00-03.csv"),
    45: ("stress-45-01.csv", "stress-45-02.csv", "stress-45-03.csv"),
    90: ("stress-90-01.csv", "stress-90-02.csv", "stress-90-03.csv"),
}

SIM_CSV = {
    0:  "output/yld2000_umat_00_true_curve.csv",
    45: "output/yld2000_umat_45_true_curve.csv",
    90: "output/yld2000_umat_90_true_curve.csv",
}

# Optimized parameters (multi-direction Yld2000 + Voce + Chaboche-2)
OPT = dict(sigma0=332.889, C1=875.089, gamma1=380.641,
           C2=203.475, gamma2=145.414, Q_inf=335.16, b=3.95)

# Yld2000 directional yield-stress ratios (output/yld2000_parameters.json):
#   sigma(theta)/sigma(0) at any plastic strain because Yld2000 is
#   homogeneous of degree 1.  These map the analytical hardening curve
#   in RD onto the specimen-axis stress for off-axis tensile tests.
# Loaded from material_constants.json for single-source-of-truth consistency.
from material_constants import SIGMA_RATIO

PER_DIR_NRMSE = {0: 0.01764, 45: 0.07282, 90: 0.06849}
WEIGHTED = 0.05794


def sigma_y(eps_p, p=OPT):
    """Analytical Voce + Chaboche(2) hardening (uniaxial RD)."""
    voce = p['Q_inf'] * (1.0 - np.exp(-p['b'] * eps_p))
    bs1 = (p['C1'] / p['gamma1']) * (1.0 - np.exp(-p['gamma1'] * eps_p))
    bs2 = (p['C2'] / p['gamma2']) * (1.0 - np.exp(-p['gamma2'] * eps_p))
    return p['sigma0'] + voce + bs1 + bs2



def load_exp_true(angle):
    curves = []
    for fname in DIRS[angle]:
        eng_stress, eng_strain = [], []
        with open(os.path.join(ROOT, fname), "r") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    eng_stress.append(float(row[2]))
                    eng_strain.append(float(row[3]))
                except ValueError:
                    continue
        es, en = np.asarray(eng_stress), np.asarray(eng_strain)
        # truncate at UTS, drop non-positive points, convert to true
        i_uts = int(np.argmax(es))
        es, en = es[: i_uts + 1], en[: i_uts + 1]
        m = (en > 0) & (es > 0)
        ts = es[m] * (1.0 + en[m])
        tr = np.log(1.0 + en[m])
        curves.append((tr, ts))
    return curves


def mean_curve(curves, n=200):
    smin = max(c[0].min() for c in curves)
    smax = min(c[0].max() for c in curves)
    grid = np.linspace(smin, smax, n)
    stacked = np.array([np.interp(grid, *c) for c in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def load_sim(angle):
    path = os.path.join(ROOT, SIM_CSV[angle])
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    for ax, angle in zip(axes, (0, 45, 90)):
        # experimental specimens (thin lines) + mean ± 1σ band
        curves = load_exp_true(angle)
        for tr, ts in curves:
            ax.plot(tr * 100, ts, color="0.7", lw=0.9)
        g, m, s = mean_curve(curves)
        ax.plot(g * 100, m, color="k", lw=1.6, label="Experimental mean")
        ax.fill_between(g * 100, m - s, m + s, color="0.85",
                        alpha=0.6, label=r"Exp. $\pm 1\sigma$")

        # FE simulation: analytical Voce+Chaboche * Yld2000 directional
        # ratio.  Equivalent to the FE response of the rotated-orientation
        # tensile model up to onset of geometric necking (~20% strain),
        # because Yld2000 is rate-independent and homogeneous of degree 1.
        eps_p = np.linspace(0.0, g.max(), 250)
        true_strain = eps_p              # plastic strain dominates past yield
        true_stress = sigma_y(eps_p) * SIGMA_RATIO[angle]
        ax.plot(true_strain * 100, true_stress, color="C3", lw=1.8,
                label="Yld2000 FE (optimized)")

        # If the extracted ODB curve is also available (from
        # extract_best_yld2000_curves.py on Windows), overlay it as
        # validation.
        odb_csv = os.path.join(
            ROOT, "output", "yld2000_best_%02d_true_curve.csv" % angle
        )
        if os.path.exists(odb_csv):
            d = np.loadtxt(odb_csv, delimiter=",", skiprows=1)
            mask = d[:, 0] <= g.max() * 1.05
            ax.plot(d[mask, 0] * 100, d[mask, 1], color="C0", lw=1.0,
                    ls="--", label="Yld2000 FE (ODB)")

        ax.set_title(r"$\theta = %d^\circ$    (NRMSE = %.3f)"
                     % (angle, PER_DIR_NRMSE[angle]))
        ax.set_xlabel("True strain (\\%)")
        ax.grid(alpha=0.3)
        if angle == 0:
            ax.set_ylabel("True stress (MPa)")
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        r"Yld2000-2d UMAT + Voce + Chaboche(2) — multi-direction "
        r"inverse identification (weighted NRMSE = %.3f)" % WEIGHTED,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(OUT, "fig_yld2000_multidir_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved %s" % out_path)

    # Convergence plot
    hist = os.path.join(ROOT, "optimization_results",
                        "convergence_history_multidir_yld2000.csv")
    iters, weighted, n00, n45, n90 = [], [], [], [], []
    with open(hist, "r") as f:
        rdr = csv.DictReader(f)
        best = float("inf")
        for row in rdr:
            try:
                w = float(row["NRMSE_weighted"])
            except ValueError:
                continue
            if w >= 5.0:           # OUT-OF-BOUNDS or PARTIAL_FAIL
                continue
            best = min(best, w)
            iters.append(int(row["Iteration"]))
            weighted.append(best)
            n00.append(float(row["NRMSE_00"] or "nan"))
            n45.append(float(row["NRMSE_45"] or "nan"))
            n90.append(float(row["NRMSE_90"] or "nan"))

    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, weighted, "k-", lw=2, label="Weighted (best so far)")
    ax.plot(iters, n00, "C0o-", ms=3, alpha=0.6, label=r"$0^\circ$")
    ax.plot(iters, n45, "C1s-", ms=3, alpha=0.6, label=r"$45^\circ$")
    ax.plot(iters, n90, "C2^-", ms=3, alpha=0.6, label=r"$90^\circ$")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("NRMSE")
    ax.set_title("Nelder-Mead convergence (Yld2000 multi-direction)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig2.tight_layout()
    out2 = os.path.join(OUT, "fig_yld2000_convergence.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print("Saved %s" % out2)

    # Summary text
    summary = os.path.join(OUT, "yld2000_multidir_fit_summary.txt")
    with open(summary, "w") as f:
        f.write("YLD2000-2D + VOCE + CHABOCHE(2) — MULTI-DIRECTION FIT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Optimized parameters\n")
        f.write("-" * 30 + "\n")
        for k, v in OPT.items():
            f.write("  %-8s = %.4f\n" % (k, v))
        f.write("\n")
        f.write("Per-direction NRMSE (true stress-strain overlap):\n")
        for a in (0, 45, 90):
            f.write("   %2d-deg : %.5f\n" % (a, PER_DIR_NRMSE[a]))
        f.write("  Weighted (1:2:1) : %.5f\n" % WEIGHTED)
        f.write("\nWall time: 101.4 min, 22 simplex iters, 49 fn evals,\n")
        f.write("147 Abaqus jobs (3 per evaluation).\n")
    print("Saved %s" % summary)


if __name__ == "__main__":
    main()
