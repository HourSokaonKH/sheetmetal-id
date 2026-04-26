"""Analytical sensitivity / Monte-Carlo uncertainty on the FEA-identified
Yld2000 + Voce + Chaboche(2) parameter set.

Closed-form surrogate (no Abaqus): true uniaxial flow stress is
    sigma_y(eps_p) = sigma0 + Q_inf*(1 - exp(-b*eps_p))
                   + (C1/g1)*(1 - exp(-g1*eps_p))
                   + (C2/g2)*(1 - exp(-g2*eps_p))
The directional response is sigma_y(eps_p) * (sigma_theta/sigma_0_yld2000),
which is the same evaluation used by the optimiser cost function; the
ratios come from the calibrated Yld2000 surface (a=6).

Outputs:
  - output/yld2000_sensitivity_summary.txt
  - output/fig_yld2000_param_sensitivity.png  (tornado)
  - output/fig_yld2000_param_mc_distributions.png  (MC histograms)
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# Optimum (new FEA Yld2000 multi-direction identification)
OPT = {
    "sigma0": 332.889,
    "C1": 875.089,
    "gamma1": 380.641,
    "C2": 203.475,
    "gamma2": 145.414,
}
from material_constants import Q_INF, B_ISO, SIGMA_RATIO

# Reference experimental NRMSE denominators (sigma_max - sigma_min) per dir, MPa
# Values taken from the optimiser stress range over the 12-mm overlap.
REF_RANGE = {0: 245.0, 45: 207.0, 90: 273.0}

EPS = np.linspace(0.002, 0.20, 200)


def stress_curve(p, theta):
    s = (
        p["sigma0"]
        + Q_INF * (1.0 - np.exp(-B_ISO * EPS))
        + (p["C1"] / p["gamma1"]) * (1.0 - np.exp(-p["gamma1"] * EPS))
        + (p["C2"] / p["gamma2"]) * (1.0 - np.exp(-p["gamma2"] * EPS))
    )
    return s * SIGMA_RATIO[theta]


REF_CURVES = {t: stress_curve(OPT, t) for t in (0, 45, 90)}


def weighted_nrmse(p, weights=(1, 2, 1)):
    """Weighted NRMSE between perturbed curve and the optimum curve."""
    sw, w_sum = 0.0, sum(weights)
    for t, w in zip((0, 45, 90), weights):
        s = stress_curve(p, t)
        rmse = np.sqrt(np.mean((s - REF_CURVES[t]) ** 2))
        sw += w * rmse / REF_RANGE[t]
    return sw / w_sum


# --------------------------------------------------------------------------
# 1) One-at-a-time sensitivity (tornado) — ±10 % perturbation per parameter.
# --------------------------------------------------------------------------
PCT = 0.10
oat = {}
for k in OPT:
    p_lo, p_hi = dict(OPT), dict(OPT)
    p_lo[k] *= 1 - PCT
    p_hi[k] *= 1 + PCT
    oat[k] = (weighted_nrmse(p_lo), weighted_nrmse(p_hi))

oat_sorted = sorted(oat.items(), key=lambda kv: -max(kv[1]))

# --------------------------------------------------------------------------
# 2) Monte-Carlo: ±5 % Gaussian noise on every parameter, N = 5000 samples.
# --------------------------------------------------------------------------
rng = np.random.default_rng(20260423)
N = 5000
SIGMA_REL = 0.05
samples = {k: rng.normal(loc=OPT[k], scale=SIGMA_REL * abs(OPT[k]), size=N) for k in OPT}

nrmse_samples = np.empty(N)
peak_stress_0 = np.empty(N)
peak_stress_45 = np.empty(N)
peak_stress_90 = np.empty(N)

for i in range(N):
    p = {k: samples[k][i] for k in OPT}
    nrmse_samples[i] = weighted_nrmse(p)
    peak_stress_0[i] = stress_curve(p, 0)[-1]
    peak_stress_45[i] = stress_curve(p, 45)[-1]
    peak_stress_90[i] = stress_curve(p, 90)[-1]


def stats(arr):
    return dict(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        cov_pct=float(100.0 * np.std(arr) / abs(np.mean(arr))),
        ci_lo=float(np.percentile(arr, 2.5)),
        ci_hi=float(np.percentile(arr, 97.5)),
    )


summary = {
    "OAT_pct": PCT,
    "OAT_NRMSE": {k: {"low": v[0], "high": v[1]} for k, v in oat.items()},
    "MC_sigma_rel": SIGMA_REL,
    "MC_N": N,
    "MC_NRMSE": stats(nrmse_samples),
    "MC_peak_stress_0deg_MPa": stats(peak_stress_0),
    "MC_peak_stress_45deg_MPa": stats(peak_stress_45),
    "MC_peak_stress_90deg_MPa": stats(peak_stress_90),
}

# --------------------------------------------------------------------------
# Write text summary
# --------------------------------------------------------------------------
txt_lines = [
    "Yld2000 + Voce + Chaboche(2) parameter sensitivity (analytical surrogate)",
    "=" * 72,
    "Reference optimum:",
]
for k, v in OPT.items():
    txt_lines.append(f"    {k:8s} = {v:10.4f}")
txt_lines.append(f"    Q_inf    = {Q_INF:10.4f}  (fixed)")
txt_lines.append(f"    b        = {B_ISO:10.4f}  (fixed)")
txt_lines.append("")
txt_lines.append(f"One-at-a-time perturbation (+/-{PCT*100:.0f}%):")
txt_lines.append("    parameter        NRMSE(low)  NRMSE(high)  max")
for k, (lo, hi) in oat_sorted:
    txt_lines.append(f"    {k:12s}   {lo:10.5f}   {hi:10.5f}   {max(lo, hi):10.5f}")
txt_lines.append("")
txt_lines.append(
    f"Monte-Carlo (N={N}, Gaussian, sigma_rel={SIGMA_REL*100:.0f}% on every param):"
)
txt_lines.append(
    "    NRMSE             mean=%(mean).5f  std=%(std).5f  CoV=%(cov_pct).2f%%  "
    "95%% CI=[%(ci_lo).5f, %(ci_hi).5f]" % summary["MC_NRMSE"]
)
for theta, key in zip((0, 45, 90), ("MC_peak_stress_0deg_MPa", "MC_peak_stress_45deg_MPa", "MC_peak_stress_90deg_MPa")):
    s = summary[key]
    txt_lines.append(
        f"    sigma(eps=0.20) {theta:>2}deg   mean={s['mean']:7.2f} MPa  std={s['std']:6.2f}  "
        f"CoV={s['cov_pct']:5.2f}%   95%% CI=[{s['ci_lo']:6.1f}, {s['ci_hi']:6.1f}]"
    )

(OUT / "yld2000_sensitivity_summary.txt").write_text("\n".join(txt_lines))
(OUT / "yld2000_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
print("\n".join(txt_lines))

# --------------------------------------------------------------------------
# Tornado plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = [k for k, _ in oat_sorted]
lows = np.array([oat[k][0] for k in labels])
highs = np.array([oat[k][1] for k in labels])
center = 0.0  # NRMSE relative to optimum which is 0 (perfect self-fit)
y = np.arange(len(labels))
ax.barh(y, highs, height=0.4, label=f"+{PCT*100:.0f}%", color="#d62728", align="edge")
ax.barh(y - 0.4, lows, height=0.4, label=f"-{PCT*100:.0f}%", color="#1f77b4", align="edge")
ax.set_yticks(y - 0.2)
ax.set_yticklabels([{"sigma0": r"$\sigma_0$", "C1": r"$C_1$", "gamma1": r"$\gamma_1$",
                     "C2": r"$C_2$", "gamma2": r"$\gamma_2$"}[k] for k in labels])
ax.set_xlabel("Weighted NRMSE w.r.t. optimum (45°: 2× weight)")
ax.set_title(f"One-at-a-time sensitivity (±{int(PCT*100)}% perturbation)")
ax.axvline(0, color="k", lw=0.6)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_yld2000_param_sensitivity.png", dpi=180)
plt.close(fig)
print(f"Saved {OUT / 'fig_yld2000_param_sensitivity.png'}")

# --------------------------------------------------------------------------
# MC distribution plot — NRMSE + 3 peak stresses
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
data = [
    (nrmse_samples, "Weighted NRMSE", axes[0, 0]),
    (peak_stress_0, r"$\sigma(\varepsilon=0.20)$ at 0° [MPa]", axes[0, 1]),
    (peak_stress_45, r"$\sigma(\varepsilon=0.20)$ at 45° [MPa]", axes[1, 0]),
    (peak_stress_90, r"$\sigma(\varepsilon=0.20)$ at 90° [MPa]", axes[1, 1]),
]
for arr, title, ax in data:
    ax.hist(arr, bins=50, color="#4c72b0", alpha=0.85, edgecolor="white")
    m, lo, hi = np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)
    ax.axvline(m, color="red", lw=1.4, label=f"mean={m:.3f}")
    ax.axvline(lo, color="orange", lw=1.0, ls="--", label=f"95% CI [{lo:.3f}, {hi:.3f}]")
    ax.axvline(hi, color="orange", lw=1.0, ls="--")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
fig.suptitle(
    f"Monte-Carlo on Yld2000+Chaboche optimum (N={N}, ±{int(SIGMA_REL*100)}% Gaussian per param)"
)
fig.tight_layout()
fig.savefig(OUT / "fig_yld2000_param_mc_distributions.png", dpi=180)
plt.close(fig)
print(f"Saved {OUT / 'fig_yld2000_param_mc_distributions.png'}")
