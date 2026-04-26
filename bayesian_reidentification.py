#!/usr/bin/env python3
"""Bayesian re-identification of the Yld2000 + Voce + Chaboche(2) hardening
parameters using the experimental monotonic curves as likelihood and a
literature-derived prior on the backstress moduli.

This addresses the rank-deficiency of the monotonic cost surface shown by
`yld2000_param_sensitivity.py` (only sigma0 is identifiable; (C_i, gamma_i)
move the NRMSE by less than 1e-3). We break the degeneracy with a physically
motivated prior drawn from the published tension-compression / bend-unbend
calibrations of sheet-steel Chaboche models (see Thesis §8.7.5 table).

Literature priors (two-backstress AF-2, mild steel / DP600, see refs):
    C1/gamma1 ~ LogNormal(mu=log(100 MPa),  sigma=0.25)   # consensus 90-120 MPa
    C2/gamma2 ~ LogNormal(mu=log(120 MPa),  sigma=0.45)   # consensus 95-200 MPa
    gamma1    ~ LogNormal(mu=log(50),       sigma=0.55)   # 35-100 (fast)
    gamma2    ~ LogNormal(mu=log(8),        sigma=0.55)   # 5-12  (slow)
    sigma0    ~ Normal(mu=333 MPa, sigma=10 MPa)          # from FEA optimum

References (mild steel / AKDQ / SPCC / DP600 two-backstress fits):
  [1] Yoshida & Uemori, Int J Plast 18 (2002) 661
  [2] Shi et al., NUMISHEET 2008 (SPCC)
  [3] Eggertsen & Mattiasson, Int J Mech Sci 51 (2009) 547 (DP600)
  [4] Chun, Jinn, Lee, Int J Plast 18 (2002) 571 (AKDQ)
  [5] Lee, Kim, Wagoner, Int J Plast 23 (2007) 1189 (DDQ)
  [6] Vladimirov et al., J Mater Process Technol 209 (2009) 4062

Outputs:
  - output/bayesian_reid_posterior.npz      MCMC chain (nwalkers, nsteps, ndim)
  - output/bayesian_reid_summary.txt        Posterior statistics
  - output/fig_bayesian_reid_corner.png     Corner plot of the posterior
  - output/fig_bayesian_reid_posterior_curves.png  Post. predictive vs experiment
"""
from __future__ import annotations

import csv
import os

import corner
import emcee
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed / known constants (loaded from material_constants.json)
# ---------------------------------------------------------------------------
from material_constants import Q_INF, B_ISO, SIGMA_RATIO
# Common plastic-strain overlap used by the optimizer (0.2% -> 20% true strain)
EPS_GRID = np.linspace(0.002, 0.20, 120)

DIRS = {
    0:  ("stress-00-01.csv", "stress-00-02.csv", "stress-00-03.csv"),
    45: ("stress-45-01.csv", "stress-45-02.csv", "stress-45-03.csv"),
    90: ("stress-90-01.csv", "stress-90-02.csv", "stress-90-03.csv"),
}
# 45 receives 2x weight because it is the most anisotropy-sensitive direction
WEIGHTS = {0: 1.0, 45: 2.0, 90: 1.0}

# ---------------------------------------------------------------------------
# Experimental data
# ---------------------------------------------------------------------------

def load_exp_mean(angle):
    """Return (strain, mean_true_stress, std) on the common EPS_GRID."""
    curves = []
    for fname in DIRS[angle]:
        es, en = [], []
        with open(os.path.join(ROOT, fname), "r") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    es.append(float(row[2])); en.append(float(row[3]))
                except ValueError:
                    continue
        es = np.asarray(es); en = np.asarray(en)
        i_uts = int(np.argmax(es))
        es, en = es[: i_uts + 1], en[: i_uts + 1]
        m = (en > 0) & (es > 0)
        ts = es[m] * (1.0 + en[m])
        tr = np.log(1.0 + en[m])
        # convert to plastic strain approximately: tr - ts/E
        E = 200e3
        ep = tr - ts / E
        keep = ep > 0
        curves.append((ep[keep], ts[keep]))
    # interpolate onto common grid, truncate to shared support
    smin = max(c[0].min() for c in curves)
    smax = min(c[0].max() for c in curves)
    grid = EPS_GRID[(EPS_GRID >= smin) & (EPS_GRID <= smax)]
    stacked = np.array([np.interp(grid, *c) for c in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


EXP = {a: load_exp_mean(a) for a in (0, 45, 90)}
print("Experimental overlap points per direction:",
      {a: len(EXP[a][0]) for a in EXP})

# ---------------------------------------------------------------------------
# Forward model (analytical surrogate)
# ---------------------------------------------------------------------------

def flow_stress_rd(eps_p, sigma0, C1, gamma1, C2, gamma2):
    return (sigma0
            + Q_INF * (1.0 - np.exp(-B_ISO * eps_p))
            + (C1 / gamma1) * (1.0 - np.exp(-gamma1 * eps_p))
            + (C2 / gamma2) * (1.0 - np.exp(-gamma2 * eps_p)))


def predict(theta, angle):
    sigma0, logC1, logg1, logC2, logg2 = theta
    C1, gamma1 = np.exp(logC1), np.exp(logg1)
    C2, gamma2 = np.exp(logC2), np.exp(logg2)
    ep, _, _ = EXP[angle]
    s_rd = flow_stress_rd(ep, sigma0, C1, gamma1, C2, gamma2)
    return s_rd * SIGMA_RATIO[angle]


# ---------------------------------------------------------------------------
# Prior (literature-informed)
# ---------------------------------------------------------------------------
# Parametrisation: theta = (sigma0, log C1, log gamma1, log C2, log gamma2)
# Literature priors induce priors on (C_i, gamma_i).  We impose priors on
# log gamma_i directly and on log(C_i/gamma_i) (the saturation) by
# reparametrising when computing the prior log-density.

PRIOR = dict(
    sigma0_mu=333.0, sigma0_sd=15.0,
    sat1_mu=np.log(100.0), sat1_sd=0.25,   # log C1/gamma1, MPa
    sat2_mu=np.log(120.0), sat2_sd=0.45,
    g1_mu=np.log(50.0),    g1_sd=0.55,
    g2_mu=np.log(8.0),     g2_sd=0.55,
)
# Bounds (open-ended but finite to avoid pathological walkers)
BOUNDS = dict(sigma0=(150.0, 500.0), logC=(np.log(1.0), np.log(5e4)),
              logg=(np.log(0.5), np.log(2000.0)))


def log_prior(theta):
    sigma0, logC1, logg1, logC2, logg2 = theta
    if not BOUNDS["sigma0"][0] < sigma0 < BOUNDS["sigma0"][1]:
        return -np.inf
    for lC in (logC1, logC2):
        if not BOUNDS["logC"][0] < lC < BOUNDS["logC"][1]:
            return -np.inf
    for lg in (logg1, logg2):
        if not BOUNDS["logg"][0] < lg < BOUNDS["logg"][1]:
            return -np.inf
    # enforce ordering: gamma1 > gamma2 (fast > slow)
    if logg1 <= logg2:
        return -np.inf

    sat1 = logC1 - logg1
    sat2 = logC2 - logg2
    lp = 0.0
    lp += -0.5 * ((sigma0 - PRIOR["sigma0_mu"]) / PRIOR["sigma0_sd"]) ** 2
    lp += -0.5 * ((sat1 - PRIOR["sat1_mu"]) / PRIOR["sat1_sd"]) ** 2
    lp += -0.5 * ((sat2 - PRIOR["sat2_mu"]) / PRIOR["sat2_sd"]) ** 2
    lp += -0.5 * ((logg1 - PRIOR["g1_mu"]) / PRIOR["g1_sd"]) ** 2
    lp += -0.5 * ((logg2 - PRIOR["g2_mu"]) / PRIOR["g2_sd"]) ** 2
    return lp


# ---------------------------------------------------------------------------
# Likelihood: Gaussian residuals, per-direction sigma from experimental scatter
# with a floor to reflect model mismatch (≈10 MPa, consistent with FEA NRMSE)
# ---------------------------------------------------------------------------
SIGMA_FLOOR = 10.0  # MPa


def log_likelihood(theta):
    ll = 0.0
    for angle in (0, 45, 90):
        ep, smean, sstd = EXP[angle]
        spred = predict(theta, angle)
        sigma_eff = np.sqrt(sstd ** 2 + SIGMA_FLOOR ** 2)
        r = (spred - smean) / sigma_eff
        ll += WEIGHTS[angle] * (-0.5 * np.sum(r ** 2)
                                - np.sum(np.log(sigma_eff * np.sqrt(2 * np.pi))))
    return ll


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# ---------------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------------
# Initialise walkers near the literature prior means with a small jitter
NDIM = 5
NWALKERS = 64
NSTEPS = 4000
BURN = 1500

rng = np.random.default_rng(20260424)
p0_mean = np.array([
    PRIOR["sigma0_mu"],
    PRIOR["sat1_mu"] + PRIOR["g1_mu"],   # log C1 = log sat1 + log gamma1
    PRIOR["g1_mu"],
    PRIOR["sat2_mu"] + PRIOR["g2_mu"],
    PRIOR["g2_mu"],
])
p0_scale = np.array([5.0, 0.1, 0.1, 0.1, 0.1])
p0 = p0_mean + rng.normal(size=(NWALKERS, NDIM)) * p0_scale

print("Starting MCMC: nwalkers=%d  ndim=%d  nsteps=%d" % (NWALKERS, NDIM, NSTEPS))
sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_posterior)
sampler.run_mcmc(p0, NSTEPS, progress=False)
chain = sampler.get_chain(discard=BURN, flat=True)
print("Chain shape (flat):", chain.shape)
print("Mean acceptance fraction: %.3f" % np.mean(sampler.acceptance_fraction))
try:
    tau = sampler.get_autocorr_time(tol=0, discard=BURN)
    print("Integrated autocorr time:", tau)
except Exception as exc:
    print("Autocorr time not estimated:", exc)

np.savez(os.path.join(OUT, "bayesian_reid_posterior.npz"),
         chain=chain,
         log_prob=sampler.get_log_prob(discard=BURN, flat=True))

# ---------------------------------------------------------------------------
# Posterior summary in physical parameters
# ---------------------------------------------------------------------------
samp = np.empty((chain.shape[0], 7))
samp[:, 0] = chain[:, 0]              # sigma0
samp[:, 1] = np.exp(chain[:, 1])      # C1
samp[:, 2] = np.exp(chain[:, 2])      # gamma1
samp[:, 3] = np.exp(chain[:, 3])      # C2
samp[:, 4] = np.exp(chain[:, 4])      # gamma2
samp[:, 5] = np.exp(chain[:, 1] - chain[:, 2])  # C1/g1
samp[:, 6] = np.exp(chain[:, 3] - chain[:, 4])  # C2/g2

LABELS = [r"$\sigma_0$ [MPa]", r"$C_1$ [MPa]", r"$\gamma_1$",
          r"$C_2$ [MPa]", r"$\gamma_2$",
          r"$C_1/\gamma_1$ [MPa]", r"$C_2/\gamma_2$ [MPa]"]


def pctile(a):
    return np.percentile(a, [2.5, 16.0, 50.0, 84.0, 97.5])


lines = ["Bayesian re-identification of Yld2000+Voce+Chaboche(2) parameters",
         "=" * 72,
         "Literature prior:  C1/g1 ~ 100 MPa  C2/g2 ~ 120 MPa  "
         "g1 ~ 50  g2 ~ 8",
         "Likelihood       :  Gaussian, sigma_floor=%.1f MPa" % SIGMA_FLOOR,
         "Chain            :  %d walkers x %d post-burn samples" % (NWALKERS, NSTEPS - BURN),
         "Mean acceptance  :  %.3f" % np.mean(sampler.acceptance_fraction),
         "", "Posterior marginals (2.5 / 16 / 50 / 84 / 97.5 %):"]
for i, name in enumerate(LABELS):
    q = pctile(samp[:, i])
    lines.append("  %-22s %10.3f %10.3f %10.3f %10.3f %10.3f" %
                 (name, *q))

with open(os.path.join(OUT, "bayesian_reid_summary.txt"), "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))

# ---------------------------------------------------------------------------
# Corner plot (subset: sigma0, C1/g1, C2/g2, g1, g2)
# ---------------------------------------------------------------------------
idx = [0, 5, 6, 2, 4]
corner_labels = [LABELS[i] for i in idx]
fig = corner.corner(samp[:, idx], labels=corner_labels,
                    show_titles=True, title_fmt=".2f",
                    quantiles=[0.16, 0.5, 0.84], color="#1f77b4")
fig.savefig(os.path.join(OUT, "fig_bayesian_reid_corner.png"), dpi=160)
plt.close(fig)
print("Saved output/fig_bayesian_reid_corner.png")

# ---------------------------------------------------------------------------
# Posterior predictive curves (200 samples overlaid on experimental mean)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
n_draw = 200
idx_draw = rng.choice(chain.shape[0], size=n_draw, replace=False)
for ax, angle in zip(axes, (0, 45, 90)):
    ep, smean, sstd = EXP[angle]
    for k in idx_draw:
        theta = chain[k]
        ax.plot(ep * 100, predict(theta, angle), color="#1f77b4",
                lw=0.3, alpha=0.15)
    ax.plot(ep * 100, smean, color="k", lw=1.8, label="Exp. mean")
    ax.fill_between(ep * 100, smean - sstd, smean + sstd,
                    color="0.85", label=r"Exp. $\pm 1\sigma$")
    ax.set_title(r"$\theta = %d^\circ$" % angle)
    ax.set_xlabel(r"True plastic strain [%]")
    ax.grid(alpha=0.3)
    if angle == 0:
        ax.set_ylabel("True stress [MPa]")
    ax.legend(loc="lower right", fontsize=9)
fig.suptitle("Posterior predictive vs experiment (200 draws)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_bayesian_reid_posterior_curves.png"), dpi=160)
plt.close(fig)
print("Saved output/fig_bayesian_reid_posterior_curves.png")
