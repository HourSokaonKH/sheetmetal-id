# 7. Sensitivity analysis & Bayesian re-identification

Reporting a single optimum (σ₀, C₁, γ₁, C₂, γ₂) is incomplete without
**uncertainty bands**. Two complementary tools are provided.

## 7.1 Monte-Carlo sensitivity (frequentist)

```bash
python sensitivity_analysis.py
```

Propagates measurement uncertainty through the **entire** identification
pipeline using N = 1000 Monte-Carlo samples:

| Source of uncertainty           | Distribution                  |
| ------------------------------- | ----------------------------- |
| DIC ε_yy noise                  | ±5 % (Gaussian)               |
| Yield stress 3-replicate scatter| Gaussian from sample s.d.     |
| Young's modulus E               | 200 ± 5 GPa (Gaussian)        |

For each sample the script perturbs the inputs, re-runs the analytical
fit, and stores the identified parameters. A 95 % confidence interval is
quoted in the printout and visualised as shaded bands in
`output/fig_combined_hardening_components.png`.

## 7.2 Bayesian re-identification (informative prior)

```bash
python bayesian_reidentification.py
```

The monotonic uniaxial cost surface is rank-deficient — only σ₀ is
truly identifiable from a single tensile test. To break the degeneracy,
this script puts a **literature-informed prior** on the saturated
backstress moduli:

| Parameter   | Prior                                     | Source                       |
| ----------- | ----------------------------------------- | ---------------------------- |
| C₁/γ₁       | LogN(log 100 MPa, 0.25)                   | Yoshida-Uemori, Shi          |
| C₂/γ₂       | LogN(log 120 MPa, 0.45)                   | Eggertsen-Mattiasson         |
| γ₁          | LogN(log 50, 0.55)                        | mild-steel 2-backstress fits |
| γ₂          | LogN(log 8, 0.55)                         | "                            |
| σ₀          | N(333 MPa, 10 MPa)                        | from FEA optimum (this work) |

It runs an **affine-invariant ensemble sampler** (`emcee`) and produces:

| File                                    | Contents                              |
| --------------------------------------- | ------------------------------------- |
| `output/bayesian_reid_posterior.npz`    | full chain `(nwalkers, nsteps, ndim)` |
| `output/bayesian_reid_summary.txt`      | posterior median ± 68 % HDI           |
| `output/fig_bayesian_reid_corner.png`   | corner plot                           |
| `output/fig_bayesian_reid_posterior_curves.png` | predictive bands              |

The corner plot makes the rank-deficiency visible: the posterior is tight
on σ₀ and C/γ ratios but unbounded on C and γ separately.

## 7.3 What to report

Recommended publication-quality reporting:

* **σ₀** — quote median ± half-95% HDI from the Bayesian posterior.
* **C₁/γ₁, C₂/γ₂** — quote the saturated backstresses (well-identified)
  rather than (C, γ) pairs (which are rank-deficient).
* **r₀, r₄₅, r₉₀** — quote the multi-zone MATLAB pooled values with the
  inter-specimen sample s.d.
