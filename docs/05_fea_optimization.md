# 5. FEA-based inverse identification

This is the heart of the methodology: the Chaboche backstress parameters
(C₁, γ₁, C₂, γ₂) — and optionally the initial yield σ₀ — are identified by
**minimising** the root-mean-square error between the Abaqus tensile
simulation and the experimental true-stress curves at 0°, 45°, **and** 90°
simultaneously.

```
       ┌──── update [σ₀, C₁, γ₁, C₂, γ₂] ───┐
       │                                   │
       ▼                                   │
  ┌──────────┐    σ_sim_θ(ε̄ᵖ)     ┌────────┴────────┐
  │ Abaqus   │ ──────────────────► │ NRMSE_θ(σ_sim ; │
  │ 3 jobs   │                     │   σ_exp)        │
  │ (0/45/90)│                     │ weighted Σ      │
  └────▲─────┘                     └────────┬────────┘
       │       ┌──── Nelder-Mead simplex ────┘
       │       │ (or DE / Bayesian)
       │       ▼
       │  best parameters
       └─────────────
```

---

## 5.1 Surrogate run (no Abaqus, ~ 1 minute)

This is the most useful entry point — the surrogate analytical model uses
the **closed-form** combined hardening expression (Eq. 5.1 of the thesis),
which is exact for monotonic uniaxial tension. It lets you:

* prototype your weighting scheme,
* compare optimisation algorithms (Nelder-Mead, DE, PSO, Bayesian),
* obtain a *first-pass* parameter set before booking Abaqus time.

```bash
python multi_objective_optimization.py
```

Outputs:

| File                                              | Contents                          |
| ------------------------------------------------- | --------------------------------- |
| `output/fig_multi_direction_optimization.png`     | predicted vs experimental curves  |
| `output/fig_algorithm_comparison.png`             | NRMSE convergence per algorithm   |
| `optimization_results/optimized_parameters.txt`   | best (σ₀, C₁, γ₁, C₂, γ₂)         |
| `optimization_results/convergence_history.csv`    | full simplex history              |

---

## 5.2 Full FEA-based identification

This stage **requires Abaqus** and is meant for the lab PC. The host script
launches Abaqus in the background, parses the ODB, computes the cost, and
asks scipy for the next parameter trial — repeat until convergence.

### Command

```bat
:: from a Windows "Abaqus Command" prompt, in the repo root
abaqus python optimize_hardening_multidir.py
```

(macOS / Linux equivalent if Abaqus is installed: `abaqus python ...`).

The script automates:

1. Generation of three input decks (one per direction) with correct
   `*ORIENTATION` rotation. Hill'48 R-values come from `material_constants.json`.
2. Sequential submission of `mopt_y_<dir>.inp`, `mopt_y_<dir>.inp`,
   `mopt_y_<dir>.inp` (Abaqus jobs).
3. Extraction of global force–displacement from the loaded boundary —
   exactly the quantity the UTM measures.
4. Conversion to true-stress and weighted NRMSE
   `J = (RMSE_0 + 2·RMSE_45 + RMSE_90) / 4·(σ_max−σ_min)_per_dir`.
5. Nelder-Mead update; iterate until `xtol = 1e-3` and `ftol = 1e-4`.

Typical run-time on a 16-core lab PC: **~ 40 minutes** for 80–120 simplex
iterations × 3 jobs/iteration.

### Outputs

| File                                                          | Meaning                              |
| ------------------------------------------------------------- | ------------------------------------ |
| `mopt_y_best_<00,45,90>.inp` / `.odb`                         | best simulation files                |
| `optimization_results/optimized_parameters_multidir_yld2000.txt` | identified parameters             |
| `optimization_results/convergence_history_multidir_yld2000.csv` | cost vs iteration                  |
| `output/fig_convergence.png`, `fig_all_directions_comparison.png` | publication plots             |

---

## 5.3 Choice of cost function & weighting

The multi-direction NRMSE
```
J(p) = (1 / Σwᵢ) Σ_θ wᵢ · RMSE_θ(p) / (σ_max,θ − σ_min,θ)
```
uses `w₀ = 1, w₄₅ = 2, w₉₀ = 1` because the 45° direction is the most
sensitive to anisotropy errors in monotonic data and tends to be
under-fit when all directions get equal weight (this is empirically
verified in the thesis; you can change weights via the `WEIGHTS` dict at
the top of `optimize_hardening_multidir.py`).

A **single-direction** version (`optimize_hardening.py`) is provided for
debugging.

---

## 5.4 Mesh, geometry, material card

The reference Abaqus deck `abaqus/inp/Tensile_CKH.inp` uses:

| Item               | Value                                           |
| ------------------ | ----------------------------------------------- |
| Symmetry           | quarter-model (0° plane and mid-plane)          |
| Element            | CPS4R, plane stress, reduced integration        |
| Mesh size          | 1.0 mm (mesh-independent — see §5.5)            |
| Half-width × half-gauge × thickness | 10 × 40 × 1.5 mm                |
| Material card      | `*ELASTIC` + `*PLASTIC, HARDENING=COMBINED`     |
| Hill'48            | `*POTENTIAL` from r-values                      |
| Loading            | displacement-controlled, NLGEOM = ON            |

Modify `generate_abaqus_inp.py` to change geometry; everything downstream
(extraction, plotting) follows.

## 5.5 Mesh convergence

Already done and tracked in the repo (`mesh_*p*_stress_strain.csv`,
`mesh_convergence_results.csv`). To re-run:

```bash
python mesh_convergence_generate.py    # writes Tensile_mesh_*.inp
abaqus job=Tensile_mesh_0p5 …          # 3 jobs on the lab PC
abaqus python mesh_convergence_extract.py
python  mesh_convergence_plot.py       # → output/fig_convergence.png
```

The 1.0 mm mesh agrees with the 0.5 mm mesh to within 0.3 % in true stress
and 0.4 % in PEEQ at ε̄ᵖ = 0.20.
