# 6. Yld2000-2d yield surface and UMAT

For a more accurate description of the yield surface than Hill'48 (in
particular the **biaxial stress state**), the repository includes a
Barlat Yld2000-2d implementation, both as a pure-Python identifier and as
an Abaqus Fortran UMAT.

> Yld2000-2d is **optional**. If you only need a uniaxial tensile
> identification, Hill'48 + Voce + Chaboche is enough — skip this file.

## 6.1 Identify the 8 α-coefficients (Python only)

```bash
python barlat_yld2000.py
```

This reads:

* r₀, r₄₅, r₉₀ from `material_constants.json`
* yield-stress ratios σ₀/σ₀, σ₄₅/σ₀, σ₉₀/σ₀ from the same file
* equi-biaxial stress σ_b and r_b (defaults: σ_b/σ₀ = 1.04, r_b = 1.0)

and solves the 8×8 nonlinear system of Barlat (2003) for α₁ … α₈.
Outputs go to `output/yld2000_*` and a yield-locus plot to
`output/fig_yld2000_evaluation.png`.

If σ_b is not measured for your material, set it from a literature value or
keep the default (1.04 σ₀ for IF / DDQ steels). The Yld2000-2d yield
surface depends only weakly on σ_b for ordinary sheet steels.

## 6.2 Sensitivity of the α-coefficients

```bash
python yld2000_param_sensitivity.py
```

Computes the Jacobian d(NRMSE)/dα by finite differences and reports which
α-components are well-conditioned. For SGCC, the monotonic uniaxial cost
surface is **rank-deficient** (only σ₀ is identifiable; (Cᵢ, γᵢ) move the
NRMSE by less than 10⁻³). This motivates the Bayesian re-identification
in the next chapter.

## 6.3 Run the Yld2000-2d UMAT in Abaqus

Two UMAT variants are provided in `abaqus/umat/`:

| File                         | Hardening law                                    |
| ---------------------------- | ------------------------------------------------ |
| `umat_yld2000.f`             | Voce + Chaboche(2) closed-form (params via PROPS) |
| `umat_yld2000_table.f`       | Tabulated `σ_y(κ)` (built by `hardening_table.py`) |

The tabulated form is recommended: the table is rebuilt by the optimiser
on every Nelder-Mead trial, so (σ₀, C₁, γ₁, C₂, γ₂) become **genuine DOFs**
of the FE cost function without re-compiling the UMAT.

### Run a 3-direction validation set

```bat
:: lab-PC convenience wrapper for Windows
abaqus\scripts\run_yld2000_umat_lab_pc.cmd
```

Or directly:

```bash
abaqus python run_yld2000_umat.py --angles 0 45 90 --compare-exp
```

This produces three `.odb` files (`mopt_y_<dir>.odb`) and writes the
extracted true-stress curves to `optimization_results/sim_best_*.csv`,
side-by-side with the experimental data so you can spot residual bias at a
glance.

## 6.4 Compile the UMAT

The `.f` files are **fixed-format Fortran 77/90** and must be linked with
Intel ifort (the only compiler supported by Abaqus on Windows). Once the
"Abaqus Command" prompt is open and Visual Studio environment variables are
loaded, no extra flags are needed: `abaqus job=… user=…` does the rest.
