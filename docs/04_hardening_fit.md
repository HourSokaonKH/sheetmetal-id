# 4. Hardening law fitting (Swift / Voce / Hockett-Sherby)

The isotropic part of the combined hardening model is fit analytically
**before** any FE optimisation. The fits are robust: a single Python script
reads the merged true-stress / true-strain curves and outputs all three
laws plus their R² values.

## Run

```bash
python data_processing.py
```

This script handles the **entire** experimental data pipeline:

1. Reads `raw_data/UTM_<ANG>_<REP>.txt` (load–time–disp).
2. Reads the per-specimen DIC strain CSVs (`<spec>.csv` + each gauge zone).
3. Synchronises UTM and DIC by image-step ↔ time mapping.
4. Computes engineering and true stress–strain.
5. Pools across the 3 replicates per direction.
6. Fits **Swift**, **Voce**, and **Hockett-Sherby** isotropic hardening
   laws on the **0° pooled curve** (taken as reference for σ̄(ε̄ᵖ)).
7. Writes Abaqus-compatible tabulated hardening curves to
   `output/abaqus_*_hardening.csv`.

## Outputs

| File                                          | Contents                                  |
| --------------------------------------------- | ----------------------------------------- |
| `stress-<spec>.csv`                           | per-specimen UTM-aligned true-stress curves |
| `<spec>.csv`                                  | per-specimen mean DIC strains             |
| `output/fig_stress_strain_combined.png`       | 9-curve overview                          |
| `output/fig_hardening_fits.png`               | Swift / Voce / Hockett-Sherby comparison  |
| `output/abaqus_voce_hardening.csv`            | tabulated for Abaqus `*PLASTIC, HARDENING=ISOTROPIC` |
| `output/abaqus_isotropic_hardening.csv`       | (legacy — Swift form)                     |
| `output/abaqus_hockett_sherby_hardening.csv`  | tabulated H-S form                        |
| `output/anisotropy_diagnostic.txt`            | single-zone r-value sanity check          |

## Fitted model used downstream

The Voce fit
```
σ_y(ε̄ᵖ) = σ₀ + Q∞ · ( 1 − exp(−b · ε̄ᵖ) )
```
yields `σ₀`, `Q∞`, `b` (stored in `material_constants.json`). These three
parameters are **frozen** during the FEA optimisation; only the Chaboche
backstresses (C₁, γ₁, C₂, γ₂) are identified inversely.

If your material requires a different isotropic law (e.g. Hockett-Sherby
for Al alloys), simply change which CSV is read by `generate_abaqus_inp.py`
(currently `output/abaqus_voce_hardening.csv`).

## Diagnostic helpers

| Script                          | Purpose                                       |
| ------------------------------- | --------------------------------------------- |
| `diagnostic_fitting.py`         | side-by-side R² for all fits, identifies overfit |
| `exy_validation.py`             | checks \|ε_xy / ε_yy\| << 1 (alignment quality) |
| `strain_rate_verification.py`   | verifies quasi-static assumption (ε̇ < 10⁻³ s⁻¹) |

---

## Porting to a new material

The pipeline is **material-agnostic**. To apply it to any other rolled
sheet metal (DC04, DP590/780/980, AA5052, AA6016-T4, AA6061, AA7075,
AZ31, brass, Ti-6Al-4V, …), you only need to update **two things** —
**no Python or Fortran source code has to be modified**.

### Step 1 — Replace the experimental data

Drop your own UTM + DIC files into `raw_data/` using the exact naming
convention documented in [02_data_format.md](02_data_format.md):

```
raw_data/
├── UTM_00_01.txt … UTM_90_03.txt        ← your UTM exports
└── 00-01/  …  90-03/                    ← one folder per specimen
       ├── <spec>.res                    ← Ufreckles binary
       └── <spec>-gage-*.csv             ← per-zone strain history
```

The number of replicates per direction is configurable; minimum is 1,
recommended is 3.

### Step 2 — Update `material_constants.json`

Open [`material_constants.json`](../material_constants.json) and replace
the entries with values appropriate for your material. The fields that
**must** be set:

```jsonc
{
  "material": "AA6016-T4 aluminium sheet",          // free text
  "elastic": {
    "E_MPa": 70000.0,                               // Young's modulus
    "nu":    0.33,                                  // Poisson ratio
    "density_tonne_per_mm3": 2.70e-9                // for explicit FEA only
  },
  "voce_isotropic_0deg": {
    "Q_inf_MPa": null,                              // overwritten by data_processing.py
    "b":         null,
    "source": "Voce fit of the 0-deg pooled curve."
  },
  "r_values_multizone_DIC": {
    "r0":  null, "r45": null, "r90": null,          // overwritten by compute_anisotropy.m
    "source": "compute_anisotropy.m"
  },
  "offset_yield_pooled_MPa": {
    "sigma_0":  null, "sigma_45": null, "sigma_90": null
  }
}
```

You can leave fields as `null` — `data_processing.py` and
`compute_anisotropy.m` will overwrite them as soon as they are run.

### Step 3 — Re-run the pipeline

The same 10 commands listed in the [README workflow](../README.md#full-reproducible-workflow-10-steps)
will now identify the parameters for your material:

```bash
python data_processing.py             # → Voce σ₀ Q∞ b for your material
# (MATLAB) extract_strains_*deg.m + compute_anisotropy.m   → r-values
python anisotropy_reference.py        # → Hill'48 F, G, H, N
python multi_objective_optimization.py # → Chaboche C, γ
python barlat_yld2000.py              # → Yld2000-2d α₁..α₈
```

### When to choose which isotropic law

| Material family                             | Recommended law         | Why                                                      |
| ------------------------------------------- | ----------------------- | -------------------------------------------------------- |
| Mild / galvanized steel (SGCC, DC04, DC06)  | **Voce**                | clear saturation plateau                                 |
| HSLA / DP / TRIP steels                     | **Voce** or **Swift**   | Voce captures saturation of DP, Swift the power-law tail |
| Austenitic stainless (304, 316L)            | **Swift**               | extensive power-law range from TWIP/TRIP                 |
| Al 5xxx (AA5052, AA5754)                    | **Hockett-Sherby**      | best fit for serrated PLC flow region                    |
| Al 6xxx (AA6016-T4, AA6061, AA6082)         | **Voce**                | well-defined saturation                                  |
| Al 7xxx (AA7075-T6)                         | **Hockett-Sherby**      | high initial yield + slow saturation                     |
| Mg sheet (AZ31, ZE10)                       | **Voce + Chaboche(2)**  | strong tension-compression asymmetry; needs kinematic    |
| Cu / brass                                  | **Swift**               | extensive cold-work hardening                            |
| Ti / Ti-6Al-4V sheet                        | **Voce + Chaboche(1)**  | moderate kinematic effect from twinning                  |

To switch the isotropic law that is fed to Abaqus, change the CSV
referenced in `generate_abaqus_inp.py` from
`output/abaqus_voce_hardening.csv` to
`output/abaqus_hockett_sherby_hardening.csv` (or the Swift one). All
three are produced by `data_processing.py` in one run.

### Yield surface choice

| Sheet type                            | Recommended yield surface |
| ------------------------------------- | ------------------------- |
| Steel sheet (any grade)               | **Hill'48** (4 params) or **Yld2000-2d** (8 params) |
| Aluminium sheet (mild anisotropy)     | **Yld2000-2d**            |
| Aluminium sheet (strong anisotropy, e.g. AA2090, AA5754) | **Yld2000-2d** with exponent `a = 8` |
| Magnesium sheet (asymmetric yield)    | Yld2000-2d is *not* sufficient — extend with CPB06 (not bundled) |

The Yld2000-2d UMAT (`abaqus/umat/umat_yld2000_table.f`) accepts the
material exponent `a` as a `*USER MATERIAL` constant, so steels (`a=6`)
and FCC alloys (`a=8`) are both supported without recompiling.
