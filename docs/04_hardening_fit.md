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
| `exy_validation.py`             | checks |ε_xy / ε_yy| << 1 (alignment quality) |
| `strain_rate_verification.py`   | verifies quasi-static assumption (ε̇ < 10⁻³ s⁻¹) |
