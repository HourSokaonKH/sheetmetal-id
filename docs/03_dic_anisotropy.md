# 3. Multi-zone DIC anisotropy (r-value) extraction

This stage turns the DIC `.res` files into Lankford r-values
(r₀, r₄₅, r₉₀) and the Hill'48 anisotropy coefficients (F, G, H, N). It is
the **only** part of the pipeline that requires MATLAB. The remaining
stages run in pure Python.

## Inputs

| Item                                         | Provided by                |
| -------------------------------------------- | -------------------------- |
| `raw_data/<spec>/<spec>.res`                 | Ufreckles                  |
| `raw_data/<spec>/<spec>-gage-NN.csv` (≥ 4)   | Ufreckles "Strain gage" GUI |
| `material_constants.json`                    | repository (edit if needed) |

## Outputs

| File                    | Meaning                                       |
| ----------------------- | --------------------------------------------- |
| `r0_result.mat`         | r₀ + per-zone diagnostics (8 zones × 3 specimens) |
| `r45_result.mat`        | r₄₅                                           |
| `r90_result.mat`        | r₉₀                                           |
| `anisotropy_results.mat`| pooled mean / planar / Hill F,G,H,N           |
| `output/fig_r_value_regression.png` | per-zone ε_xx vs ε_yy linear fits |

## Procedure

```matlab
>> cd /path/to/sheetmetal-id
>> extract_strains_00deg          % → r0_result.mat
>> extract_strains_45deg          % → r45_result.mat
>> extract_strains_90deg          % → r90_result.mat
>> compute_anisotropy             % → anisotropy_results.mat
```

Each `extract_strains_*deg.m` script:

1. Loads the binary `.res` for each replicate (up to 3).
2. For every gauge zone, computes the mean ε_yy and ε_xx histories.
3. Fits ε_xx = m·ε_yy + c on the **uniform plastic regime**
   (default ε_yy ∈ [0.02, 0.10]; tweak at the top of the script).
4. Filters bad zones using
   - `R² ≥ 0.99`,
   - `CV(ε_yy) ≤ 0.05` between specimens,
   - inter-quartile-range rejection on slopes.
5. Pools the surviving zones with weights proportional to the number of
   GOOD zones per replicate (eq. 4.5 of the thesis).
6. Reports r = −m / (1 + m).

## What "GOOD zones" means

A zone is GOOD if it **(a)** has R² ≥ 0.99 in the regression band,
**(b)** is not an IQR outlier, and **(c)** does not show ε_xy / ε_yy > 0.05
(would indicate misalignment or shear localisation). The threshold values
are at the top of each script — adjust them if your DIC noise floor differs.

A typical SGCC specimen yields 6 / 8 GOOD zones. If you have ≤ 3 GOOD zones
across all 3 replicates of a given angle, the script aborts: collect more
specimens.

## Sanity check

After running `compute_anisotropy.m`, compare the printed values against
those in `material_constants.json`:

```matlab
>> compute_anisotropy
…
r0  = 0.712,   r45 = 0.800,   r90 = 0.742
rbar  = 0.763   (normal anisotropy)
Δr    = -0.073  (planar anisotropy)
F = 0.502, G = 0.584, H = 0.416, N = 1.486
```

If your pooled values differ by more than ±0.05, edit
`material_constants.json` and **re-run all downstream Python scripts** —
the Hill'48 yield surface is anchored on r₀, r₄₅, r₉₀.

## Optional: zone-convergence study

The script `zone_convergence_study.m` plots r as a function of the number
of zones included in the pool, for a quick visual confirmation that 8 zones
is sufficient (the SGCC dataset converges by 4 zones).

## Pure-Python fallback (diagnostic only)

If MATLAB is unavailable, `data_processing.py` recomputes single-zone
r-values from the per-specimen `<spec>.csv` files. **The pure-Python value
is reported only for diagnostic purposes** — it does not apply zone-quality
filtering and tends to give r-values that are 2–4 % higher than the MATLAB
multi-zone values. Use the MATLAB pipeline for any publication-quality
result.
