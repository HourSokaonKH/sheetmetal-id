# DIC + FEM Inverse Identification Toolkit for Anisotropic Sheet-Metal Plasticity

*Hill'48 / Yld2000-2d yield surfaces · Voce isotropic + Chaboche kinematic
hardening · material-agnostic · open-source.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-%E2%89%A5R2021a-orange.svg)](https://www.mathworks.com/)
[![Abaqus](https://img.shields.io/badge/Abaqus%2FStandard-%E2%89%A52023-red.svg)](https://www.3ds.com/products-services/simulia/products/abaqus/)
[![DOI](https://img.shields.io/badge/cite-CITATION.cff-informational.svg)](CITATION.cff)

> **End-to-end open-source pipeline** for identifying combined isotropic +
> kinematic hardening parameters (Voce + Chaboche) of **any anisotropic
> sheet metal** from uniaxial tension + multi-zone Digital Image
> Correlation (DIC) and inverse Abaqus/Standard finite-element
> optimisation, with Hill'48 and Yld2000-2d yield surfaces.
>
> Designed so that **a researcher with little Python or Abaqus experience
> can reproduce all numerical results from raw UTM and DIC data** by
> following [`docs/`](docs/) step-by-step.

### Applicable to any rolled / cold-rolled sheet metal

The pipeline is material-agnostic. It has been validated on **SGCC JIS G
3302** galvanized steel (the dataset bundled in [`examples/`](examples/)),
but every script reads its constants from a single
[`material_constants.json`](material_constants.json) file, so it can be
applied without code changes to:

- low-carbon and HSLA steels (DC04, DC06, DP590, DP780, DP980, …),
- stainless steels (304, 316L, 430, …),
- aluminium sheet (AA1050, AA5052, AA5754, AA6016-T4, AA6061-T6, AA7075, …),
- magnesium sheet (AZ31, ZE10, …),
- copper, brass, titanium sheet (CP-Ti, Ti-6Al-4V), and
- any other rolled sheet exhibiting in-plane plastic anisotropy.

Bringing a new material in is described in
[docs/04_hardening_fit.md](docs/04_hardening_fit.md#porting-to-a-new-material) —
**no Python or Fortran code needs to be edited**, only the JSON file and
the contents of `raw_data/`.

> Reference experimental campaign for the bundled example: SGCC JIS G 3302
> (DIN 50125 dog-bone 80 × 20 × 1.5 mm, 0° / 45° / 90°, 3 replicates each,
> ε̇ ≈ 4 × 10⁻⁴ s⁻¹, Canon EOS R6 II 4K @ 25 fps DIC).

---

## Table of Contents

1. [What this repository does](#what-this-repository-does)
2. [Pipeline overview](#pipeline-overview)
3. [Repository layout](#repository-layout)
4. [System requirements](#system-requirements)
5. [5-minute quick start (smoke test)](#5-minute-quick-start-smoke-test)
6. [Full reproducible workflow (10 steps)](#full-reproducible-workflow-10-steps)
7. [Headline numerical results](#headline-numerical-results)
8. [Data requirements (UTM + DIC)](#data-requirements-utm--dic)
9. [Outputs & where they go](#outputs--where-they-go)
10. [Troubleshooting](#troubleshooting)
11. [How to cite](#how-to-cite)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)

---

## What this repository does

Given a small experimental dataset (uniaxial tension + DIC at 0°, 45°, 90°),
this codebase identifies a **complete plasticity model** for sheet metals:

| Stage                                             | Output                                                          |
| ------------------------------------------------- | --------------------------------------------------------------- |
| 1. Anisotropy from multi-zone DIC                 | Lankford r-values `r₀, r₄₅, r₉₀`                                |
| 2. Hill'48 yield-surface coefficients             | `F, G, H, N`                                                    |
| 3. Voce isotropic hardening                       | `σ₀, Q∞, b`                                                     |
| 4. Chaboche kinematic hardening                   | `C₁, γ₁, C₂, γ₂` via inverse FEM optimisation                   |
| 5. Barlat Yld2000-2d yield-surface coefficients   | `α₁ … α₈`                                                       |
| 6. Yld2000-2d UMAT                                | Fortran user material for Abaqus/Standard                       |
| 7. Uncertainty quantification                     | Monte-Carlo + Bayesian re-identification                        |

Every parameter is determined from your own measurements, not taken from
literature.

---

## Pipeline overview

```
                ┌────────────────────────────────────────────────────────────┐
                │                       INPUT                                │
                │                                                            │
                │  UTM ASCII (force-displacement)   +   DIC video → Ufreckles│
                │   raw_data/UTM_<dir>_<rep>.txt        raw_data/<spec>/     │
                └─────────────────┬─────────────────────────┬────────────────┘
                                  │                         │
                                  ▼                         ▼
                       ┌──────────────────┐    ┌──────────────────────┐
                       │ data_processing  │    │ extract_strains_*deg │
                       │       .py        │    │       .m  (MATLAB)   │
                       └────────┬─────────┘    └──────────┬───────────┘
                                │                         │
                  Stress-Strain │                         │ r-values
                  CSVs + Voce   │                         │ r0, r45, r90
                                ▼                         ▼
                       ┌────────────────────────────────────────┐
                       │          material_constants.json       │
                       │     (single source of truth, MPa)      │
                       └──────┬──────────────────────┬──────────┘
                              │                      │
                              ▼                      ▼
                ┌──────────────────────────┐ ┌──────────────────────┐
                │ multi_objective_         │ │ barlat_yld2000.py    │
                │ optimization.py  +       │ │ → Yld2000 α₁..α₈     │
                │ optimize_hardening_      │ │                      │
                │ multidir.py (Abaqus)     │ │                      │
                └──────────────┬───────────┘ └──────────┬───────────┘
                               │                        │
                               ▼                        ▼
                  Voce + Chaboche params      umat_yld2000_table.f
                  C₁, γ₁, C₂, γ₂              + yld2000_umat_*.inp
                               │                        │
                               └────────────┬───────────┘
                                            ▼
                              ┌─────────────────────────────┐
                              │ bayesian_reidentification   │
                              │ + sensitivity_analysis      │
                              │ → uncertainty bounds        │
                              └─────────────┬───────────────┘
                                            ▼
                                   figures/, output/,
                                  optimization_results/
```

Detailed per-stage instructions live in [`docs/`](docs/). Each docs file
matches the order above.

---

## Repository layout

```
sheetmetal-id/
├── README.md                     ← this file
├── INSTALL.md                    ← OS-by-OS install commands
├── LICENSE                       ← MIT
├── CITATION.cff                  ← citation metadata
├── requirements.txt              ← Python dependencies (loose pins)
│
├── docs/                         ← 8-step user guide
│   ├── 01_data_acquisition.md
│   ├── 02_data_format.md
│   ├── 03_dic_anisotropy.md
│   ├── 04_hardening_fit.md
│   ├── 05_fea_optimization.md
│   ├── 06_yld2000_umat.md
│   ├── 07_uncertainty.md
│   └── 08_troubleshooting.md
│
├── *.py                          ← 34 Python scripts (flat layout, on purpose)
├── *.m                           ← 6 MATLAB scripts
├── material_constants.json/.py   ← single source of truth for σ₀, r-values, …
│
├── abaqus/
│   ├── inp/                      ← Abaqus input decks
│   ├── umat/                     ← Yld2000-2d UMATs (Fortran)
│   └── scripts/                  ← Windows convenience launchers
│
├── examples/                     ← 9 UTM ASCII + 1 fully-populated DIC specimen
├── raw_data/                     ← (empty, fill from your experiments)
│
├── figures/                      ← reference output figures (PNG)
├── output/                       ← derived numerical outputs
└── optimization_results/         ← optimiser convergence histories
```

> **Why flat instead of `src/`?** Every Python script uses
> `os.path.dirname(__file__)`-relative I/O to find `raw_data/`,
> `material_constants.json`, and the per-specimen CSVs at the same
> level. Likewise MATLAB scripts use `fullfile(pwd, 'raw_data')`.
> Keeping the code flat lets you run any script with no `cd` gymnastics
> and no `PYTHONPATH` edits.

---

## System requirements

| Stage | Tool                     | Min. version       | OS              | Required for                                 |
| ----- | ------------------------ | ------------------ | --------------- | -------------------------------------------- |
| A     | Python                   | 3.9                | macOS/Linux/Win | All Python scripts                           |
| A     | Python pkgs              | see `requirements.txt` | any         | numpy, scipy, pandas, matplotlib, emcee, corner, pillow |
| B     | MATLAB + Ufreckles       | R2021a + 2.4       | macOS/Linux/Win | DIC `.res` parsing, multi-zone r-values      |
| C     | Abaqus/Standard          | 2023               | Linux/Win       | Inverse FEM optimisation, UMAT smoke tests   |
| C     | Intel oneAPI ifort       | 2021               | Linux/Win       | Compiling `umat_yld2000.f` and `umat_yld2000_table.f` |

Hardware that produced the reference results: 16-core workstation, 32 GB RAM, ~ 40 min wall time for the full FEM optimisation. The surrogate path runs on a laptop in ~ 1 min.

Detailed install instructions per OS: [`INSTALL.md`](INSTALL.md).

---

## 5-minute quick start (smoke test)

This runs the analytical-only path on the bundled example data. It does
**not** require Abaqus or MATLAB.

```bash
# 1. Clone
git clone git@github.com:HourSokaonKH/sheetmetal-id.git
cd sheetmetal-id

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Stage the bundled examples into raw_data/
cp examples/UTM_*.txt raw_data/
mkdir -p raw_data/00-01
cp examples/specimen_00-01/* raw_data/00-01/

# 4. Yld2000-2d coefficient solver (uses material_constants.json directly)
python barlat_yld2000.py
#  → yld2000_parameters.txt, fig_yield_surface_comparison.png

# 5. Surrogate (analytical) hardening optimisation, ~1 min
python multi_objective_optimization.py
#  → optimization_results/optimized_parameters_multidir_stress.txt
```

Expected output:

```
r0:     exp=0.7120,  pred=0.7120,  err=0.00%
r45:    exp=0.7998,  pred=0.7998,  err=0.00%
r90:    exp=0.7420,  pred=0.7420,  err=0.00%
```

If you see those three lines, the install is healthy.

---

## Full reproducible workflow (10 steps)

> Each step has a dedicated chapter under [`docs/`](docs/). Read the chapter before running the command.

### Step 1 — Acquire experimental data → [docs/01](docs/01_data_acquisition.md)

- Cut DIN 50125 dog-bones at 0°, 45°, 90° (3 replicates each).
- Apply ~ 0.30–0.40 mm random speckle (50–70 % coverage).
- Record uniaxial tension at ε̇ ≈ 4 × 10⁻⁴ s⁻¹ on the UTM, simultaneously
  film the gauge area at 4K @ 25 fps.
- Convert to image sequences:
  ```bash
  ffmpeg -i video.mp4 -vf "fps=5,crop=1000:2160:(iw-ow)/2:(ih-oh)/2" img_%04d.png
  ```

### Step 2 — Standardise file layout → [docs/02](docs/02_data_format.md)

Drop everything into `raw_data/` exactly as shown in [`raw_data/README.md`](raw_data/README.md):

```
raw_data/
├── UTM_00_01.txt … UTM_90_03.txt
└── 00-01/  00-02/  …  90-03/
       ├── <spec>.res                ← Ufreckles binary
       ├── <spec>-gage-01.csv … N    ← per-zone strain history
       └── img_*.png                 (optional, never committed)
```

### Step 3 — DIC anisotropy (MATLAB) → [docs/03](docs/03_dic_anisotropy.md)

```matlab
% in MATLAB, from the repo root
extract_strains_00deg
extract_strains_45deg
extract_strains_90deg
compute_anisotropy
```

Produces `r0_result.mat`, `r45_result.mat`, `r90_result.mat`,
`anisotropy_results.mat`. Sanity check: r-values should fall in 0.4–1.5 range.

### Step 4 — Hardening fits (Python) → [docs/04](docs/04_hardening_fit.md)

```bash
python data_processing.py
```

Produces:
- `<spec>.csv`, `stress-<spec>.csv` (engineering and true stress-strain) for all 9 specimens.
- `output/abaqus_voce_hardening.csv`, `output/abaqus_isotropic_hardening.csv`, `output/abaqus_hockett_sherby_hardening.csv` (tabulated for Abaqus `*PLASTIC` cards).
- `output/fig_hardening_fits.png`.

### Step 5 — Hill'48 + Voce material card → [docs/04](docs/04_hardening_fit.md#hill48)

```bash
python anisotropy_reference.py
```

Computes Hill'48 `F, G, H, N` from the r-values and writes
`output/hill48_parameters.txt`.

### Step 6 — Inverse FEM optimisation (Abaqus) → [docs/05](docs/05_fea_optimization.md)

```bash
# Surrogate (analytical), ~ 1 min, no Abaqus needed
python multi_objective_optimization.py

# OR full FEM, ~ 40 min on 16 cores
abaqus python optimize_hardening_multidir.py
```

Identifies Chaboche backstresses `C₁, γ₁, C₂, γ₂` while keeping
`σ₀, Q∞, b` frozen at the analytical-fit values. The 3-direction
loss is weighted `w₀ : w₄₅ : w₉₀ = 1 : 2 : 1` to compensate for the
sensitivity bias of 45° tests.

### Step 7 — Yld2000-2d coefficients → [docs/06](docs/06_yld2000_umat.md)

```bash
python barlat_yld2000.py
```

Solves the 8-coefficient nonlinear system from
`r₀, r₄₅, r₉₀, σ₀, σ₄₅/σ₀, σ₉₀/σ₀` (and balanced biaxial if available).
Writes `yld2000_parameters.json`, `yld2000_parameters.txt`, plus three
diagnostic figures.

### Step 8 — Yld2000-2d UMAT verification → [docs/06](docs/06_yld2000_umat.md#smoke-test)

```bash
# In the Abaqus command shell, for each direction
abaqus job=yld2000_umat_00 user=abaqus/umat/umat_yld2000_table interactive
abaqus job=yld2000_umat_45 user=abaqus/umat/umat_yld2000_table interactive
abaqus job=yld2000_umat_90 user=abaqus/umat/umat_yld2000_table interactive
```

The convenience wrapper [`abaqus/scripts/run_yld2000_umat_lab_pc.cmd`](abaqus/scripts/run_yld2000_umat_lab_pc.cmd) automates this on the lab Windows PC.

### Step 9 — Uncertainty quantification → [docs/07](docs/07_uncertainty.md)

```bash
# Monte Carlo (DIC noise ±5 %, σ_y from 3 replicates, E = 200 ± 5 GPa)
python sensitivity_analysis.py

# Bayesian re-identification (literature priors, emcee NUTS)
python bayesian_reidentification.py
```

### Step 10 — Figures for publication → [`generate_fig_*.py`](.)

```bash
python generate_fig_setup.py
python generate_fig_speckle.py
python generate_fig_contours.py
python generate_fig_convergence.py
python generate_fig_flowchart.py
```

Outputs land in `figures/`. They are the same figures used in the
PhD thesis and the JMPT / Metals manuscripts.

---

## Headline numerical results (SGCC G 3302 — reference example)

The values below were produced by running the **same scripts you are
about to run** on the bundled SGCC G 3302 dataset. They are reported
here so that you can verify the install end-to-end. When you point the
pipeline at a different material, the structure of the output is
identical — only the numbers change.

### Anisotropy (DIC, multi-zone, GOOD-zone filter)

| Direction | r-value (this work) |
| --------- | ------------------: |
| 0°        | **0.712**           |
| 45°       | **0.800**           |
| 90°       | **0.742**           |

### Hill'48

| F      | G      | H      | N      |
| -----: | -----: | -----: | -----: |
| 0.502  | 0.584  | 0.416  | 1.486  |

### Isotropic hardening (Voce, 0°)

| σ₀ (MPa)       | Q∞ (MPa) | b      |
| -------------: | -------: | -----: |
| 312.35 → **324.13** (after FEM-tuning) | 335.16 | 3.95   |

### Kinematic hardening (Chaboche, 2 backstresses)

| C₁ (MPa) | γ₁    | C₂ (MPa) | γ₂    |
| -------: | ----: | -------: | ----: |
| ~ 8 600  | ~ 95  | ~ 1 250  | ~ 18  |

> The exact converged values for **your** dataset will be in
> `optimization_results/optimized_parameters_multidir_stress.txt` after
> running step 6.

### Yld2000-2d coefficients

| α₁    | α₂    | α₃    | α₄    | α₅    | α₆    | α₇    | α₈    |
| ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 0.967 | 1.083 | 0.937 | 1.012 | 0.992 | 1.038 | 0.115 | 1.776 |

(See `yld2000_parameters.txt` for full precision.)

---

## Data requirements (UTM + DIC)

This is the **#1 reason inverse identification fails** — getting your
inputs into the exact format the scripts expect.

### UTM ASCII export (one file per specimen)

- File pattern: `raw_data/UTM_<dir>_<rep>.txt`, e.g. `UTM_00_01.txt`.
- Encoding: UTF-8 or Latin-1; both are auto-detected.
- The parser ignores everything before the literal line
  `-----------Curve----------`. After that, expects a tab- or
  whitespace-delimited table whose **header row** contains at least:
  `Time`, `Load`, `Disp`. (Stress and strain columns produced by the
  UTM are deliberately ignored — we recompute them from the raw load
  and the DIC strain to avoid mixed measurement bases.)
- Units: load in **N**, displacement in **mm**, time in **s**.

A working example is provided in [`examples/UTM_00_01.txt`](examples/UTM_00_01.txt).

### DIC: Ufreckles correlation settings

- **Subset size**: 32 × 32 pixels
- **Step size**: 16 px
- **Correlation criterion**: ZNCC (zero-mean normalised cross-correlation)
- **Reference**: image 0 (undeformed)
- **Output**: binary `.res` containing fields `xo, yo, conn, U, pscale`.

### DIC multi-zone strain export (per specimen)

After Ufreckles correlation finishes, export ≥ 4 virtual strain gauges
(zones) per specimen using Ufreckles "Strain gage" tool, saved as:

- File pattern: `raw_data/<spec>/<spec>-gage-<NN>.csv`,
  e.g. `00-01/00-01-gage-01.csv`.
- Delimiter: `;` or `,` (auto-detected).
- Columns required: `Step`, `Eyy`, `Exx`, `Exy`. (`Step` ↔ image
  index, used to align with the UTM time vector.)

A working example is provided in
[`examples/specimen_00-01/`](examples/specimen_00-01/).

### GOOD-zone selection criteria

The MATLAB extractor only keeps zones that satisfy **all** of:

- Linear regression of `Exx` vs `Eyy` has R² ≥ 0.99.
- CV(Eyy) ≤ 5 % within the zone.
- Within IQR (inter-quartile range) of the per-direction pool.
- Shear contamination |Exy / Eyy| < 0.05.

Pooled r-values are obtained by weighting each zone by its `N_GOOD`
sample count.

---

## Outputs & where they go

| File / folder                                  | Produced by                          | Used by                                    |
| ---------------------------------------------- | ------------------------------------ | ------------------------------------------ |
| `<spec>.csv`, `stress-<spec>.csv`              | `data_processing.py`                 | downstream optimisers, figure scripts      |
| `r0_result.mat`, `r45_result.mat`, `r90_result.mat` | MATLAB extractors               | `compute_anisotropy.m`, `material_constants.json` |
| `material_constants.json`                      | hand-curated, Voce fit               | every Python optimiser                     |
| `output/hill48_parameters.txt`                 | `anisotropy_reference.py`            | Abaqus `*POTENTIAL` card                   |
| `output/abaqus_voce_hardening.csv`             | `data_processing.py`                 | Abaqus `*PLASTIC` card                     |
| `optimization_results/optimized_parameters_multidir_stress.txt` | `optimize_hardening_multidir.py` | reporting                |
| `yld2000_parameters.txt/.json`                 | `barlat_yld2000.py`                  | UMATs                                      |
| `figures/fig_*.png`                            | `generate_fig_*.py`                  | thesis, papers, presentation               |

---

## Troubleshooting

The most common pitfalls and their fixes are tabulated in
[`docs/08_troubleshooting.md`](docs/08_troubleshooting.md). Quick links:

- [Python: `ModuleNotFoundError`, `KeyError: 'Eyy'`, blank plots](docs/08_troubleshooting.md#python)
- [MATLAB: `Result file not found`, `No GOOD zones`](docs/08_troubleshooting.md#matlab)
- [Abaqus: `ifort.exe not found`, plastic strain out of table, negative Jacobian](docs/08_troubleshooting.md#abaqus)
- [Git: SSH key, > 100 MB push rejection](docs/08_troubleshooting.md#git)

---

## How to cite

If you use this software, please cite **both** the software and the
underlying PhD thesis. Citation metadata is provided in
[`CITATION.cff`](CITATION.cff). BibTeX:

```bibtex
@software{hour_2026_sheetmetal_id,
  author       = {Hour, Sokaon},
  title        = {{sheetmetal-id: DIC + FEM Inverse Identification
                   Toolkit for Anisotropic Sheet-Metal Plasticity}},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/HourSokaonKH/sheetmetal-id},
  license      = {MIT}
}

@phdthesis{hour_2026_thesis,
  author       = {Hour, Sokaon},
  title        = {{Inverse Identification of Combined Hardening
                   Parameters for Anisotropic Sheet Metals using
                   DIC-Assisted Finite-Element Optimisation}},
  school       = {Institute of Technology of Cambodia},
  year         = {2026}
}
```

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
You are free to use, modify, and redistribute the code in academic and
commercial work, provided the copyright notice is retained.

The bundled experimental data in `examples/` is released under the same
terms.

---

## Acknowledgements

- **Ufreckles** (J. Réthoré, École Centrale de Nantes) — DIC engine.
- **Abaqus/Standard** (Dassault Systèmes) — finite-element solver.
- **Intel oneAPI** — Fortran toolchain for the UMAT.
- **emcee, corner, scipy, numpy, matplotlib** maintainers.
- The **National Polytechnic Institute of Cambodia (NPIC)** for funding
  the doctoral scholarship that supported this research.
- The **Institute of Technology of Cambodia (ITC)** for hosting the
  experimental campaign and providing access to the UTM and DIC rig.

---

> **Questions, bugs, contributions?** Open an issue or pull request on
> [GitHub](https://github.com/HourSokaonKH/sheetmetal-id).
> When reporting a problem, please include the smallest possible
> reproducer, the exact command, and the full traceback.
