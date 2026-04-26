# Installation

This pipeline runs in three layers, each on its own software stack. You can
install only the layers you need.

| Layer                          | Tools                                  | Purpose                              |
| ------------------------------ | -------------------------------------- | ------------------------------------ |
| **A.** Pure-Python analysis    | Python ≥ 3.9                           | UTM/DIC processing, plots, surrogate optimisation, Bayesian re-id |
| **B.** DIC multi-zone extract  | MATLAB ≥ R2021a + Ufreckles            | reading `.res` files, multi-zone r-values |
| **C.** FEA-based identification| Abaqus/Standard ≥ 2023, Intel Fortran  | Hill'48 / Yld2000-2d forward model + UMAT |

Layer A reproduces every figure in the paper from the bundled CSVs. Add B if
you want to recompute r-values from your own DIC results. Add C if you want
to run a fresh inverse identification.

---

## A. Python environment (mandatory)

### macOS / Linux

```bash
git clone git@github.com:HourSokaonKH/Inverse-hardening-identification-dic-fem.git
cd Inverse-hardening-identification-dic-fem

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
git clone git@github.com:HourSokaonKH/Inverse-hardening-identification-dic-fem.git
cd Inverse-hardening-identification-dic-fem

py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify

```bash
python -c "import numpy, scipy, pandas, matplotlib, emcee, corner; print('OK')"
python data_processing.py            # should write to output/
```

---

## B. MATLAB + Ufreckles (optional, only for fresh DIC extraction)

1. Install MATLAB ≥ R2021a (any toolbox-free configuration is sufficient
   — only built-in functions are used).
2. Install [Ufreckles](https://github.com/jrethore/Ufreckles) following its
   own README. The MATLAB scripts in this repo read the binary `.res` files
   produced by Ufreckles.
3. Add the repository root to your MATLAB path **once** per session:
   ```matlab
   >> cd /path/to/Inverse-hardening-identification-dic-fem
   ```
   All scripts (`extract_strains_00deg.m`, …, `compute_anisotropy.m`) assume
   that you are sitting in the repository root and that raw DIC results live
   in `raw_data/<specimen>/<specimen>.res`.

---

## C. Abaqus + UMAT (optional, only for FEA-based identification)

The Abaqus stage was developed against **Abaqus/Standard 2024** on a
Windows 11 lab PC, but the input decks are version-portable back to 2019.

### Required tools on the lab PC

| Tool                          | Version             | Purpose                          |
| ----------------------------- | ------------------- | -------------------------------- |
| Abaqus/Standard               | ≥ 2023              | implicit FE solver               |
| Intel oneAPI Fortran (`ifort`)| ≥ 2021              | UMAT compilation                 |
| Microsoft Visual Studio       | 2019 / 2022 BuildTools | linker (Windows only)         |
| Python (Abaqus-bundled)       | 2.7                 | Abaqus script kernel             |
| Python                        | ≥ 3.9               | host-side optimiser              |

### Verify Abaqus + ifort

Open a "Compiler ⇒ Abaqus Command" prompt and run:

```bat
abaqus information=system
abaqus verify -user_std
```

The second line compiles a tiny dummy UMAT; if it reports `PASS`, the UMAT
toolchain is ready.

### Smoke test on the bundled `.inp` files

```bat
cd abaqus\inp
abaqus job=Tensile_CKH analysis interactive
```

This runs the reference Hill'48 + Voce + Chaboche(2) tensile model
(quarter-symmetry, 1.0 mm CPS4R mesh) and produces `Tensile_CKH.odb`.

### UMAT smoke test

```bat
cd abaqus\inp
abaqus job=yld2000_umat_00 user=..\umat\umat_yld2000_table interactive
```

If the job converges, the Yld2000-2d UMAT is correctly linked.

---

## Trouble?

See [docs/08_troubleshooting.md](docs/08_troubleshooting.md) for the most
common failure modes (missing toolboxes, ifort linking, `raw_data/` missing,
etc.).
