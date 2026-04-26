# 8. Troubleshooting

## Python

| Symptom                                                     | Fix                                                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| `ModuleNotFoundError: emcee`                                | `pip install -r requirements.txt` (Bayesian script needs `emcee`, `corner`) |
| `FileNotFoundError: raw_data/UTM_00_01.txt`                 | Copy your UTM exports into `raw_data/`. See [02_data_format.md](02_data_format.md). |
| `KeyError: 'Eyy'` in `data_processing.py`                   | The DIC `<spec>.csv` is missing — run Ufreckles "Strain gage" export first. |
| Plots are empty / blank                                     | Matplotlib backend issue. The scripts force `Agg`; if you re-enable an interactive backend, ensure `$DISPLAY` is set. |
| `numpy 2.x` `Cannot interpret '<NA>' as a data type`        | Pandas + NumPy 2 mismatch. Pin `numpy>=1.23,<2`.                    |

## MATLAB

| Symptom                                                     | Fix                                                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| `Result file not found: raw_data/00-01/00-01.res`           | Run Ufreckles correlation first; the `.res` file must exist before the extraction script.|
| `No GOOD zones — aborting`                                  | Lower the R² threshold at the top of `extract_strains_*deg.m`, or cut more replicates. |
| `Undefined function 'load' for input arguments of type ...` | Old MATLAB. Upgrade to R2021a+.                                     |

## Abaqus

| Symptom                                                     | Fix                                                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| `*ERROR ifort.exe not found`                                | Open the *Abaqus Command* prompt **after** sourcing the Intel oneAPI environment, or set `ABAQUS_FORTRAN_CMD` explicitly. |
| `THE VALUE OF EQUIVALENT PLASTIC STRAIN IS GREATER THAN ALL VALUES IN THE TABLE` | Increase `eps_max` in `hardening_table.py` (default 0.40). |
| Job aborts with negative Jacobian on quarter model          | Mesh is too coarse near the corner. Run `mesh_convergence_generate.py` and check that 1.0 mm or finer is used. |
| `Tensile_CKH.odb` exists but the optimiser still re-runs it | The optimiser deletes stale `.odb` and `.lck` files automatically; if it cannot, close any open viewer windows on that file. |
| ODB is empty / 0 frames                                     | The job did not converge. Check `Tensile_CKH.msg` for `***WARNING ... TIME INCREMENT REQUIRED IS LESS THAN MINIMUM`. Decrease `*STEP, INC=` or increase `MAX TIME INCREMENT`. |

## Git

| Symptom                                       | Fix                                                                |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `Permission denied (publickey)` on push       | Make sure your SSH key is loaded: `ssh-add ~/.ssh/id_ed25519`.      |
| Repository says > 100 MB                      | Make sure you did not commit the raw DIC images (`raw_data/<spec>/img_*.png`). The default `.gitignore` excludes them. |
| `error: failed to push some refs`             | Pull first: `git pull --rebase origin main`.                        |

## Reproducibility checklist

Before reporting numbers in a paper:

1. `git status` shows a clean tree.
2. `pip freeze | diff - requirements.txt` shows only acceptable
   version differences.
3. `material_constants.json` matches the values you intend to publish.
4. `python -c "import numpy as np; np.random.seed(0)"` is set wherever
   stochastic algorithms are used (already the case in
   `bayesian_reidentification.py` and `multi_objective_optimization.py`).
