# `abaqus/` — Abaqus assets

| Sub-folder      | Contents                                                |
| --------------- | ------------------------------------------------------- |
| `inp/`          | reference Abaqus input decks (`.inp`)                   |
| `umat/`         | Yld2000-2d UMATs (Fortran 77/90, fixed format)          |
| `scripts/`      | Windows convenience launchers                           |

See [../docs/05_fea_optimization.md](../docs/05_fea_optimization.md) for
the FEA-based identification workflow and
[../docs/06_yld2000_umat.md](../docs/06_yld2000_umat.md) for the Yld2000-2d
yield-surface tooling.

## Reference decks (`inp/`)

| File                       | Purpose                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `Tensile_CKH.inp`          | quarter-symmetry tensile model with Hill'48 + Voce + Chaboche(2) |
| `Tensile_bend_unbend.inp`  | reverse-loading deck (used to verify the Bauschinger response)  |
| `Tensile_mesh_0p5.inp`     | mesh-convergence study, 0.5 mm CPS4R                            |
| `Tensile_mesh_1p0.inp`     | mesh-convergence study, 1.0 mm (production)                     |
| `Tensile_mesh_2p0.inp`     | mesh-convergence study, 2.0 mm                                  |
| `yld2000_umat_<dir>.inp`   | Yld2000-2d UMAT smoke tests at 0°, 45°, 90°                     |

## UMATs (`umat/`)

| File                       | Hardening law (in PROPS)                                |
| -------------------------- | ------------------------------------------------------- |
| `umat_yld2000.f`           | Voce + Chaboche(2) closed-form                          |
| `umat_yld2000_table.f`     | Tabulated `σ_y(κ)`, rebuilt by `hardening_table.py`     |

Both UMATs implement Barlat (2003) Yld2000-2d in plane stress with a
cutting-plane return mapping. The tabulated form is the one used by the
3-direction inverse identification driver.
