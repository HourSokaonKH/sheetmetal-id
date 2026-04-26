# `raw_data/` — your experimental inputs go here

This directory is intentionally **empty** in the published repository
because raw DIC image sequences for nine specimens are too large for Git
(≈ 27 GB on the development machine). To reproduce the full pipeline on
your own data, populate this directory exactly as shown below.

## Required layout

```
raw_data/
├── UTM_00_01.txt        ← UTM ASCII export, one per specimen
├── UTM_00_02.txt
├── UTM_00_03.txt
├── UTM_45_01.txt
├── UTM_45_02.txt
├── UTM_45_03.txt
├── UTM_90_01.txt
├── UTM_90_02.txt
├── UTM_90_03.txt
│
├── 00-01/                                 ← one folder per specimen
│   ├── 00-01.res                          ← Ufreckles binary  ★ REQUIRED
│   ├── 00-01-gage-01.csv                  ← per-zone strain history (≥ 4)
│   ├── 00-01-gage-02.csv
│   ├── …
│   ├── img_0001.png … img_NNNN.png        ← optional, NOT committed
│   └── 00-01.dat                          ← optional Ufreckles log
├── 00-02/  …
├── …
└── 90-03/  …
```

The bundled [`examples/`](../examples/) directory shows this layout filled
in for **specimen 00-01**.

## What is parsed by which script?

| File pattern                     | Read by                                       |
| -------------------------------- | --------------------------------------------- |
| `UTM_*_*.txt`                    | `data_processing.py`, `strain_rate_verification.py` |
| `<spec>/<spec>.res`              | `extract_strains_*deg.m`, `extract_multizone_strains.m` |
| `<spec>/<spec>-gage-*.csv`       | `data_processing.py`, MATLAB extraction scripts |
| `<spec>/img_*.png`               | **none** — kept only for traceability         |

## Format spec

See [docs/02_data_format.md](../docs/02_data_format.md) for the precise
column / field spec.
