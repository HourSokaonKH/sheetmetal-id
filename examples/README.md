# `examples/` — bundled experimental data

This directory ships with the **complete UTM dataset** (9 ASCII files) plus
**one fully-populated DIC specimen** (`specimen_00-01/` with all 8 gauge
exports), so that new users can follow the entire docs/ tutorial without
needing access to a UTM or a DIC system.

## Quick reuse for a smoke test

```bash
# Stage the example data into raw_data/ (the path the scripts expect)
cp examples/UTM_*.txt   raw_data/
mkdir -p raw_data/00-01
cp examples/specimen_00-01/* raw_data/00-01/

# Run the analytical pipeline
python data_processing.py
```

You will get processed CSVs at the repository root and figures in
`output/`. The other 8 specimens (`00-02`, `00-03`, …, `90-03`) will simply
be skipped with a warning, but the 0° pipeline will run completely.

## Real reproduction

For the full identification you need all 9 specimens. The pre-processed
CSVs (`stress-*.csv`, `*.csv`, and `r*_result.mat`) are tracked at the
repository root so that downstream scripts (`multi_objective_optimization.py`,
`barlat_yld2000.py`, `bayesian_reidentification.py`, …) work *immediately*
even without rebuilding `raw_data/`.
