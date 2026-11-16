# DL4STG PA3: GNN expressiveness and traffic forecasting on LargeST-SD

AI651 Deep Learning for Space, Time and Graphs, LUMS, Fall 2026, Assignment 3.

* Q1 and Q2: `scripts/q1_q2.py` checks the derivations numerically and draws the ring-road heatmaps.
* Q3: `scripts/prepare_data.py` extracts the 716 San Diego sensors (District 11) for 2020 and 2021 from the LargeST CA release, sums the 5-minute counts into 15-minute bins and slices the road-distance adjacency. `scripts/q3_analysis.py` produces the missing-value, weekly, adjacency and neighbourhood results.
* Q4 and Q5: `pa3/` holds the windows and splits (`data.py`), the models (`models.py`: temporal convolution encoder, one graph convolution with fixed, adaptive or hybrid adjacency, MLP head) and the training loop (`train.py`). `run_all.sh` trains all four models.
* Q6: `scripts/analyze.py` builds the comparison table, the prediction plots and the learned-graph analysis.
* Checks: `tests/` (pytest) and `scripts/verify_checkpoints.py` (re-evaluates every saved checkpoint against its logged metrics).
* Report: `report/build.py` fills every number from `results/` and renders `report/DL4STG_PA3_report.pdf` (needs pandoc and xelatex).

## Reproduce

```bash
./setup_env.sh && source .venv/bin/activate
# Put the LargeST files (Kaggle liuxu77/largest) in data/raw/ as
#   2020.h5 (ca_his_raw_2020.h5), 2021.h5 (ca_his_raw_2021.h5), metadata.csv (ca_meta.csv), adjacency.npy (ca_rn_adj.npy)
python scripts/prepare_data.py --raw_dir data/raw
python scripts/q1_q2.py
python scripts/q3_analysis.py
./run_all.sh 0 1 2                     # about 1 hour per seed on a 2-core CPU
python scripts/analyze.py --seeds 0 1 2
python scripts/verify_checkpoints.py
PA3_RAW_DIR=data/raw python -m pytest tests
python report/build.py
```

The skip-connection comparison in Q4(c) uses `python -m pa3.train --graph <fixed|adaptive|hybrid> --seed <s> --skip --tag _skip`.

Everything runs on CPU. Each run writes `results/runs/<graph>_s<seed>.json` with the configuration, per-epoch log, metrics, hardware and timings.
