# Spatio-temporal traffic forecasting on LargeST San Diego

AI651 Deep Learning for Space, Time and Graphs, LUMS, Fall 2026. Assignment 3.

The assignment has two parts:

* **Theory (Q1 and Q2):** what sum, mean and max aggregation can and cannot represent, and why a graph layer needs either directional weights or history to tell which way a slowdown is moving.
* **Experiments (Q3 to Q7):** one-hour traffic flow forecasting for the 716 San Diego sensors of LargeST (2020 and 2021, 15-minute steps), comparing a temporal-only model with graph models that use a fixed road graph, a learned graph and a mix of the two.

The full write-up is in [`report/DL4STG_PA3_report.pdf`](report/DL4STG_PA3_report.pdf).

## Results

Test period April to December 2021, vehicles per 15 minutes, mean ± standard deviation over seeds 0, 1 and 2.

| Model | Adjacency | MAE | RMSE | Training time |
|---|---|---|---|---|
| Temporal-only | None | 63.25 ± 0.46 | 99.39 ± 0.50 | 6.5 min |
| STGNN | Fixed | 96.03 ± 1.13 | 148.99 ± 0.98 | 20.2 min |
| STGNN | Adaptive | 79.19 ± 0.66 | 126.04 ± 1.66 | 22.6 min |
| STGNN | Hybrid | 77.77 ± 0.26 | 123.41 ± 1.37 | 21.6 min |

Main findings:

* **With the architecture as specified, the graph layer hurts.** The head only sees $H = \mathrm{ReLU}(\hat A Z W_g)$. The road graph is dense (median 20 neighbours), so each sensor keeps about 7% of its own features and its forecast becomes a forecast of its neighbourhood. The loss grows with the number of neighbours.
* **Neighbours do help when a sensor keeps its own features.** With a skip connection ($Z + H$ into the head), all three graph models beat the temporal-only model (fixed 58.48, adaptive 53.43, hybrid 53.11). This is reported only as a comparison.
* **The learned graph is not a road map.** Only 7% of its strongest edges are road-graph edges, they are about as far apart as random sensor pairs, and different seeds learn almost completely different graphs with similar error.

Training times are wall-clock CPU time for the seed 0 runs (Intel Xeon, 2 logical CPUs). No GPU is needed.

## Repository layout

| Path | Content |
|---|---|
| `scripts/q1_q2.py` | numerical checks for Q1 and Q2, ring-road heatmaps |
| `scripts/prepare_data.py` | extracts San Diego (District 11) for 2020 and 2021, sums 5-minute counts into 15-minute bins, slices the adjacency |
| `scripts/q3_analysis.py` | missing values, weekly plot, graph statistics, neighbourhoods |
| `pa3/data.py` | forecasting windows, chronological splits, normalisation |
| `pa3/models.py` | temporal convolution encoder, graph convolution (fixed, adaptive or hybrid adjacency), MLP head |
| `pa3/train.py` | training and evaluation, one run per call |
| `run_all.sh` | trains the four models for the given seeds |
| `scripts/analyze.py` | comparison table, prediction plots, learned-graph analysis |
| `scripts/verify_checkpoints.py` | reloads every saved checkpoint and checks it reproduces its logged metrics |
| `tests/` | pytest suite |
| `report/` | report sources and `build.py`, which fills every number from `results/` and renders the PDF |
| `results/`, `logs/`, `figures/` | per-run metrics and configuration, training logs, figures |

## Setup

Python 3.10 or newer.

```bash
./setup_env.sh
source .venv/bin/activate
```

## Data

The LargeST California release is on Kaggle (`liuxu77/largest`). Place these files in `data/raw/` under the names on the left:

| File in `data/raw/` | LargeST file |
|---|---|
| `2020.h5` | `ca_his_raw_2020.h5` |
| `2021.h5` | `ca_his_raw_2021.h5` |
| `metadata.csv` | `ca_meta.csv` |
| `adjacency.npy` | `ca_rn_adj.npy` |

The raw files are about 15 GB and are not tracked. `prepare_data.py` reads only the San Diego columns, so it runs in a few GB of memory.

## Reproduce

```bash
python scripts/prepare_data.py --raw_dir data/raw
python scripts/q1_q2.py
python scripts/q3_analysis.py
./run_all.sh 0 1 2                    # about 1 hour per seed on a 2-core CPU
python scripts/analyze.py --seeds 0 1 2
python scripts/verify_checkpoints.py
PA3_RAW_DIR=data/raw python -m pytest tests
python report/build.py                # needs pandoc and xelatex
```

A single model can be trained with `python -m pa3.train --graph {none,fixed,adaptive,hybrid} --seed 0`. The skip-connection comparison uses `--skip --tag _skip`. Each run writes `results/runs/<graph>_s<seed>.json` with its configuration, per-epoch log, metrics, hardware and timing.

## Configuration

Hidden size 32, two temporal convolution layers (kernel 3), one graph convolution, head 32 to 32 to 4, node embedding size 10, learned hybrid weight (starting at 0.5). Adam with learning rate 0.001, batch 64, gradient clipping at 5, at most 20 epochs with early stopping (patience 4), masked MAE loss, per-sensor normalisation from 2020 data only.

## Data source and licence

LargeST: Liu et al., *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting*, NeurIPS 2023 Datasets and Benchmarks. The dataset is released under CC BY-NC 4.0. For this work the files were downloaded from a HuggingFace mirror of the official release and verified against its SHA-256 hashes. The San Diego graph matches the paper (716 nodes, 17,319 edges).
