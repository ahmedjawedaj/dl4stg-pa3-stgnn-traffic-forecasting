"""Sanity baselines (not trained): last-value persistence and the 2020 time-of-week average.

Run: python scripts/naive_baselines.py   (writes results/naive_baselines.json)
"""
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pa3.data import Prepared, T_OUT  # noqa: E402


def score(pred, y, m):
    e = (pred - y)[m]
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}


def main():
    p = Prepared(os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    filled = p.z * p.std + p.mean                                   # the model input in original units
    tow = (p.times.dayofweek * 96 + p.times.hour * 4 + p.times.minute // 15).to_numpy()
    tr = np.asarray(p.times.year == 2020)
    prof = np.full((672, p.N), np.nan, dtype=np.float32)
    for k in range(672):
        sel = tr & (tow == k)
        with warnings.catch_warnings():                 # sensors without 2020 data, filled below
            warnings.simplefilter("ignore", RuntimeWarning)
            prof[k] = np.nanmean(p.flow[sel], 0)
    prof = np.where(np.isnan(prof), p.mean[None, :], prof)
    out = {}
    for split in ("val", "test"):
        t = p.idx[split]
        to = t[:, None] + np.arange(1, T_OUT + 1)
        y, m = p.target[to], p.mask[to]
        out[split] = {"persistence": score(np.repeat(filled[t][:, None], T_OUT, 1), y, m),
                      "time_of_week_average_2020": score(prof[tow[to]], y, m)}
    json.dump(out, open(os.path.join(ROOT, "results", "naive_baselines.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
