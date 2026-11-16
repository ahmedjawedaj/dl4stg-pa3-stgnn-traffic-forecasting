"""Q3(a)/(b) data preparation: extract LargeST San Diego (District 11), 2020 and 2021.

Inputs (LargeST CA release, same files as the Kaggle dataset liuxu77/largest):
  RAW_DIR/2020.h5, RAW_DIR/2021.h5   raw 5-minute flow, pandas HDF (fixed format), all 8600 CA sensors
  RAW_DIR/metadata.csv               ca_meta.csv (ID, Lat, Lng, District, ..., ID2)
  RAW_DIR/adjacency.npy              ca_rn_adj.npy, 8600 x 8600 road-distance adjacency

Output: data/sd_2020_2021.npz with
  flow15   (T15, 716) float32, 15-minute vehicle counts, NaN where all three 5-minute readings are missing
  n_obs    (T15, 716) uint8, number of observed 5-minute readings in each 15-minute bin (0 to 3)
  times15  (T15,) int64 ns timestamps (bin start)
  ids      (716,) sensor IDs in the column order of flow15
  adj      (716, 716) float32 road-distance adjacency, rows and columns in the same order as ids
  lat, lng (716,)   sensor coordinates
  fwy      (716,)   freeway and direction, e.g. "I15-N"
  missing5 scalar fraction of missing 5-minute readings

The SD selection and adjacency slicing follow the official generate_sd_dataset.ipynb:
sd_meta = ca_meta[District == 11], adjacency rows/cols indexed by sd_meta.ID2.

Usage: python scripts/prepare_data.py --raw_dir /path/to/raw
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_columns(path, col_ids, chunk=6000):
    """Read selected sensor columns from a pandas fixed-format HDF file without loading all 8600."""
    with h5py.File(path, "r") as f:
        g = f[list(f.keys())[0]]
        cols = g["axis0"][:].astype(str)
        index = g["axis1"][:].astype("int64")            # ns since epoch
        vals = g["block0_values"]                          # (T, 8600)
        assert vals.shape == (len(index), len(cols)), (vals.shape, len(index), len(cols))
        pos = {c: i for i, c in enumerate(cols)}
        sel = np.array([pos[c] for c in col_ids])
        order = np.argsort(sel)
        sel_sorted = sel[order]
        out = np.empty((len(index), len(sel)), dtype=np.float32)
        for s in range(0, len(index), chunk):
            block = vals[s:s + chunk, :]                   # contiguous read, then column pick
            out[s:s + chunk, :] = block[:, sel_sorted][:, np.argsort(order)]
        dtype = str(vals.dtype)
    return index, out, dtype


def to_15min(times, x):
    """Sum three 5-minute counts per 15-minute bin.

    PeMS 5-minute flow is the number of vehicles counted in the 5-minute interval (summed over lanes),
    so the handout asks for summation. If 1 or 2 of the 3 readings are missing we scale the mean of the
    observed readings by 3 so a partly missing bin is not biased low. A bin with no readings stays NaN.
    """
    t = pd.to_datetime(times)
    assert (np.diff(times) == 300 * 10**9).all(), "5-minute index is not regular"
    assert t[0].minute % 15 == 0, "series must start on a 15-minute boundary"
    T = (len(t) // 3) * 3
    x = x[:T].reshape(T // 3, 3, -1)
    obs = ~np.isnan(x)
    n_obs = obs.sum(1).astype(np.uint8)
    cnt = np.maximum(n_obs, 1)
    mean = np.where(obs, x, 0.0).sum(1) / cnt                # mean of observed readings in the bin
    flow15 = (3.0 * mean).astype(np.float32)
    flow15[n_obs == 0] = np.nan
    return times[:T:3], flow15, n_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.raw_dir, "metadata.csv"))
    sd = meta[meta.District == 11].reset_index(drop=True)
    assert len(sd) == 716, len(sd)
    ids = sd.ID.astype(str).tolist()

    times_all, flows, n_obs_all = [], [], []
    miss5_num = miss5_den = 0
    for year in (2020, 2021):
        idx, x, dtype = read_columns(os.path.join(args.raw_dir, f"{year}.h5"), ids)
        ts = pd.to_datetime(idx)
        print(year, "raw", x.shape, dtype, ts[0], ts[-1])
        assert ts[0] == pd.Timestamp(f"{year}-01-01 00:00:00")
        assert ts[-1] == pd.Timestamp(f"{year}-12-31 23:55:00")
        miss5_num += int(np.isnan(x).sum())
        miss5_den += x.size
        t15, f15, n15 = to_15min(idx, x)
        times_all.append(t15)
        flows.append(f15)
        n_obs_all.append(n15)

    times15 = np.concatenate(times_all)
    flow15 = np.concatenate(flows)
    n_obs = np.concatenate(n_obs_all)
    assert (np.diff(times15) == 900 * 10**9).all()

    adj_ca = np.load(os.path.join(args.raw_dir, "adjacency.npy"), mmap_mode="r")
    id2 = sd.ID2.values
    adj = np.asarray(adj_ca[np.ix_(id2, id2)], dtype=np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, flow15=flow15, n_obs=n_obs, times15=times15, ids=np.array(ids),
                        adj=adj, lat=sd.Lat.values, lng=sd.Lng.values,
                        fwy=np.array(sd.Fwy.astype(str).tolist()),
                        missing5=miss5_num / miss5_den)
    print("saved", args.out, flow15.shape, "missing 5-min", miss5_num / miss5_den,
          "missing 15-min", float(np.isnan(flow15).mean()))


if __name__ == "__main__":
    main()
