"""Forecasting windows, chronological splits and normalisation (Q4a)."""
import warnings

import numpy as np
import pandas as pd
import torch

T_IN, T_OUT = 8, 4
SPLITS = {  # inclusive date ranges, by timestamp of every step in the window
    "train": ("2020-01-01", "2020-12-31 23:59"),
    "val": ("2021-01-01", "2021-03-31 23:59"),
    "test": ("2021-04-01", "2021-12-31 23:59"),
}
FFILL_LIMIT = 4  # forward-fill input gaps up to 1 hour


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def causal_ffill(x, limit):
    """Forward fill NaNs along time (axis 0) for at most `limit` steps. Uses only past values."""
    df = pd.DataFrame(x)
    return df.ffill(limit=limit).to_numpy(dtype=np.float32)


class Prepared:
    """Holds the normalised input series, raw targets and the window start indices per split."""

    def __init__(self, path):
        d = load_npz(path)
        self.times = pd.to_datetime(d["times15"])
        self.flow = d["flow15"].astype(np.float32)            # (T, N) with NaN
        self.adj = d["adj"].astype(np.float32)
        self.ids = d["ids"]
        self.lat, self.lng = d["lat"], d["lng"]
        self.T, self.N = self.flow.shape
        self.mask = ~np.isnan(self.flow)                       # observed targets

        # window index t = position of the last input step; window covers t-7 .. t+4
        starts = {}
        for name, (a, b) in SPLITS.items():
            in_split = (self.times >= pd.Timestamp(a)) & (self.times <= pd.Timestamp(b))
            in_split = np.asarray(in_split)
            t = np.arange(T_IN - 1, self.T - T_OUT)
            ok = in_split[t - (T_IN - 1)] & in_split[t + T_OUT]   # first and last step inside split
            starts[name] = t[ok]
        self.idx = starts

        # normalisation statistics from observed training values only (per sensor)
        tr_end = self.idx["train"].max() + T_OUT + 1
        tr = self.flow[:tr_end]
        assert self.times[tr_end - 1] <= pd.Timestamp(SPLITS["train"][1])
        g_mean, g_std = np.nanmean(tr), np.nanstd(tr)
        with warnings.catch_warnings():                     # sensors with no 2020 data give NaN, handled below
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(tr, axis=0)
            std = np.nanstd(tr, axis=0)
        bad = ~np.isfinite(mean) | ~np.isfinite(std) | (std < 1e-3)
        mean[bad] = g_mean; std[bad] = g_std
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.n_bad_stats = int(bad.sum())

        # model input: causal forward fill, remaining gaps set to the training mean (0 after scaling)
        filled = causal_ffill(self.flow, FFILL_LIMIT)
        z = (filled - self.mean) / self.std
        z[np.isnan(z)] = 0.0
        self.z = z.astype(np.float32)
        self.target = np.nan_to_num(self.flow, nan=0.0).astype(np.float32)

    def split_info(self):
        out = {}
        for k, v in self.idx.items():
            out[k] = {"windows": int(len(v)),
                      "first_input": str(self.times[v[0] - T_IN + 1]),
                      "last_target": str(self.times[v[-1] + T_OUT])}
        return out


class WindowLoader:
    """Yields (x, y, m) batches: x (B, 8, N) normalised, y (B, 4, N) raw flow, m (B, 4, N) bool."""

    def __init__(self, prep, split, batch_size, shuffle, seed=0):
        self.z = torch.from_numpy(prep.z)
        self.y = torch.from_numpy(prep.target)
        self.m = torch.from_numpy(prep.mask)
        self.idx = torch.from_numpy(prep.idx[split])
        self.bs, self.shuffle = batch_size, shuffle
        self.g = torch.Generator().manual_seed(seed)
        self.off_in = torch.arange(-(T_IN - 1), 1)
        self.off_out = torch.arange(1, T_OUT + 1)

    def __len__(self):
        return (len(self.idx) + self.bs - 1) // self.bs

    def __iter__(self):
        order = torch.randperm(len(self.idx), generator=self.g) if self.shuffle else torch.arange(len(self.idx))
        for s in range(0, len(order), self.bs):
            t = self.idx[order[s:s + self.bs]]
            ti = t[:, None] + self.off_in
            to = t[:, None] + self.off_out
            yield self.z[ti], self.y[to], self.m[to], t
