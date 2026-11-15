"""Q3 / Q4(a) data checks: resampling, ordering, splits, leakage, missing values."""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import DATA, RAW, need_raw
from pa3.data import FFILL_LIMIT, T_IN, T_OUT, Prepared, causal_ffill


@pytest.fixture(scope="module")
def d():
    if not os.path.exists(DATA):
        pytest.skip("processed data missing")
    z = np.load(DATA)
    return {k: z[k] for k in z.files}


def test_shapes_and_time_axis(d):
    t = pd.to_datetime(d["times15"])
    assert d["flow15"].shape == (366 * 96 + 365 * 96, 716)
    assert t[0] == pd.Timestamp("2020-01-01 00:00") and t[-1] == pd.Timestamp("2021-12-31 23:45")
    assert (np.diff(d["times15"]) == 900 * 10 ** 9).all()
    assert d["adj"].shape == (716, 716) and len(d["ids"]) == 716 and len(set(d["ids"])) == 716


def test_flow_values_sane(d):
    f = d["flow15"]
    obs = f[~np.isnan(f)]
    assert obs.min() >= 0 and obs.max() < 5000
    # fully observed bins are sums of integer vehicle counts, so they are integers too
    v = f[d["n_obs"] == 3]
    frac = np.minimum(v % 1, 1 - v % 1)
    assert (frac < 1e-3).all()


def test_sensor_order_matches_metadata(d):
    need_raw()
    meta = pd.read_csv(os.path.join(RAW, "metadata.csv"))
    sd = meta[meta.District == 11].reset_index(drop=True)
    assert list(sd.ID.astype(str)) == list(d["ids"])
    adj = np.load(os.path.join(RAW, "adjacency.npy"), mmap_mode="r")
    assert np.array_equal(np.asarray(adj[np.ix_(sd.ID2.values, sd.ID2.values)], dtype=np.float32), d["adj"])
    off = ~np.eye(716, dtype=bool)
    assert int(((d["adj"] != 0) & off).sum()) == 17319          # LargeST paper, SD


def test_resampling_against_raw(d):
    """Every checked 15-min value equals the sum of its three raw 5-min readings (3x mean if partial)."""
    need_raw()
    import h5py
    rng = np.random.default_rng(0)
    with h5py.File(os.path.join(RAW, "2021.h5"), "r") as f:
        cols = f["t/axis0"][:].astype(str)
        pos = {c: i for i, c in enumerate(cols)}
        vals = f["t/block0_values"]
        off2021 = 366 * 96                                         # 2021 starts after 2020 bins
        rows = rng.choice(365 * 96, 40, replace=False)
        for r in sorted(rows):
            raw = vals[3 * r:3 * r + 3, :]
            for j in rng.choice(716, 30, replace=False):
                x = raw[:, pos[d["ids"][j]]]
                got = d["flow15"][off2021 + r, j]
                obs = ~np.isnan(x)
                if obs.sum() == 0:
                    assert np.isnan(got)
                else:
                    assert np.isclose(got, 3 * x[obs].mean(), rtol=1e-5)
                    if obs.all():
                        assert np.isclose(got, x.sum(), rtol=1e-5)
                assert d["n_obs"][off2021 + r, j] == obs.sum()


def test_windows_inside_splits(prep):
    bounds = {"train": ("2020-01-01", "2021-01-01"), "val": ("2021-01-01", "2021-04-01"),
              "test": ("2021-04-01", "2022-01-01")}
    seen = set()
    for name, (a, b) in bounds.items():
        t = prep.idx[name]
        first = prep.times[t - (T_IN - 1)]; last = prep.times[t + T_OUT]
        assert (first >= pd.Timestamp(a)).all() and (last < pd.Timestamp(b)).all()
        assert (np.diff(t) == 1).all()                              # contiguous, nothing skipped inside a split
        assert not (seen & set(t.tolist()))
        seen |= set(t.tolist())
    # windows per split = steps in split - 11
    n = lambda a, b: int(((prep.times >= a) & (prep.times < b)).sum()) - (T_IN + T_OUT - 1)
    for name, (a, b) in bounds.items():
        assert len(prep.idx[name]) == n(a, b)


def test_normalisation_uses_training_data_only(tmp_path, d):
    """Replacing all 2021 values with garbage must not change the statistics or any 2020 input."""
    t = pd.to_datetime(d["times15"])
    d2 = dict(d)
    f = d["flow15"].copy()
    f[t.year == 2021] = np.random.default_rng(0).uniform(0, 5000, size=f[t.year == 2021].shape)
    d2["flow15"] = f
    p2 = tmp_path / "x.npz"
    np.savez(p2, **d2)
    a, b = Prepared(DATA), Prepared(str(p2))
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.std, b.std)
    tr = np.asarray(t.year == 2020)
    assert np.array_equal(a.z[tr], b.z[tr])


def test_inputs_have_no_nan_and_targets_masked(prep):
    assert not np.isnan(prep.z).any()
    assert np.array_equal(prep.mask, ~np.isnan(prep.flow))
    assert (prep.target[~prep.mask] == 0).all()


def test_forward_fill_is_causal_and_limited():
    x = np.array([[1.0], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [7.0]], dtype=np.float32)
    y = causal_ffill(x, FFILL_LIMIT)
    assert y[1:5, 0].tolist() == [1.0] * 4 and np.isnan(y[5, 0]) and y[6, 0] == 7.0
    x2 = x.copy(); x2[6] = 100.0                                    # changing the future
    assert np.array_equal(causal_ffill(x2, FFILL_LIMIT)[:6], y[:6], equal_nan=True)


def test_loader_alignment(prep):
    from pa3.data import WindowLoader
    L = WindowLoader(prep, "val", 7, False)
    x, y, m, t = next(iter(L))
    k = 3
    tt = int(t[k])
    assert np.allclose(x[k].numpy(), prep.z[tt - 7:tt + 1])
    assert np.allclose(y[k].numpy(), prep.target[tt + 1:tt + 5])
    assert np.array_equal(m[k].numpy(), prep.mask[tt + 1:tt + 5])
