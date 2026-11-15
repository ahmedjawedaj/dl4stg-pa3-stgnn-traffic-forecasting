"""End-to-end smoke test and seed reproducibility of the training script (short runs)."""
import json
import os
import subprocess
import sys

import pytest

from conftest import DATA, ROOT


def run(tmp, seed, tag):
    out = os.path.join(tmp, "runs")
    cmd = [sys.executable, "-m", "pa3.train", "--graph", "hybrid", "--seed", str(seed), "--epochs", "1",
           "--max_batches", "5", "--out", out, "--tag", tag, "--threads", "1"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return json.load(open(os.path.join(out, f"hybrid_s{seed}{tag}.json")))


@pytest.mark.skipif(not os.path.exists(DATA), reason="processed data missing")
def test_smoke_and_reproducible(tmp_path):
    a = run(str(tmp_path), 0, "_a")
    b = run(str(tmp_path), 0, "_b")
    assert a["epochs"][0]["train_mae"] == b["epochs"][0]["train_mae"]
    assert a["test"]["MAE"] == b["test"]["MAE"]
    for k in ("val", "test"):
        assert a[k]["MAE"] > 0 and a[k]["RMSE"] >= a[k]["MAE"]
    assert set(a["splits"]) == {"train", "val", "test"}
    assert 0 < a["alpha_final"] < 1
