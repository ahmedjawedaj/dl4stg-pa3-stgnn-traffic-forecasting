"""Re-evaluate every saved checkpoint and check that it reproduces the metrics stored in results/runs/*.json.

Run: python scripts/verify_checkpoints.py   (writes results/verify_checkpoints.json)
"""
import glob
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pa3.data import Prepared, WindowLoader  # noqa: E402
from pa3.models import Forecaster  # noqa: E402
from pa3.train import evaluate  # noqa: E402


def main():
    prep = Prepared(os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    mean, std = torch.from_numpy(prep.mean), torch.from_numpy(prep.std)
    out = {}
    for js in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*_s*.json"))):
        name = os.path.basename(js)[:-5]
        ckpt = js[:-5] + ".pt"
        if not os.path.exists(ckpt):
            continue
        log = json.load(open(js))
        a = log["args"]
        torch.manual_seed(0)
        m = Forecaster(prep.N, a["graph"], prep.adj, d=a["d"], de=a["de"], skip=a.get("skip", False))
        m.load_state_dict(torch.load(ckpt))
        rec = {}
        for split in ("val", "test"):
            r = evaluate(m, WindowLoader(prep, split, 256, False), mean, std)
            rec[split] = {"MAE": r["MAE"], "RMSE": r["RMSE"],
                          "MAE_logged": log[split]["MAE"], "RMSE_logged": log[split]["RMSE"]}
            rec[split]["match"] = abs(r["MAE"] - log[split]["MAE"]) < 1e-3 and abs(r["RMSE"] - log[split]["RMSE"]) < 1e-3
        out[name] = rec
        print(name, {s: (round(rec[s]["MAE"], 3), rec[s]["match"]) for s in rec}, flush=True)
    out["all_match"] = all(v[s]["match"] for v in out.values() if isinstance(v, dict) for s in ("val", "test"))
    json.dump(out, open(os.path.join(ROOT, "results", "verify_checkpoints.json"), "w"), indent=2)
    print("ALL MATCH" if out["all_match"] else "MISMATCH")


if __name__ == "__main__":
    main()
