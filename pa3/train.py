"""Training and evaluation protocol shared by every model (Q4, Q5)."""
import argparse
import json
import os
import platform
import random
import time

import numpy as np
import torch

from .data import Prepared, WindowLoader
from .models import Forecaster

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def masked_mae(pred, y, m):
    return (pred - y).abs()[m].mean()


@torch.no_grad()
def evaluate(model, loader, mean, std, keep_nodes=None):
    model.eval()
    abs_sum = torch.zeros(4, dtype=torch.float64); sq_sum = torch.zeros(4, dtype=torch.float64)
    cnt = torch.zeros(4, dtype=torch.float64)
    kept = {"t": [], "pred": [], "true": [], "mask": []}
    node_abs = torch.zeros(mean.shape[-1], dtype=torch.float64); node_cnt = torch.zeros_like(node_abs)
    for x, y, m, t in loader:
        p = model(x) * std + mean
        e = (p - y).double()
        mf = m.double()
        abs_sum += (e.abs() * mf).sum((0, 2)); sq_sum += (e ** 2 * mf).sum((0, 2)); cnt += mf.sum((0, 2))
        node_abs += (e.abs() * mf).sum((0, 1)); node_cnt += mf.sum((0, 1))
        if keep_nodes is not None:
            kept["t"].append(t.numpy()); kept["pred"].append(p[:, :, keep_nodes].numpy())
            kept["true"].append(y[:, :, keep_nodes].numpy()); kept["mask"].append(m[:, :, keep_nodes].numpy())
    res = {"MAE": float(abs_sum.sum() / cnt.sum()), "RMSE": float(np.sqrt(sq_sum.sum() / cnt.sum())),
           "MAE_h": (abs_sum / cnt).tolist(), "RMSE_h": torch.sqrt(sq_sum / cnt).tolist(),
           "MAE_node": (node_abs / node_cnt.clamp(min=1)).tolist()}
    if keep_nodes is not None:
        kept = {k: np.concatenate(v) for k, v in kept.items()}
        return res, kept
    return res


def hardware():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip(); break
        info["logical_cpus"] = os.cpu_count()
        with open("/proc/meminfo") as f:
            info["mem_gb"] = round(int(f.readline().split()[1]) / 1024 ** 2, 1)
    except OSError:
        import subprocess
        try:
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            info["logical_cpus"] = os.cpu_count()
            info["mem_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024 ** 3, 1)
        except Exception:
            pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["none", "fixed", "adaptive", "hybrid"], required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--de", type=int, default=10)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None, help="fixed hybrid alpha, default learnable")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_batches", type=int, default=0, help="smoke tests only: stop each epoch after this many batches")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    set_seed(args.seed)
    prep = Prepared(args.data)
    mean = torch.from_numpy(prep.mean); std = torch.from_numpy(prep.std)
    model = Forecaster(prep.N, args.graph, prep.adj, d=args.d, de=args.de, alpha=args.alpha,
                       topk=args.topk or None)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = WindowLoader(prep, "train", args.batch, True, args.seed)
    val_loader = WindowLoader(prep, "val", 256, False)
    test_loader = WindowLoader(prep, "test", 256, False)

    name = f"{args.graph}_s{args.seed}{args.tag}"
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, name + ".pt")
    log = {"args": vars(args), "params": n_params, "hardware": hardware(), "splits": prep.split_info(),
           "n_bad_stats": prep.n_bad_stats, "epochs": []}
    best, bad_epochs, train_time = float("inf"), 0, 0.0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.perf_counter(); tot = 0.0; nb = 0
        for x, y, m, _ in train_loader:
            pred = model(x) * std + mean
            loss = masked_mae(pred, y, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
            if args.max_batches and nb >= args.max_batches:
                break
        ep_time = time.perf_counter() - t0
        train_time += ep_time
        val = evaluate(model, val_loader, mean, std)
        rec = {"epoch": ep, "train_mae": tot / nb, "val_mae": val["MAE"], "val_rmse": val["RMSE"], "sec": ep_time}
        if args.graph == "hybrid":
            rec["alpha"] = float(model.alpha())
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
        if val["MAE"] < best - 1e-4:
            best, bad_epochs = val["MAE"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(ckpt))
    best_ep = min(log["epochs"], key=lambda r: r["val_mae"])["epoch"]
    keep_file = os.path.join(ROOT, "results", "plot_sensors.json")
    keep = np.array(json.load(open(keep_file))) if os.path.exists(keep_file) else \
        np.linspace(0, prep.N - 1, 12).astype(int)        # sensors whose test predictions are saved for plots
    val = evaluate(model, val_loader, mean, std)
    t0 = time.perf_counter()
    test, kept = evaluate(model, test_loader, mean, std, keep_nodes=keep)
    infer_time = time.perf_counter() - t0
    log.update({"best_epoch": best_ep, "epochs_run": len(log["epochs"]), "train_time_sec": train_time,
                "sec_per_epoch": train_time / len(log["epochs"]), "test_inference_sec": infer_time,
                "val": val, "test": test})
    if args.graph == "hybrid":
        log["alpha_final"] = float(model.alpha())
    with torch.no_grad():
        A = model.adjacency()
        if A is not None and args.graph != "fixed":
            np.save(os.path.join(args.out, name + "_adj.npy"), A.numpy().astype(np.float32))
            if args.graph == "hybrid":
                np.save(os.path.join(args.out, name + "_adj_adaptive_part.npy"),
                        model.adapt().numpy().astype(np.float32))
    np.savez_compressed(os.path.join(args.out, name + "_testpred.npz"), t=kept["t"], nodes=keep,
                        pred=kept["pred"].astype(np.float16), true=kept["true"].astype(np.float16),
                        mask=kept["mask"])
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(log, f, indent=2)
    print("FINAL", name, json.dumps({"val": val["MAE"], "test_mae": test["MAE"], "test_rmse": test["RMSE"],
                                     "train_time": train_time, "epochs": len(log["epochs"])}))


if __name__ == "__main__":
    main()
