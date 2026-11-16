"""Q6: comparison table, predicted vs actual plots and learned graph analysis.

Run after run_all.sh: python scripts/analyze.py --seeds 0 1 2
Writes results/q6.json, results/table_main.md and figures/q6_*.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
FIG = os.path.join(ROOT, "figures")
GRAPHS = [("none", "Temporal-only", "None"), ("fixed", "STGNN", "Fixed"),
          ("adaptive", "STGNN", "Adaptive"), ("hybrid", "STGNN", "Hybrid")]
COLORS = {"none": "#7f7f7f", "fixed": "#1f77b4", "adaptive": "#ff7f0e", "hybrid": "#2ca02c"}


def haversine(lat1, lng1, lat2, lng2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lng2 - lng1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def top_k(M, k=10):
    """Indices of the k largest off-diagonal entries of each row."""
    W = M.copy()
    np.fill_diagonal(W, -np.inf)
    return np.argsort(-W, 1)[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    s0 = args.seeds[0]
    out = {"seeds": args.seeds}

    # ---------------- Q6(a) table, in the handout's format ----------------
    logs = {g: [json.load(open(os.path.join(RUNS, f"{g}_s{s}.json"))) for s in args.seeds] for g, _, _ in GRAPHS}
    rows = []
    for g, model, adjname in GRAPHS:
        L = logs[g]
        r = {"graph": g, "model": model, "adjacency": adjname, "params": L[0]["params"]}
        for part in ("val", "test"):
            for key in ("MAE", "RMSE"):
                r[f"{part}_{key}"] = [l[part][key] for l in L]
        r["train_time_sec"] = [l["train_time_sec"] for l in L]
        r["sec_per_epoch"] = [l["sec_per_epoch"] for l in L]
        r["epochs_run"] = [l["epochs_run"] for l in L]
        r["best_epoch"] = [l["best_epoch"] for l in L]
        r["test_MAE_h"] = np.mean([l["test"]["MAE_h"] for l in L], 0).tolist()
        if g == "hybrid":
            r["alpha"] = [l["alpha_final"] for l in L]
        rows.append(r)
    out["rows"] = rows

    def pm(v):
        return f"{np.mean(v):.2f} ± {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"

    def table(rs):
        lines = ["| Model | Adjacency | MAE | RMSE | Training time |", "|---|---|---|---|---|"]
        for r in rs:
            # training time from the first seed, whose runs had the CPU almost to themselves
            lines.append(f"| {r['model']} | {r['adjacency']} | {pm(r['test_MAE'])} | {pm(r['test_RMSE'])} | "
                         f"{r['train_time_sec'][0] / 60:.1f} min |")
        return "\n".join(lines) + "\n"

    # comparison only: the same graph models with a skip connection, head input Z + H
    skip_rows = []
    for g, model, adjname in GRAPHS[1:]:
        L = [json.load(open(os.path.join(RUNS, f"{g}_s{s}_skip.json"))) for s in args.seeds]
        skip_rows.append({"graph": g, "model": "STGNN + skip", "adjacency": adjname,
                          "test_MAE": [l["test"]["MAE"] for l in L], "test_RMSE": [l["test"]["RMSE"] for l in L],
                          "val_MAE": [l["val"]["MAE"] for l in L], "val_RMSE": [l["val"]["RMSE"] for l in L],
                          "train_time_sec": [l["train_time_sec"] for l in L],
                          "alpha": [l.get("alpha_final") for l in L]})
    out["skip_rows"] = skip_rows
    open(os.path.join(ROOT, "results", "table_main.md"), "w").write(table(rows))
    open(os.path.join(ROOT, "results", "table_skip.md"), "w").write(table(skip_rows))
    print(table(rows) + table(skip_rows))

    # ---------------- predicted vs actual ----------------
    d = np.load(os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    times = pd.to_datetime(d["times15"])
    ids, fwy = d["ids"], d["fwy"]
    lat, lng, A = d["lat"], d["lng"], d["adj"]
    N = len(ids)
    preds = {g: np.load(os.path.join(RUNS, f"{g}_s{s0}_testpred.npz")) for g, _, _ in GRAPHS}
    nodes = preds["none"]["nodes"]
    t_idx = preds["none"]["t"]
    tgt = times[t_idx + 4]                                    # timestamp of the 60-min-ahead target

    win = (tgt >= "2021-06-14") & (tgt < "2021-06-17")      # Mon 14 to Wed 16 June 2021
    fig, axs = plt.subplots(4, 1, figsize=(11, 9.2), sharex=True)
    for k, ax in enumerate(axs):
        ax.plot(tgt[win], preds["none"]["true"][win, 3, k].astype(float), c="k", lw=1.6, label="actual")
        for g, model, adjname in GRAPHS:
            ax.plot(tgt[win], preds[g]["pred"][win, 3, k].astype(float), c=COLORS[g], lw=0.9,
                    label=f"{model} ({adjname})")
        ax.set_title(f"sensor {ids[nodes[k]]} ({fwy[nodes[k]]})", fontsize=9)
        ax.set_ylabel("veh / 15 min", fontsize=8)
    axs[0].legend(fontsize=7, ncol=5, loc="upper left")
    fig.suptitle("Test period, 14 to 16 June 2021: predicted vs actual flow, 60 minutes ahead", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q6_pred_vs_actual.png"), dpi=150)
    plt.close(fig)

    zw = (tgt >= "2021-06-15 04:00") & (tgt < "2021-06-15 12:00")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(tgt[zw], preds["none"]["true"][zw, 3, 0].astype(float), c="k", lw=1.8, label="actual")
    for g, model, adjname in GRAPHS:
        ax.plot(tgt[zw], preds[g]["pred"][zw, 3, 0].astype(float), c=COLORS[g], lw=1.1, label=f"{model} ({adjname})")
    ax.set_title(f"Sensor {ids[nodes[0]]}, morning of Tue 15 June 2021, 60-minute-ahead forecasts", fontsize=10)
    ax.set_ylabel("veh / 15 min")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q6_zoom.png"), dpi=150)
    plt.close(fig)

    # numbers quoted with the plots: weekday morning ramp (05:00 to 09:00) and the dip at 08:00 in the zoom
    ramp = np.asarray((tgt.dayofweek < 5) & (tgt.hour >= 5) & (tgt.hour < 9))
    out["ramp_60min"] = {"n_sensors": int(len(nodes))}
    for g, _, _ in GRAPHS:
        e = (preds[g]["pred"][ramp, 3].astype(float) - preds[g]["true"][ramp, 3].astype(float))[preds[g]["mask"][ramp, 3]]
        out["ramp_60min"][g] = {"MAE": float(np.abs(e).mean()), "bias": float(e.mean())}
    j8 = int(np.where(times == pd.Timestamp("2021-06-15 08:00"))[0][0])
    r8 = int(np.where(t_idx == j8)[0][0])
    fl = d["flow15"][:, nodes[0]]
    out["zoom"] = {"sensor": str(ids[nodes[0]]), "flow_0745": float(fl[j8 - 1]), "flow_0800": float(fl[j8]),
                   "flow_0815": float(fl[j8 + 1]), "actual_0900": float(fl[j8 + 4]),
                   "pred_0900": {g: float(preds[g]["pred"][r8, 3, 0]) for g, _, _ in GRAPHS}}

    # Q4(c): per-sensor change of the fixed graph against temporal-only, by number of road neighbours
    mn = np.array(logs["none"][0]["test"]["MAE_node"])
    mf = np.array(logs["fixed"][0]["test"]["MAE_node"])
    valid = mn > 0                                            # sensors with test data
    off = ~np.eye(N, dtype=bool)
    road = (A != 0) & off
    deg = road.sum(1)
    gain = (mn - mf) / np.where(valid, mn, 1) * 100
    out["fixed_vs_none"] = {"frac_sensors_better": float((mf[valid] < mn[valid]).mean()),
                            "median_change_pct": float(np.median(-gain[valid]))}
    out["gain_by_degree"] = {}
    for a, b in [(0, 0), (1, 10), (11, 25), (26, 70)]:
        sel = (deg >= a) & (deg <= b) & valid
        out["gain_by_degree"][f"{a}-{b}"] = {"n": int(sel.sum()), "mean_gain_pct": float(gain[sel].mean())}

    # ---------------- Q6(b) learned graph analysis ----------------
    Aad = np.load(os.path.join(RUNS, f"adaptive_s{s0}_adj.npy"))
    deg_fixed = road.sum(1)
    out["self_weight"] = {"fixed_mean": float(np.mean(1 / (deg_fixed + 1))),       # diagonal of D^-1 (A + I)
                          "fixed_median_deg": float(np.median(deg_fixed)),
                          "adaptive_mean": float(np.diag(Aad).mean()), "uniform": 1 / N}
    dist = haversine(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    rng = np.random.default_rng(0)
    # 2020 similarity between sensors: correlation of 15-min changes of deviations from the weekly profile,
    # and correlation of the weekly profiles themselves
    tr = times.year == 2020
    X = pd.DataFrame(d["flow15"][tr])
    tow = (times[tr].dayofweek * 96 + times[tr].hour * 4 + times[tr].minute // 15).to_numpy()
    C = (X - X.groupby(tow).transform("mean")).diff().corr(min_periods=1000).to_numpy()
    PC = np.corrcoef(np.nan_to_num(X.groupby(tow).mean().to_numpy()).T)
    have = ~np.isnan(np.diag(C))                              # sensors with 2020 data

    top = top_k(Aad)
    rr, cc = np.repeat(np.arange(N), 10), top.ravel()
    keep = have[rr] & have[cc]
    rr, cc = rr[keep], cc[keep]
    rc = rng.integers(0, N, size=len(rr))                     # random pairs with the same source sensors
    ok = (rc != rr) & have[rc]
    road_r, road_c = np.where(road & have[:, None] & have[None, :])
    indeg = np.bincount(cc, minlength=N)
    out["graph_adaptive"] = {
        "pairs": int(len(rr)),
        "frac_top_in_road_graph": float(road[rr, cc].mean()),
        "road_density_baseline": float(road.sum() / (N * (N - 1))),
        "median_km_top": float(np.median(dist[rr, cc])),
        "median_km_road_edges": float(np.median(dist[road_r, road_c])),
        "median_km_random": float(np.median(dist[rr[ok], rc[ok]])),
        "same_fwy_top": float((fwy[rr] == fwy[cc]).mean()),
        "same_fwy_random": float((fwy[rr[ok]] == fwy[rc[ok]]).mean()),
        "corr_top": float(np.nanmean(C[rr, cc])),
        "corr_road_edges": float(np.nanmean(C[road_r, road_c])),
        "corr_random": float(np.nanmean(C[rr[ok], rc[ok]])),
        "profile_corr_top": float(np.nanmean(PC[rr, cc])),
        "profile_corr_random": float(np.nanmean(PC[rr[ok], rc[ok]])),
        "row_top10_mass_mean": float(np.sort(Aad, 1)[:, -10:].sum(1).mean()),
        "hubs": [{"id": str(ids[j]), "fwy": str(fwy[j]), "in_top_edges": int(indeg[j])} for j in np.argsort(-indeg)[:3]],
    }
    if len(args.seeds) > 1:                                   # do different seeds learn the same graph?
        mats = [np.load(os.path.join(RUNS, f"adaptive_s{s}_adj.npy")) for s in args.seeds]
        pairs = [(a, b) for a in range(len(mats)) for b in range(a + 1, len(mats))]
        tops = [top_k(M) for M in mats]
        out["seed_top10_overlap"] = float(np.mean([np.mean([len(set(tops[a][i]) & set(tops[b][i])) / 10
                                                            for i in np.where(have)[0]]) for a, b in pairs]))
        out["seed_weight_corr"] = float(np.mean([np.corrcoef(mats[a][off], mats[b][off])[0, 1] for a, b in pairs]))

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    Afix = A.copy()
    np.fill_diagonal(Afix, 0)
    Afix = (Afix + np.eye(N)) / (Afix + np.eye(N)).sum(1, keepdims=True)
    for ax, M, title in [(axs[0], Afix, "Fixed: D^-1 (A + I)"), (axs[1], Aad, "Learned adaptive adjacency")]:
        im = ax.imshow(np.where(M > 0, M, np.nan), cmap="viridis", norm=LogNorm(vmin=1e-6, vmax=1), interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("sensor j")
        ax.set_ylabel("sensor i")
    fig.colorbar(im, ax=axs, shrink=0.8, label="row-normalised weight (log), white = 0")
    fig.savefig(os.path.join(FIG, "q6_adjacency_learned.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    best = top[:, 0]
    axs[0].scatter(lng, lat, s=3, c="lightgray")
    for i in np.where(have)[0]:
        j = best[i]
        if have[j]:
            axs[0].plot([lng[i], lng[j]], [lat[i], lat[j]], c="tab:blue" if road[i, j] else "tab:red", lw=0.5, alpha=0.6)
    axs[0].plot([], [], c="tab:blue", label="also a road-graph edge")
    axs[0].plot([], [], c="tab:red", label="not in road graph")
    axs[0].legend(fontsize=8, loc="lower left")
    axs[0].set_title("Strongest learned out-neighbour of each sensor", fontsize=10)
    axs[0].set_xlabel("longitude")
    axs[0].set_ylabel("latitude")
    bins = np.linspace(0, 80, 41)
    rp = rng.integers(0, N, size=(20000, 2))
    rp = rp[rp[:, 0] != rp[:, 1]]
    axs[1].hist(dist[road], bins=bins, density=True, alpha=0.5, label="road-graph edges")
    axs[1].hist(dist[rr, cc], bins=bins, density=True, alpha=0.5, label="top-10 learned edges")
    axs[1].hist(dist[rp[:, 0], rp[:, 1]], bins=bins, density=True, histtype="step", color="k", label="random pairs")
    axs[1].set_xlabel("straight-line distance (km)")
    axs[1].set_ylabel("density")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Distance between connected sensors", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q6_learned_edges.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(ROOT, "results", "q6.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
