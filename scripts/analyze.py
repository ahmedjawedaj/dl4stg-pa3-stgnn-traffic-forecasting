"""Q6: comparison table, predicted vs actual plots, learned graph analysis.

Run after run_all.sh: python scripts/analyze.py [--seeds 0 1 2]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    out = {"seeds": args.seeds}

    # ---------------- table ----------------
    logs = {g: [json.load(open(os.path.join(RUNS, f"{g}_s{s}.json"))) for s in args.seeds] for g, _, _ in GRAPHS}
    rows = []
    for g, model, adjname in GRAPHS:
        L = logs[g]
        f = lambda key, part: np.array([l[part][key] for l in L])
        r = {"graph": g, "model": model, "adjacency": adjname, "params": L[0]["params"],
             "test_MAE": f("MAE", "test").tolist(), "test_RMSE": f("RMSE", "test").tolist(),
             "val_MAE": f("MAE", "val").tolist(), "val_RMSE": f("RMSE", "val").tolist(),
             "train_time_sec": [l["train_time_sec"] for l in L], "sec_per_epoch": [l["sec_per_epoch"] for l in L],
             "epochs_run": [l["epochs_run"] for l in L], "best_epoch": [l["best_epoch"] for l in L],
             "test_MAE_h": np.mean([l["test"]["MAE_h"] for l in L], 0).tolist(),
             "test_RMSE_h": np.mean([l["test"]["RMSE_h"] for l in L], 0).tolist()}
        if g == "hybrid":
            r["alpha"] = [l["alpha_final"] for l in L]
        rows.append(r)
    out["rows"] = rows
    multi = len(args.seeds) > 1

    def fmt(v, nd=2):
        v = np.array(v)
        return f"{v.mean():.{nd}f} ± {v.std():.{nd}f}" if multi else f"{v.mean():.{nd}f}"

    def tfmt(v):
        # timings from the first seed only: those runs had almost no other CPU load
        return f"{v[0] / 60:.1f} min"

    lines = [f"| Model | Adjacency | Val MAE | Val RMSE | Test MAE | Test RMSE | Train time | Epochs (best) |",
             "|---------|-------|-----------|-----------|-----------|-----------|------|------|"]
    for r in rows:
        ep = f"{np.mean(r['epochs_run']):.0f} ({np.mean(r['best_epoch']):.0f})" if not multi else \
            f"{np.mean(r['epochs_run']):.1f} ({np.mean(r['best_epoch']):.1f})"
        lines.append(f"| {r['model']} | {r['adjacency']} | {fmt(r['val_MAE'])} | {fmt(r['val_RMSE'])} | "
                     f"**{fmt(r['test_MAE'])}** | **{fmt(r['test_RMSE'])}** | {tfmt(r['train_time_sec'])} | {ep} |")
    open(os.path.join(ROOT, "results", "table_main.md"), "w").write("\n".join(lines) + "\n")
    hl = ["| Model | Adjacency | 15 min | 30 min | 45 min | 60 min |", "|---|---|---|---|---|---|"]
    for r in rows:
        hl.append(f"| {r['model']} | {r['adjacency']} | " + " | ".join(
            f"{m:.2f} / {e:.2f}" for m, e in zip(r["test_MAE_h"], r["test_RMSE_h"])) + " |")
    open(os.path.join(ROOT, "results", "table_horizon.md"), "w").write("\n".join(hl) + "\n")
    if multi:
        sl = ["| Model | Adjacency | " + " | ".join(f"seed {s}" for s in args.seeds) + " |",
              "|---|---|" + "---|" * len(args.seeds)]
        for r in rows:
            sl.append(f"| {r['model']} | {r['adjacency']} | " + " | ".join(
                f"{m:.2f} / {e:.2f}" for m, e in zip(r["test_MAE"], r["test_RMSE"])) + " |")
        open(os.path.join(ROOT, "results", "table_seeds.md"), "w").write("\n".join(sl) + "\n")
    print("\n".join(lines)); print("\n".join(hl))

    # ---------------- learning curves ----------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for g, model, adjname in GRAPHS:
        e = logs[g][0]["epochs"]
        ax.plot([r["epoch"] for r in e], [r["val_mae"] for r in e], marker="o", ms=3, c=COLORS[g],
                label=f"{model} ({adjname})")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation MAE (veh / 15 min)"); ax.legend(fontsize=8)
    ax.set_title(f"Validation MAE per epoch (seed {args.seeds[0]})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q6_curves.png"), dpi=150); plt.close(fig)

    # ---------------- predicted vs actual ----------------
    d = np.load(os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    times = pd.to_datetime(d["times15"]); ids = d["ids"]; fwy = d["fwy"]
    lat, lng, A = d["lat"], d["lng"], d["adj"]
    N = len(ids)
    s0 = args.seeds[0]
    preds = {g: np.load(os.path.join(RUNS, f"{g}_s{s0}_testpred.npz")) for g, _, _ in GRAPHS}
    nodes = preds["none"]["nodes"]
    t_idx = preds["none"]["t"]
    show_nodes = list(range(4))                               # first four saved sensors
    win = (times[t_idx + 1] >= "2021-06-14") & (times[t_idx + 1] < "2021-06-17")   # Mon 14 to Wed 16 June 2021
    for h, hname in [(0, "15min"), (3, "60min")]:
        fig, axs = plt.subplots(len(show_nodes), 1, figsize=(11, 2.3 * len(show_nodes)), sharex=True)
        for k, ax in zip(show_nodes, axs):
            tt = times[t_idx[win] + 1 + h]
            p0 = preds["none"]
            ax.plot(tt, p0["true"][win, h, k].astype(float), c="k", lw=1.6, label="actual")
            for g, model, adjname in GRAPHS:
                ax.plot(tt, preds[g]["pred"][win, h, k].astype(float), c=COLORS[g], lw=0.9,
                        label=f"{model} ({adjname})")
            ax.set_title(f"sensor {ids[nodes[k]]} ({fwy[nodes[k]]})", fontsize=9)
            ax.set_ylabel("veh / 15 min", fontsize=8)
        axs[0].legend(fontsize=7, ncol=5, loc="upper left")
        fig.suptitle(f"Test period, Mon 14 to Wed 16 June 2021: predicted vs actual, {h + 1}-step ahead ({hname})",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"q6_pred_vs_actual_{hname}.png"), dpi=150); plt.close(fig)

    # zoom on one morning to show differences between models
    k = 0
    zw = (times[t_idx + 4] >= "2021-06-15 04:00") & (times[t_idx + 4] < "2021-06-15 12:00")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    tt = times[t_idx[zw] + 4]
    ax.plot(tt, preds["none"]["true"][zw, 3, k].astype(float), c="k", lw=1.8, label="actual")
    for g, model, adjname in GRAPHS:
        ax.plot(tt, preds[g]["pred"][zw, 3, k].astype(float), c=COLORS[g], lw=1.1, label=f"{model} ({adjname})")
    ax.set_title(f"Zoom: sensor {ids[nodes[k]]}, morning of Tue 15 June 2021, 60-min-ahead forecasts", fontsize=10)
    ax.set_ylabel("veh / 15 min"); ax.legend(fontsize=7)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "q6_zoom.png"), dpi=150); plt.close(fig)

    # 60-min forecasts during the weekday morning ramp (05:00 to 09:00), the saved sensors, seed s0
    tgt = times[t_idx + 4]
    ramp = np.asarray((tgt.dayofweek < 5) & (tgt.hour >= 5) & (tgt.hour < 9))
    out["ramp_60min"] = {}
    for g, _, _ in GRAPHS:
        pp = preds[g]["pred"][ramp, 3].astype(float); yy = preds[g]["true"][ramp, 3].astype(float)
        mm = preds[g]["mask"][ramp, 3]
        out["ramp_60min"][g] = {"MAE": float(np.abs(pp - yy)[mm].mean()), "bias": float((pp - yy)[mm].mean())}
    out["ramp_60min"]["n_sensors"] = int(len(nodes))
    # the one-step drop in the zoom plot and the 60-min forecasts issued at that step
    j8 = int(np.where(times == pd.Timestamp("2021-06-15 08:00"))[0][0]); rr_ = int(np.where(t_idx == j8)[0][0])
    fl = d["flow15"]
    out["zoom"] = {"sensor": str(ids[nodes[0]]),
                   "flow_0745": float(fl[j8 - 1, nodes[0]]), "flow_0800": float(fl[j8, nodes[0]]),
                   "flow_0815": float(fl[j8 + 1, nodes[0]]), "actual_0900": float(fl[j8 + 4, nodes[0]]),
                   "pred_0900": {g: float(preds[g]["pred"][rr_, 3, 0]) for g, _, _ in GRAPHS}}

    # per-sensor improvement of fixed graph over temporal-only, test MAE
    mn = np.array(logs["none"][0]["test"]["MAE_node"]); mf = np.array(logs["fixed"][0]["test"]["MAE_node"])
    ma = np.array(logs["adaptive"][0]["test"]["MAE_node"]); mh = np.array(logs["hybrid"][0]["test"]["MAE_node"])
    valid = mn > 0
    out["per_sensor"] = {g: {"frac_sensors_better_than_temporal": float((m[valid] < mn[valid]).mean()),
                             "median_rel_change_pct": float(np.median((m[valid] - mn[valid]) / mn[valid]) * 100)}
                         for g, m in [("fixed", mf), ("adaptive", ma), ("hybrid", mh)]}
    off = ~np.eye(N, dtype=bool)
    deg = ((A != 0) & off).sum(1)
    gain = (mn - mf) / np.where(mn > 0, mn, 1) * 100
    bins = [(0, 0), (1, 10), (11, 25), (26, 70)]
    out["gain_by_degree"] = {f"{a}-{b}": {"n": int(((deg >= a) & (deg <= b) & valid).sum()),
                                          "mean_gain_pct": float(gain[(deg >= a) & (deg <= b) & valid].mean())}
                             for a, b in bins if ((deg >= a) & (deg <= b) & valid).any()}

    # ---------------- learned graph analysis ----------------
    Aad = np.load(os.path.join(RUNS, f"adaptive_s{s0}_adj.npy"))
    Ahy_part = np.load(os.path.join(RUNS, f"hybrid_s{s0}_adj_adaptive_part.npy"))
    road = (A != 0) & off
    dist = haversine(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    rng = np.random.default_rng(0)
    # data-driven similarity on 2020: correlation of 15-min changes of deviations from weekly profile
    tr = times.year == 2020
    X = pd.DataFrame(d["flow15"][tr])
    tow = (times[tr].dayofweek * 96 + times[tr].hour * 4 + times[tr].minute // 15).to_numpy()
    R = (X - X.groupby(tow).transform("mean")).diff()
    C = R.corr(min_periods=1000).to_numpy()
    prof = X.groupby(tow).mean()                                # weekly profile, 672 x N
    PC = np.corrcoef(np.nan_to_num(prof.to_numpy()).T)
    have = ~np.isnan(np.diag(C))                               # sensors with data in 2020

    def analyse(Aw, name, K=10):
        W = Aw.copy(); np.fill_diagonal(W, -np.inf)
        top = np.argsort(-W, 1)[:, :K]                        # strongest K out-neighbours per row
        rows_ = np.repeat(np.arange(N), K); cols_ = top.ravel()
        keep = have[rows_] & have[cols_]
        rr, cc = rows_[keep], cols_[keep]
        rand_c = rng.integers(0, N, size=len(rr))
        ok = (rand_c != rr) & have[rand_c]
        road_r, road_c = np.where(road & have[:, None] & have[None, :])
        in_road = road[rr, cc]
        res = {
            "K": K, "pairs": int(len(rr)),
            "frac_top_in_road_graph": float(in_road.mean()),
            "road_density_baseline": float(road.sum() / (N * (N - 1))),
            "median_km_top": float(np.median(dist[rr, cc])), "median_km_road_edges": float(np.median(dist[road_r, road_c])),
            "median_km_random": float(np.median(dist[rr[ok], rand_c[ok]])),
            "frac_top_within_5km": float((dist[rr, cc] < 5).mean()),
            "frac_random_within_5km": float((dist[rr[ok], rand_c[ok]] < 5).mean()),
            "corr_top": float(np.nanmean(C[rr, cc])), "corr_top_not_in_road": float(np.nanmean(C[rr[~in_road], cc[~in_road]])),
            "corr_road_edges": float(np.nanmean(C[road_r, road_c])), "corr_random": float(np.nanmean(C[rr[ok], rand_c[ok]])),
            "profile_corr_top": float(np.nanmean(PC[rr, cc])), "profile_corr_random": float(np.nanmean(PC[rr[ok], rand_c[ok]])),
            "same_fwy_top": float((fwy[rr] == fwy[cc]).mean()),
            "same_fwy_top_not_in_road": float((fwy[rr[~in_road]] == fwy[cc[~in_road]]).mean()),
            "same_fwy_random": float((fwy[rr[ok]] == fwy[rand_c[ok]]).mean()),
            "row_max_weight_mean": float(np.sort(Aw, 1)[:, -1].mean()),
            "row_top10_mass_mean": float(np.sort(Aw, 1)[:, -10:].sum(1).mean()),
            "uniform_weight": 1.0 / N,
            "self_weight_mean": float(np.diag(Aw).mean()),
            "entropy_ratio": float((-(Aw * np.log(Aw + 1e-12)).sum(1)).mean() / np.log(N)),
        }
        # spearman-type check: learned weight vs data correlation over all pairs with data
        m = off & have[:, None] & have[None, :]
        res["pearson_weight_vs_corr_all_pairs"] = float(np.corrcoef(Aw[m], np.nan_to_num(C[m]))[0, 1])
        # in-degree concentration: are a few sensors attended by everyone?
        indeg = np.bincount(cc, minlength=N)
        res["top5_hub_share_of_top_edges"] = float(np.sort(indeg)[-5:].sum() / len(cc))
        res["hubs"] = [{"id": str(ids[j]), "fwy": str(fwy[j]), "in_top_edges": int(indeg[j])}
                       for j in np.argsort(-indeg)[:5]]
        return res, rr, cc, in_road

    ga, rr, cc, in_road = analyse(Aad, "adaptive")
    gh, _, _, _ = analyse(Ahy_part, "hybrid_adaptive_part")
    out["graph_adaptive"] = ga; out["graph_hybrid_adaptive_part"] = gh

    # do different seeds learn the same graph? overlap of top-10 neighbour sets
    def topsets(M, K=10):
        W = M.copy(); np.fill_diagonal(W, -np.inf)
        return np.argsort(-W, 1)[:, :K]
    if len(args.seeds) > 1:
        mats = [np.load(os.path.join(RUNS, f"adaptive_s{s}_adj.npy")) for s in args.seeds]
        tops = [topsets(M) for M in mats]
        ov = []
        for a_ in range(len(tops)):
            for b_ in range(a_ + 1, len(tops)):
                ov.append(np.mean([len(set(tops[a_][i]) & set(tops[b_][i])) / 10 for i in range(N) if have[i]]))
        out["seed_top10_overlap"] = float(np.mean(ov))
        out["seed_weight_corr"] = float(np.mean([np.corrcoef(mats[a_][off], mats[b_][off])[0, 1]
                                                 for a_ in range(len(mats)) for b_ in range(a_ + 1, len(mats))]))

    # figures: learned adjacency heatmaps side by side with road graph
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    Afix = A.copy(); np.fill_diagonal(Afix, 0); Afix = Afix + np.eye(N); Afix = Afix / Afix.sum(1, keepdims=True)
    for ax, M, title in [(axs[0], Afix, "Fixed: D^-1 (A + I)"),
                         (axs[1], Aad, "Learned adaptive adjacency"),
                         (axs[2], Ahy_part, "Adaptive part of the hybrid")]:
        im = ax.imshow(np.where(M > 0, M, np.nan), cmap="viridis", norm=LogNorm(vmin=1e-6, vmax=1), interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_xlabel("sensor j"); ax.set_ylabel("sensor i")
    fig.colorbar(im, ax=axs, shrink=0.8, label="row-normalised weight (log), white = 0")
    fig.savefig(os.path.join(FIG, "q6_adjacency_learned.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # map of strongest learned edges coloured by whether they are road edges
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    W = Aad.copy(); np.fill_diagonal(W, -np.inf)
    best = np.argmax(W, 1)
    ax = axs[0]
    ax.scatter(lng, lat, s=3, c="lightgray")
    for i in np.where(have)[0]:
        j = best[i]
        if not have[j]:
            continue
        c = "tab:blue" if road[i, j] else "tab:red"
        ax.plot([lng[i], lng[j]], [lat[i], lat[j]], c=c, lw=0.5, alpha=0.6)
    ax.plot([], [], c="tab:blue", label="also a road-graph edge"); ax.plot([], [], c="tab:red", label="not in road graph")
    ax.legend(fontsize=8, loc="lower left"); ax.set_title("Strongest learned out-neighbour of each sensor", fontsize=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax = axs[1]
    bins_ = np.linspace(0, 80, 41)
    road_r, road_c = np.where(road)
    ax.hist(dist[road_r, road_c], bins=bins_, density=True, alpha=0.5, label="road-graph edges")
    ax.hist(dist[rr, cc], bins=bins_, density=True, alpha=0.5, label="top-10 learned edges")
    rp = rng.integers(0, N, size=(20000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]
    ax.hist(dist[rp[:, 0], rp[:, 1]], bins=bins_, density=True, histtype="step", color="k", label="random pairs")
    ax.set_xlabel("straight-line distance (km)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Distance between connected sensors", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "q6_learned_edges.png"), dpi=150); plt.close(fig)

    with open(os.path.join(ROOT, "results", "q6.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
