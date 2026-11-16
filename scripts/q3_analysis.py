"""Q3: missing values, one-week sensor plot, adjacency statistics and neighbourhoods.

Run after prepare_data.py: python scripts/q3_analysis.py
Writes results/q3.json and figures/q3_*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
d = np.load(os.path.join(ROOT, "data", "sd_2020_2021.npz"))
flow, n_obs, times = d["flow15"], d["n_obs"], pd.to_datetime(d["times15"])
ids, adj, lat, lng = d["ids"], d["adj"], d["lat"], d["lng"]
fwy = d["fwy"]
T, N = flow.shape
res = {"T15": int(T), "N": int(N), "first": str(times[0]), "last": str(times[-1])}

# ---------------- missing values ----------------
miss15 = np.isnan(flow)
per_sensor = miss15.mean(0)
res["missing"] = {
    "frac_5min_readings_missing": float(d["missing5"]),
    "frac_15min_bins_all_missing": float(miss15.mean()),
    "frac_15min_bins_partial": float(((n_obs > 0) & (n_obs < 3)).mean()),
    "by_year_15min": {str(y): float(miss15[times.year == y].mean()) for y in (2020, 2021)},
    "sensors_never_observed": int(miss15.all(0).sum()),
    "sensors_no_data_2020": int(miss15[times.year == 2020].all(0).sum()),
    "sensors_no_data_2021": int(miss15[times.year == 2021].all(0).sum()),
    "sensors_over_50pct_missing": int((per_sensor > 0.5).sum()),
    "sensors_under_5pct_missing": int((per_sensor < 0.05).sum()),
    "sensors_60pct_or_more_missing": int((per_sensor >= 0.6).sum()),
    "timestamps_all_sensors_missing": [str(x) for x in times[miss15.all(1)]],
}
# share of missing bins that sit in gaps of at most 4 steps (1 hour), the gaps the input forward fill covers
short = 0
for j in range(N):
    dm = np.diff(np.r_[0, miss15[:, j].astype(np.int8), 0])
    lens = np.where(dm == -1)[0] - np.where(dm == 1)[0]
    short += lens[lens <= 4].sum()
res["missing"]["missing_bins_in_gaps_le_4_frac"] = float(short / miss15.sum())
res["flow_stats_observed"] = {"mean": float(np.nanmean(flow)), "max": float(np.nanmax(flow)),
                              "median": float(np.nanmedian(flow)),
                              "mean_2020": float(np.nanmean(flow[times.year == 2020])),
                              "mean_test_period": float(np.nanmean(flow[times >= "2021-04-01"]))}

# missing over time
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
daily = pd.Series(miss15.mean(1), index=times).resample("D").mean()
axs[0].plot(daily.index, daily.values * 100, lw=0.8)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
axs[0].set_ylabel("% bins missing")
axs[0].set_title("Missing 15-min bins per day (all sensors)", fontsize=10)
axs[1].hist(miss15.mean(0) * 100, bins=50)
axs[1].set_xlabel("% of 15-min bins missing")
axs[1].set_ylabel("sensors")
axs[1].set_title("Missingness per sensor (2020 and 2021)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "q3a_missing.png"), dpi=150)
plt.close(fig)

# ---------------- three sensors over one week ----------------
wk = (times >= "2020-02-03") & (times < "2020-02-10")       # Mon 3 Feb to Sun 9 Feb 2020
miss_wk = miss15[wk].mean(0)
cands = np.where(miss_wk == 0)[0]
cand_fwy = {}
mean_flow = np.nanmean(flow[wk], 0)
for j in cands[np.argsort(-mean_flow[cands])]:
    cand_fwy.setdefault(fwy[j], j)
picks = list(cand_fwy.values())[:3]
fig, ax = plt.subplots(figsize=(11, 3.6))
for j in picks:
    ax.plot(times[wk], flow[wk, j], lw=1, label=f"sensor {ids[j]} ({fwy[j]})")
ax.set_ylabel("vehicles per 15 min")
ax.legend(fontsize=8)
ax.set_title("Traffic flow, Mon 3 Feb to Sun 9 Feb 2020 (15-minute bins)", fontsize=10)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "q3a_week.png"), dpi=150)
plt.close(fig)
res["week_plot_sensors"] = [{"index": int(j), "id": str(ids[j]), "fwy": str(fwy[j])} for j in picks]

# ---------------- adjacency ----------------
off = ~np.eye(N, dtype=bool)
nz = adj != 0
out_deg = (nz & off).sum(1)
in_deg = (nz & off).sum(0)
edges = nz & off
res["adjacency"] = {
    "nodes": int(N), "nonzero_total": int(nz.sum()), "nonzero_offdiag_edges": int(edges.sum()),
    "density_offdiag": float(edges.sum() / (N * (N - 1))),
    "symmetric": bool(np.allclose(adj, adj.T)),
    "reciprocated_edge_frac": float((edges & edges.T).sum() / edges.sum()),
    "max_abs_asym": float(np.abs(adj - adj.T).max()),
    "weight_min_nonzero": float(adj[edges].min()), "weight_max": float(adj[edges].max()),
    "out_degree": {"mean": float(out_deg.mean()), "median": float(np.median(out_deg)), "max": int(out_deg.max())},
    "isolated_out": int((out_deg == 0).sum()),
}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
A_show = np.where(nz, adj, np.nan)
im = axs[0].imshow(A_show, cmap="viridis", norm=LogNorm(vmin=0.01, vmax=1), interpolation="nearest")
axs[0].set_title("SD road-distance adjacency A (716 x 716), zeros white", fontsize=10)
axs[0].set_xlabel("sensor j")
axs[0].set_ylabel("sensor i")
fig.colorbar(im, ax=axs[0], shrink=0.8, label="A_ij (log scale)")
axs[1].hist(out_deg, bins=40, alpha=0.7, label="out-degree (row nonzeros)")
axs[1].hist(in_deg, bins=40, alpha=0.5, label="in-degree (column nonzeros)")
axs[1].set_xlabel("number of neighbours (self excluded)")
axs[1].set_ylabel("sensors")
axs[1].legend(fontsize=8)
axs[1].set_title("Neighbour count distribution", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "q3b_adjacency.png"), dpi=150)
plt.close(fig)

# ---------------- neighbourhoods of one sensor ----------------
med = int(np.argsort(np.abs(out_deg - np.median(out_deg)))[0])
one = np.where(nz[med] & off[med])[0]
B = (nz & off).astype(np.int32)
reach2 = ((B + B @ B)[med] > 0)
reach2[med] = False
two = np.where(reach2)[0]
two_only = np.setdiff1d(two, one)
res["neighbourhood"] = {"sensor_index": med, "sensor_id": str(ids[med]), "fwy": str(fwy[med]),
                        "one_hop": len(one), "two_hop_total": int(len(two)), "two_hop_only": int(len(two_only)),
                        "one_hop_ids": [str(ids[j]) for j in one], "two_hop_only_ids": [str(ids[j]) for j in two_only],
                        "one_hop_fwys": sorted(set(map(str, fwy[one]))),
                        "two_hop_fwys": sorted(set(map(str, fwy[two])))}
# correlation of the selected sensor with its 1-hop, 2-hop-only and other sensors (2020 data)
tr = times.year == 2020
Xdf = pd.DataFrame(flow[tr])
corr = Xdf.corrwith(Xdf[med], min_periods=1000).to_numpy()   # pairwise complete observations
others = np.setdiff1d(np.arange(N), np.r_[two, med])
# raw flow is dominated by the shared daily cycle (and the 2020 lockdown level shift), so also
# correlate the 15-minute changes of each sensor's deviation from its average weekly profile
tow = (times[tr].dayofweek * 96 + times[tr].hour * 4 + times[tr].minute // 15).to_numpy()
prof = Xdf.groupby(tow).transform("mean")
Rdf = (Xdf - prof).diff()
rcorr = Rdf.corrwith(Rdf[med], min_periods=1000).to_numpy()
res["neighbourhood"]["mean_corr_2020"] = {"one_hop": float(np.nanmean(corr[one])),
                                          "two_hop_only": float(np.nanmean(corr[two_only])),
                                          "beyond_two_hop": float(np.nanmean(corr[others]))}
res["neighbourhood"]["mean_corr_2020_deseasonalised_changes"] = {
    "one_hop": float(np.nanmean(rcorr[one])), "two_hop_only": float(np.nanmean(rcorr[two_only])),
    "beyond_two_hop": float(np.nanmean(rcorr[others]))}
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.scatter(lng, lat, s=4, c="lightgray", label="other sensors")
ax.scatter(lng[two_only], lat[two_only], s=10, c="tab:orange", label=f"2-hop only ({len(two_only)})")
ax.scatter(lng[one], lat[one], s=14, c="tab:blue", label=f"1-hop ({len(one)})")
ax.scatter(lng[med], lat[med], s=90, c="red", marker="*", label=f"sensor {ids[med]}")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.legend(fontsize=8, loc="lower left")
ax.set_title(f"1-hop and 2-hop neighbourhood of sensor {ids[med]} ({fwy[med]})", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "q3c_neighbourhood.png"), dpi=150)
plt.close(fig)

with open(os.path.join(ROOT, "results", "q3.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=1))
