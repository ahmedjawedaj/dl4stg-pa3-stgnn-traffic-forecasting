"""Assemble the report: fill {{placeholders}} from results/*.json and render the PDF with pandoc + xelatex.

Run: python report/build.py   (writes report/DL4STG_PA3_report.pdf)
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = lambda name: json.load(open(os.path.join(ROOT, "results", name)))
PARTS = ["front.md", "part1_q1_q3.md", "part2_q4_q7.md", "ai_usage.md"]


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


def values():
    v = {}
    q = R("q1_q2.json")
    v.update(q1b_triples=q["q1b"]["num_triples_checked"], q1b_distinct=q["q1b"]["num_distinct_sums"],
             q1b_collisions=q["q1b"]["collisions"], q1b_max=q["q1b"]["max_sum"], q1c_hoods=q["q1c"]["num_neighbourhoods"],
             q2_a=f"{abs(q['q2a_i']['a']):.3f}", q2_b=f"{q['q2a_i']['b']:.3f}", q2_sse=f"{q['q2a_i']['min_sse']:.3f}",
             q2_centre=f"{q['q2a_ii']['pred_centre_t20']:.2f}", q2_peak20=f"{q['q2a_ii']['pred_peak_t20']:.2f}",
             q2_mass=f"{q['q2a_ii']['pred_mass_t0']:.2f}")
    q = R("q3.json"); m = q["missing"]; a = q["adjacency"]; nb = q["neighbourhood"]
    d = np.load(os.path.join(ROOT, "data", "sd_2020_2021.npz"))
    import pandas as pd
    t = pd.to_datetime(d["times15"])
    never21 = int(np.isnan(d["flow15"][t.year == 2021]).all(0).sum())
    v["bad_stats"] = int(np.isnan(d["flow15"][t.year == 2020]).all(0).sum())
    pm = np.isnan(d["flow15"]).mean(0)
    v["sens_lt5"] = int((pm < 0.05).sum()); v["sens_ge60"] = int((pm >= 0.6).sum())
    v.update(mirror="`ttydsc30/TrafficForecast-LargeST-Project-Dataset`", T15=f"{q['T15']:,}",
             flow_mean=f"{q['flow_stats_observed']['mean']:.0f}", flow_median=f"{q['flow_stats_observed']['median']:.0f}",
             flow_max=f"{q['flow_stats_observed']['max']:.0f}",
             m5=pct(m["frac_5min_readings_missing"]), m15=pct(m["frac_15min_bins_all_missing"]),
             m15_2020=pct(m["by_year_15min"]["2020"]), m15_2021=pct(m["by_year_15min"]["2021"]),
             m_partial=pct(m["frac_15min_bins_partial"], 3), never=m["sensors_never_observed"], never21=never21,
             over50=m["sensors_over_50pct_missing"], allmiss_t=m["timestamps_all_sensors_missing"],
             short_gap_frac=pct(m["missing_bins_in_gaps_le_4_frac"]),
             adj_nodes=a["nodes"], adj_nnz=f"{a['nonzero_total']:,}", adj_edges=f"{a['nonzero_offdiag_edges']:,}",
             adj_density=f"{a['density_offdiag']:.4f}", adj_recip=pct(a["reciprocated_edge_frac"], 1),
             adj_asym=f"{a['max_abs_asym']:.2f}", adj_wmin=f"{a['weight_min_nonzero']:.3f}", adj_wmax=f"{a['weight_max']:.3f}",
             deg_mean=f"{a['out_degree']['mean']:.1f}", deg_med=f"{a['out_degree']['median']:.0f}",
             deg_max=a["out_degree"]["max"], deg_iso=a["isolated_out"],
             nb_id=nb["sensor_id"], nb_fwy=nb["fwy"], nb_one=nb["one_hop"], nb_two=nb["two_hop_total"],
             nb_two_only=nb["two_hop_only"], nb_fwys=", ".join(nb["two_hop_fwys"]),
             corr1=f"{nb['mean_corr_2020_deseasonalised_changes']['one_hop']:.2f}",
             corr2=f"{nb['mean_corr_2020_deseasonalised_changes']['two_hop_only']:.2f}",
             corr3=f"{nb['mean_corr_2020_deseasonalised_changes']['beyond_two_hop']:.2f}",
             rawcorr1=f"{nb['mean_corr_2020']['one_hop']:.2f}", rawcorr3=f"{nb['mean_corr_2020']['beyond_two_hop']:.2f}")
    v["github"] = os.environ.get("PA3_GITHUB", "(GitHub link to be added at submission)")
    p6 = os.path.join(ROOT, "results", "q6.json")
    if os.path.exists(p6):
        q = R("q6.json")
        v["table_main"] = open(os.path.join(ROOT, "results", "table_main.md")).read()
        v["table_horizon"] = open(os.path.join(ROOT, "results", "table_horizon.md")).read()
        ts = os.path.join(ROOT, "results", "table_seeds.md")
        v["table_seeds"] = open(ts).read() if os.path.exists(ts) else ""
        rows = {r["graph"]: r for r in q["rows"]}
        for g, r in rows.items():
            for k in ("test_MAE", "test_RMSE", "val_MAE", "val_RMSE"):
                v[f"{g}_{k}"] = f"{np.mean(r[k]):.2f}"
                v[f"{g}_{k}_sd"] = f"{np.std(r[k]):.2f}"
            v[f"{g}_time"] = f"{r['train_time_sec'][0] / 60:.1f}"      # first seed: clean timing
            v[f"{g}_spe"] = f"{r['sec_per_epoch'][0]:.0f}"
            v[f"{g}_params"] = f"{r['params']:,}"
            v[f"{g}_h1"] = f"{r['test_MAE_h'][0]:.2f}"; v[f"{g}_h4"] = f"{r['test_MAE_h'][3]:.2f}"
        base = np.mean(rows["none"]["test_MAE"])
        for g in ("fixed", "adaptive", "hybrid"):
            v[f"{g}_gain"] = f"{100 * (base - np.mean(rows[g]['test_MAE'])) / base:.1f}"
        v["alpha"] = f"{np.mean(rows['hybrid']['alpha']):.2f}"
        v["alpha_list"] = ", ".join(f"{x:.2f}" for x in rows["hybrid"]["alpha"])
        for g in ("fixed", "adaptive", "hybrid"):
            ps = q["per_sensor"][g]
            v[f"{g}_better_frac"] = pct(ps["frac_sensors_better_than_temporal"], 0)
            v[f"{g}_med_rel"] = f"{ps['median_rel_change_pct']:.1f}"
        for key in ("graph_adaptive", "graph_hybrid_adaptive_part"):
            pre = "ga" if key == "graph_adaptive" else "gh"
            for k, x in q[key].items():
                if isinstance(x, float):
                    v[f"{pre}_{k}"] = f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
                elif isinstance(x, int):
                    v[f"{pre}_{k}"] = f"{x:,}"
            for kk in ("median_km_top", "median_km_road_edges", "median_km_random"):
                v[f"{pre}_{kk}"] = f"{q[key][kk]:.1f}"
            v[f"{pre}_top10_mass_pct"] = pct(q[key]["row_top10_mass_mean"], 0)
            for kk in ("frac_top_within_5km", "frac_random_within_5km", "same_fwy_top", "same_fwy_random"):
                v[f"{pre}_{kk}"] = pct(q[key][kk], 1)
            v[f"{pre}_road_pct"] = pct(q[key]["frac_top_in_road_graph"], 1)
            v[f"{pre}_base_pct"] = pct(q[key]["road_density_baseline"], 1)
            v[f"{pre}_hubs"] = ", ".join(f"{h['id']} ({h['fwy']}, {h['in_top_edges']})" for h in q[key]["hubs"][:3])
        lab = lambda k: "0" if k == "0-0" else k.replace("-", " to ")
        v["gain_by_degree"] = "\n".join(f"| {lab(k)} | {x['n']} | {x['mean_gain_pct']:.1f}% |"
                                        for k, x in q["gain_by_degree"].items())
        g0 = q["gain_by_degree"].get("0-0")
        v["gain_deg0"] = f"{g0['mean_gain_pct']:.1f}% over {g0['n']} sensors" if g0 else "n/a"
        # hybrid vs adaptive wording, using the seed spread
        ha, aa = np.array(rows["hybrid"]["test_MAE"]), np.array(rows["adaptive"]["test_MAE"])
        diff = ha.mean() - aa.mean()
        spread = max(np.std(ha), np.std(aa)) if len(ha) > 1 else None
        within = spread is not None and abs(diff) <= 2 * spread
        if len(ha) == 1:
            v["hybrid_vs_adaptive"] = f"{'slightly better than' if diff < 0 else 'slightly worse than'} the adaptive model ({abs(diff):.2f} MAE, one seed)"
            v["hybrid_vs_adaptive_short"] = "close to the adaptive model"
        elif within:
            v["hybrid_vs_adaptive"] = (f"indistinguishable from the adaptive model within seed variation (difference {diff:+.2f} MAE, "
                                       f"within the seed spread of about {spread:.2f})")
            v["hybrid_vs_adaptive_short"] = "indistinguishable from the adaptive model"
        else:
            v["hybrid_vs_adaptive"] = f"{'better' if diff < 0 else 'worse'} than the adaptive model by {abs(diff):.2f} MAE, more than the seed spread ({spread:.2f})"
            v["hybrid_vs_adaptive_short"] = f"{'slightly better' if diff < 0 else 'slightly worse'} than the adaptive model"
        a05p = os.path.join(ROOT, "results", "runs", f"hybrid_s{q['seeds'][0]}_a05.json")
        if os.path.exists(a05p):
            a5 = json.load(open(a05p))
            v["a05_test_MAE"] = f"{a5['test']['MAE']:.2f}"; v["a05_test_RMSE"] = f"{a5['test']['RMSE']:.2f}"
            v["a05_gain"] = f"{100 * (base - a5['test']['MAE']) / base:.1f}"; v["a05_time"] = f"{a5['train_time_sec'] / 60:.1f}"
            ref = np.array(rows["hybrid"]["test_MAE"])[0]
            d5 = a5["test"]["MAE"] - ref
            v["a05_sentence"] = (f"gives test MAE {a5['test']['MAE']:.2f}, {abs(d5):.2f} {'higher' if d5 > 0 else 'lower'} than the learned-"
                                 f"$\\alpha$ hybrid with the same seed ({ref:.2f}), "
                                 + ("so keeping half the weight on the road graph costs accuracy." if d5 > 0 else
                                    "so keeping half the weight on the road graph does not hurt, and even helps slightly."))
        else:
            for k in ("a05_test_MAE", "a05_test_RMSE", "a05_gain", "a05_time"):
                v[k] = "n/a"
            v["a05_sentence"] = "was not run."
        if "seed_top10_overlap" in q:
            v["seed_overlap_sentence"] = (f"across our seeds, on average only {100 * q['seed_top10_overlap']:.0f}% of a sensor's top-10 "
                                          f"learned neighbours are shared between two seeds, and the weights correlate at "
                                          f"{q['seed_weight_corr']:.2f}, although their test MAEs differ by at most "
                                          f"{np.ptp(rows['adaptive']['test_MAE']):.2f}")
        else:
            v["seed_overlap_sentence"] = "this can be checked by comparing seeds"
        rp = q["ramp_60min"]
        v["ramp_n"] = rp["n_sensors"]
        for g in ("none", "fixed", "adaptive", "hybrid"):
            v[f"ramp_{g}"] = f"{rp[g]['MAE']:.1f}"; v[f"ramp_{g}_bias"] = f"{rp[g]['bias']:+.1f}"
        z = q["zoom"]
        v.update(zoom_sensor=z["sensor"], zoom_0745=f"{z['flow_0745']:.0f}", zoom_0800=f"{z['flow_0800']:.0f}",
                 zoom_0815=f"{z['flow_0815']:.0f}", zoom_0900=f"{z['actual_0900']:.0f}",
                 **{f"zoom_p_{g}": f"{x:.0f}" for g, x in z["pred_0900"].items()})
        nb_ = R("naive_baselines.json")["test"]
        v.update(naive_persist_mae=f"{nb_['persistence']['MAE']:.2f}", naive_persist_rmse=f"{nb_['persistence']['RMSE']:.2f}",
                 naive_tow_mae=f"{nb_['time_of_week_average_2020']['MAE']:.2f}",
                 naive_tow_rmse=f"{nb_['time_of_week_average_2020']['RMSE']:.2f}")
        vc = R("verify_checkpoints.json")
        assert vc["all_match"], "a checkpoint does not reproduce its logged metrics"
        v["n_ckpt"] = sum(1 for k in vc if k != "all_match")
        col = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                             capture_output=True, text=True).stdout
        v["n_tests"] = sum(1 for line in col.splitlines() if "::" in line)
        v["n_resample_checks"] = "1,200"
        tt = pd.to_datetime(d["times15"]); ff = d["flow15"]
        v["flow_train"] = f"{np.nanmean(ff[tt.year == 2020]):.0f}"; v["flow_test"] = f"{np.nanmean(ff[tt >= '2021-04-01']):.0f}"
        run = json.load(open(os.path.join(ROOT, "results", "runs", f"fixed_s{q['seeds'][0]}.json")))
        hw = run["hardware"]
        v["hw"] = f"{hw.get('cpu', 'unknown CPU')}, {hw.get('logical_cpus', '?')} logical CPUs, {hw.get('mem_gb', '?')} GB RAM, " \
                  f"PyTorch {hw['torch']} on CPU ({hw['threads']} threads), Python {hw['python']}, {hw['platform']}"
        v["seeds"] = ", ".join(map(str, q["seeds"])); v["nseeds"] = len(q["seeds"])
        v["seeds_space"] = " ".join(map(str, q["seeds"])); v["seed0"] = q["seeds"][0]
        multi = len(q["seeds"]) > 1
        v["table_note"] = (f"Values are mean ± standard deviation over seeds {v['seeds']}. " if multi else
                           f"Single seed ({q['seeds'][0]}). ") + "Train time is wall-clock time of the training epochs on CPU, measured on the seed " + str(q["seeds"][0]) + " runs, which ran with almost no other load on the machine (later seeds shared the CPU with analysis jobs, so their times are not comparable)."
        v["table_seeds_block"] = ("Test MAE / RMSE of every seed:\n\n" + v["table_seeds"]) if multi else ""
        sp = run["splits"]
        for k in ("train", "val", "test"):
            v[f"n_{k}"] = f"{sp[k]['windows']:,}"
            v[f"{k}_first"] = sp[k]["first_input"]; v[f"{k}_last"] = sp[k]["last_target"]
    return v


def main():
    v = values()
    text = "\n\n".join(open(os.path.join(HERE, p)).read() for p in PARTS if os.path.exists(os.path.join(HERE, p)))

    def sub(mt):
        k = mt.group(1)
        if k not in v:
            raise KeyError(f"missing value for {{{{{k}}}}}")
        return str(v[k])

    filled = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", sub, text)
    bad = [c for c in ("—", "–") if c in filled]
    assert not bad, f"dash characters found: {bad}"
    if "EDITS_PLACEHOLDER" in filled:
        print("WARNING: the AI usage section still has EDITS_PLACEHOLDER. Fill it in before submitting.")
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    md = os.path.join(HERE, "build", "report.md")
    open(md, "w").write(filled)
    pdf = os.path.join(HERE, "DL4STG_PA3_report.pdf")
    cmd = ["pandoc", md, "-o", pdf, "--pdf-engine=xelatex", "--resource-path", ROOT,
           "-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
           "-V", "mainfont=DejaVu Serif", "-V", "monofont=DejaVu Sans Mono", "--toc", "--toc-depth=2",
           "-V", "header-includes=\\usepackage{float}\\floatplacement{figure}{H}\\usepackage{etoolbox}\\AtBeginEnvironment{longtable}{\\small}\\renewcommand{\\abstractname}{Code}"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
