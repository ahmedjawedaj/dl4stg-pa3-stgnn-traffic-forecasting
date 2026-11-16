"""Fill the {{placeholders}} in report/*.md from results/*.json and render the PDF with pandoc and xelatex.

Run: python report/build.py   (writes report/DL4STG_PA3_report.pdf)
Set PA3_GITHUB to the repository URL to put it on the first page.
"""
import glob
import json
import os
import re
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTS = ["front.md", "part1_q1_q3.md", "part2_q4_q7.md", "ai_usage.md"]


def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


def values():
    v = {"github": os.environ.get("PA3_GITHUB", "(GitHub link to be added at submission)"),
         "mirror": "`ttydsc30/TrafficForecast-LargeST-Project-Dataset`"}

    q = load("q1_q2.json")
    v.update(q1b_triples=q["q1b"]["num_triples_checked"], q1b_distinct=q["q1b"]["num_distinct_sums"],
             q1b_collisions=q["q1b"]["collisions"], q1b_max=q["q1b"]["max_sum"], q1c_hoods=q["q1c"]["num_neighbourhoods"],
             q2_a=f"{abs(q['q2a_i']['a']):.3f}", q2_b=f"{q['q2a_i']['b']:.3f}", q2_sse=f"{q['q2a_i']['min_sse']:.3f}",
             q2_centre=f"{q['q2a_ii']['pred_centre_t20']:.2f}", q2_peak20=f"{q['q2a_ii']['pred_peak_t20']:.2f}",
             q2_mass=f"{q['q2a_ii']['pred_mass_t0']:.2f}")

    q = load("q3.json")
    m, a, nb, fs = q["missing"], q["adjacency"], q["neighbourhood"], q["flow_stats_observed"]
    dc, rc = nb["mean_corr_2020_deseasonalised_changes"], nb["mean_corr_2020"]
    v.update(T15=f"{q['T15']:,}", flow_mean=f"{fs['mean']:.0f}", flow_median=f"{fs['median']:.0f}",
             flow_max=f"{fs['max']:.0f}", flow_train=f"{fs['mean_2020']:.0f}", flow_test=f"{fs['mean_test_period']:.0f}",
             m5=pct(m["frac_5min_readings_missing"]), m15=pct(m["frac_15min_bins_all_missing"]),
             m15_2020=pct(m["by_year_15min"]["2020"]), m15_2021=pct(m["by_year_15min"]["2021"]),
             m_partial=pct(m["frac_15min_bins_partial"], 3), never=m["sensors_never_observed"],
             never21=m["sensors_no_data_2021"], bad_stats=m["sensors_no_data_2020"],
             over50=m["sensors_over_50pct_missing"], sens_lt5=m["sensors_under_5pct_missing"],
             sens_ge60=m["sensors_60pct_or_more_missing"], allmiss_t=len(m["timestamps_all_sensors_missing"]),
             short_gap_frac=pct(m["missing_bins_in_gaps_le_4_frac"]),
             adj_nodes=a["nodes"], adj_nnz=f"{a['nonzero_total']:,}", adj_edges=f"{a['nonzero_offdiag_edges']:,}",
             adj_density=f"{a['density_offdiag']:.4f}", adj_recip=pct(a["reciprocated_edge_frac"], 1),
             adj_asym=f"{a['max_abs_asym']:.2f}", adj_wmin=f"{a['weight_min_nonzero']:.3f}",
             adj_wmax=f"{a['weight_max']:.3f}", deg_mean=f"{a['out_degree']['mean']:.1f}",
             deg_med=f"{a['out_degree']['median']:.0f}", deg_max=a["out_degree"]["max"], deg_iso=a["isolated_out"],
             nb_id=nb["sensor_id"], nb_fwy=nb["fwy"], nb_one=nb["one_hop"], nb_two=nb["two_hop_total"],
             nb_two_only=nb["two_hop_only"], nb_fwys=", ".join(nb["two_hop_fwys"]),
             nb_one_ids=", ".join(nb["one_hop_ids"]), nb_two_ids=", ".join(nb["two_hop_only_ids"]),
             corr1=f"{dc['one_hop']:.2f}", corr2=f"{dc['two_hop_only']:.2f}", corr3=f"{dc['beyond_two_hop']:.2f}",
             rawcorr1=f"{rc['one_hop']:.2f}", rawcorr3=f"{rc['beyond_two_hop']:.2f}")

    q = load("q6.json")
    seeds = q["seeds"]
    v.update(seeds=", ".join(map(str, seeds)), nseeds=len(seeds), seed0=seeds[0],
             table_main=open(os.path.join(ROOT, "results", "table_main.md")).read(),
             table_skip=open(os.path.join(ROOT, "results", "table_skip.md")).read())
    rows = {r["graph"]: r for r in q["rows"]}
    skip = {r["graph"]: r for r in q["skip_rows"]}
    mae = {g: np.mean(r["test_MAE"]) for g, r in rows.items()}
    for g, r in rows.items():
        for k in ("test_MAE", "test_RMSE", "val_MAE", "val_RMSE"):
            v[f"{g}_{k}"] = f"{np.mean(r[k]):.2f}"
        v[f"{g}_time"] = f"{r['train_time_sec'][0] / 60:.1f}"
        v[f"{g}_spe"] = f"{r['sec_per_epoch'][0]:.0f}"
    v["fixed_loss"] = f"{100 * (mae['fixed'] - mae['none']) / mae['none']:.0f}"
    v["adaptive_vs_fixed"] = f"{100 * (mae['fixed'] - mae['adaptive']) / mae['fixed']:.0f}"
    v["skip_fixed_gain"] = f"{100 * (mae['none'] - np.mean(skip['fixed']['test_MAE'])) / mae['none']:.1f}"
    def span(x):
        return f"{min(x)}" if min(x) == max(x) else f"{min(x)} to {max(x)}"
    v["epochs_note"] = ", ".join(f"{'temporal-only' if g == 'none' else r['adjacency'].lower()} {span(r['epochs_run'])}"
                                 for g, r in rows.items())
    alphas = rows["hybrid"]["alpha"]
    v["alpha"] = f"{np.mean(alphas):.2f}"
    v["alpha_list"] = ", ".join(f"{x:.2f}" for x in alphas)

    ha, aa = np.array(rows["hybrid"]["test_MAE"]), np.array(rows["adaptive"]["test_MAE"])
    diff, spread = ha.mean() - aa.mean(), max(ha.std(), aa.std())
    if abs(diff) <= 2 * spread:
        v["hybrid_vs_adaptive"] = (f"the difference in test MAE ({diff:+.2f}) is within the seed spread of about "
                                   f"{spread:.2f}")
        v["hybrid_vs_adaptive_short"] = "within seed variation of the adaptive model"
    else:
        word = "lower" if diff < 0 else "higher"
        v["hybrid_vs_adaptive"] = (f"the hybrid has {abs(diff):.2f} {word} test MAE than the adaptive graph, more than "
                                   f"the seed spread of {spread:.2f}")
        v["hybrid_vs_adaptive_short"] = f"{'slightly better' if diff < 0 else 'slightly worse'} than the adaptive model"

    fv = q["fixed_vs_none"]
    v["fixed_worse_frac"] = pct(1 - fv["frac_sensors_better"], 0)
    lab = {"0-0": "0", "1-10": "1 to 10", "11-25": "11 to 25", "26-70": "26 to 70"}
    v["gain_by_degree"] = "\n".join(f"| {lab[k]} | {x['n']} | {-x['mean_gain_pct']:+.1f}% |"
                                    for k, x in q["gain_by_degree"].items())
    sw = q["self_weight"]
    v.update(self_w=pct(sw["fixed_mean"], 0), self_w_adapt=f"{sw['adaptive_mean']:.3f}",
             self_w_uniform=f"{sw['uniform']:.4f}")

    z = q["zoom"]
    v.update(zoom_sensor=z["sensor"], zoom_0745=f"{z['flow_0745']:.0f}", zoom_0800=f"{z['flow_0800']:.0f}",
             zoom_0815=f"{z['flow_0815']:.0f}", zoom_0900=f"{z['actual_0900']:.0f}",
             **{f"zoom_p_{g}": f"{x:.0f}" for g, x in z["pred_0900"].items()})

    ga = q["graph_adaptive"]
    v.update(ga_pairs=f"{ga['pairs']:,}", ga_road_pct=pct(ga["frac_top_in_road_graph"], 1),
             ga_base_pct=pct(ga["road_density_baseline"], 1), ga_top10_mass_pct=pct(ga["row_top10_mass_mean"], 0),
             ga_same_fwy_top=pct(ga["same_fwy_top"], 1), ga_same_fwy_random=pct(ga["same_fwy_random"], 1),
             ga_hubs=", ".join(f"{h['id']} on {h['fwy']} is in the top 10 of {h['in_top_edges']} sensors"
                               for h in ga["hubs"]))
    for k in ("median_km_top", "median_km_road_edges", "median_km_random"):
        v[f"ga_{k}"] = f"{ga[k]:.1f}"
    for k in ("corr_top", "corr_road_edges", "corr_random", "profile_corr_top", "profile_corr_random"):
        v[f"ga_{k}"] = f"{ga[k]:.3f}"
    if "seed_top10_overlap" in q:
        v["seed_overlap_sentence"] = (f"on average only {100 * q['seed_top10_overlap']:.0f}% of a sensor's top-10 "
                                      f"learned neighbours are shared between two seeds, and the weights correlate at "
                                      f"{q['seed_weight_corr']:.2f}")
    else:
        v["seed_overlap_sentence"] = "compare the seeds to check"

    run = json.load(open(os.path.join(ROOT, "results", "runs", f"fixed_s{seeds[0]}.json")))
    sp, hw = run["splits"], run["hardware"]
    for k in ("train", "val", "test"):
        v[f"n_{k}"] = f"{sp[k]['windows']:,}"
        v[f"{k}_first"], v[f"{k}_last"] = sp[k]["first_input"], sp[k]["last_target"]
    v["hw"] = (f"{hw.get('cpu', 'unknown CPU')}, {hw.get('logical_cpus', '?')} logical CPUs, {hw.get('mem_gb', '?')} GB RAM, "
               f"PyTorch {hw['torch']} on CPU ({hw['threads']} threads), Python {hw['python']}, {hw['platform']}")

    vc = load("verify_checkpoints.json")
    assert vc["all_match"], "a checkpoint does not reproduce its logged metrics"
    v["n_ckpt"] = sum(1 for k in vc if k != "all_match")
    v["n_tests"] = sum(len(re.findall(r"^def test_", open(f).read(), re.M))
                       for f in glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    return v


def main():
    v = values()
    text = "\n\n".join(open(os.path.join(HERE, p)).read() for p in PARTS)

    def sub(mt):
        if mt.group(1) not in v:
            raise KeyError(f"no value for {{{{{mt.group(1)}}}}}")
        return str(v[mt.group(1)])

    filled = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", sub, text)
    em_dash, en_dash, semicolon = chr(0x2014), chr(0x2013), chr(59)
    assert em_dash not in filled and en_dash not in filled, "dash character found in the report text"
    prose = re.sub(r"```.*?```", "", filled, flags=re.S)       # code blocks may contain shell syntax
    assert semicolon not in prose, "semicolon found in the report text"
    if "EDITS_PLACEHOLDER" in filled:
        print("WARNING: the AI usage section still has EDITS_PLACEHOLDER. Fill it in before submitting.")
    os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
    md = os.path.join(HERE, "build", "report.md")
    open(md, "w").write(filled)
    pdf = os.path.join(HERE, "DL4STG_PA3_report.pdf")
    header = ("\\usepackage{float}\\floatplacement{figure}{H}\\usepackage{etoolbox}"
              "\\AtBeginEnvironment{longtable}{\\small}\\renewcommand{\\abstractname}{Code}")
    cmd = ["pandoc", md, "-o", pdf, "--pdf-engine=xelatex", "--resource-path", ROOT,
           "-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
           "-V", "mainfont=DejaVu Serif", "-V", "monofont=DejaVu Sans Mono", "--toc", "--toc-depth=2",
           "-V", f"header-includes={header}"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
