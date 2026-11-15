# Question 4: Spatio-temporal traffic forecasting

## Q4(a) Forecasting windows

For each prediction time $t$ the input is $X_t = [x_{t-7},\dots,x_t] \in \mathbb R^{8\times 716}$ (2 hours) and the target is $Y_t = [x_{t+1},\dots,x_{t+4}]\in\mathbb R^{4\times 716}$ (1 hour). A window belongs to a split only if **all 12 of its timestamps** lie inside that split, so no window crosses a boundary. The few windows that would straddle 31 Dec 2020 or 31 Mar 2021 are dropped.

| Split | Period | Windows | First input step | Last target step |
|---|---|---|---|---|
| Train | Jan to Dec 2020 | {{n_train}} | {{train_first}} | {{train_last}} |
| Validation | Jan to Mar 2021 | {{n_val}} | {{val_first}} | {{val_last}} |
| Test | Apr to Dec 2021 | {{n_test}} | {{test_first}} | {{test_last}} |

**Normalisation.** Each sensor's input is z-scored with the mean and standard deviation of its observed 2020 values only, $\tilde x = (x-\mu_i)/\sigma_i$. Validation and test data are never used for any statistic. Models predict in normalised units and the prediction is mapped back with $\hat y = \sigma_i \hat{\tilde y} + \mu_i$ before the loss, so the loss and all metrics are in the original units (vehicles per 15 minutes). Missing inputs and targets are handled as described in Q3(a).

## Q4(b) Architecture

All models share the same three components (code in `pa3/models.py`), with hidden size $d = 32$.

1. **Temporal encoder** $Z = f_{\text{temporal}}(X)$. Two 1D convolutions over time, kernel 3, no padding, ReLU, weights shared by all sensors ($1\to 32 \to 32$ channels, sequence length $8 \to 6 \to 4$). The remaining $4\times 32$ features of each sensor are flattened and projected to $d=32$ by a linear layer with ReLU. Output $Z\in\mathbb R^{B\times N\times d}$.
2. **Spatial graph convolution** $H = \mathrm{ReLU}(\hat A Z W_g)$, with $W_g\in\mathbb R^{d\times d}$. The product with $\hat A$ is taken over the node dimension for each batch item (`einsum("ij,bjd->bid")`).
3. **Forecast head** $\hat Y = f_{\text{head}}(Z + H)$: a two-layer MLP ($32\to 32\to 4$, ReLU) applied to every sensor, giving the 4 future steps.

The head receives $Z + H$ (a residual connection) instead of $H$ alone. With this, the temporal-only model of Q4(c) is exactly the same network with the graph term removed, and each graph variant adds only $\hat A Z W_g$ on top of it. It also keeps each sensor's own temporal features available when a learned $\hat A$ puts little weight on the diagonal.

**Training protocol (identical for every model).** Masked MAE loss on observed targets in original units, Adam (learning rate $10^{-3}$), batch size 64, gradient clipping at norm 5, at most 20 epochs, early stopping when validation MAE has not improved for 4 epochs, and the checkpoint with the best validation MAE is evaluated on test. Seeds {{seeds}} (`torch`, `numpy` and `random` seeded, and the same seed fixes the shuffling order). Everything runs on CPU.

**Metrics** on observed targets, averaged over all sensors, test windows and the 4 horizons:
$$\text{MAE} = \frac{1}{|\Omega|}\sum_{(t,h,i)\in\Omega}|\hat y_{t,h,i}-y_{t,h,i}|,\qquad \text{RMSE} = \sqrt{\frac{1}{|\Omega|}\sum_{(t,h,i)\in\Omega}(\hat y_{t,h,i}-y_{t,h,i})^2}.$$

## Q4(c) Temporal-only baseline vs the graph model

The temporal-only model has the same encoder and head and no graph term ({{none_params}} parameters, against {{fixed_params}} for the fixed-graph STGNN). Results (mean over {{nseeds}} seeds, vehicles per 15 minutes):

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---|---|---|---|
| Temporal-only | {{none_val_MAE}} | {{none_val_RMSE}} | {{none_test_MAE}} | {{none_test_RMSE}} |
| STGNN, fixed road graph | {{fixed_val_MAE}} | {{fixed_val_RMSE}} | {{fixed_test_MAE}} | {{fixed_test_RMSE}} |

Explicitly modelling spatial dependencies helps. Adding one graph convolution over the road graph lowers test MAE by {{fixed_gain}}% and RMSE by a similar margin, and in seed {{seed0}} the fixed-graph model has lower error than the temporal-only model on {{fixed_better_frac}} of the sensors with test data (median per-sensor change {{fixed_med_rel}}%). The temporal-only model also reaches a higher training error, so the gap is not overfitting: one sensor's own last two hours do not contain the information needed. A slowdown that is about to reach a sensor is already visible at its upstream neighbours, which is the mechanism shown on the ring road in Q2. The improvement depends on how many neighbours a sensor has (fixed vs temporal-only, test MAE, seed {{seed0}}):

| Road-graph neighbours | Sensors with test data | Mean MAE reduction |
|---|---|---|
{{gain_by_degree}}

The gain grows with the number of road neighbours. Sensors with no road neighbours do not benefit ({{gain_deg0}}): for them $\hat A$ is the identity, so the graph term only adds another transform of the sensor's own features and no new information.

# Question 5: Fixed versus learnable adjacency

All three variants use the same encoder, head, splits and protocol as Q4. Only $\hat A$ changes.

## Q5(a) Fixed adjacency

The SD road-distance matrix is directed (Q3(b)), so the symmetric $D^{-1/2}(A+I)D^{-1/2}$ does not apply. We use the random-walk (row) normalisation
$$\hat A_{\text{fixed}} = D^{-1}(A + I),\qquad D_{ii} = \sum_j (A+I)_{ij},$$
after setting the diagonal of $A$ to zero (LargeST stores 1 there) and adding $I$. Row $i$ of $\hat A Z$ is then a weighted average of $Z_i$ and the $Z_j$ with $A_{ij}\neq 0$, so the scale of $Z$ is preserved, and messages follow the direction of the stored edges ($j\to i$ when $A_{ij}\neq 0$). $\hat A_{\text{fixed}}$ is a constant buffer, never updated.

Result: validation MAE {{fixed_val_MAE}}, RMSE {{fixed_val_RMSE}}. Test MAE {{fixed_test_MAE}}, RMSE {{fixed_test_RMSE}}, training time {{fixed_time}} min ({{fixed_spe}} s per epoch).

## Q5(b) Learnable adjacency

Two embedding matrices $E_1, E_2\in\mathbb R^{716\times 10}$ ($d_e = 10$) are initialised from $\mathcal N(0,1)$ and trained by backpropagation with the rest of the model:
$$\hat A_{\text{adaptive}} = \mathrm{Softmax}_{\text{row}}\big(\mathrm{ReLU}(E_1E_2^\top)\big).$$
Each row sums to 1, like $\hat A_{\text{fixed}}$. Nothing restricts it to road neighbours, so any sensor can attend to any other. This adds $2\times 716\times 10 = 14{,}320$ parameters.

**Sparsification.** We did **not** sparsify. For 716 nodes the dense matrix has 513k entries, and building it and multiplying by $Z$ costs about the same as the fixed graph (per-epoch times in Q5(c)). The code supports top-$k$ (`--topk 20`): it would keep the $k$ largest logits of each row and set the others to $-\infty$ **before** the softmax, so the kept weights still sum to 1. Gradients would then flow only through the kept entries (and the embeddings that produce them). The top-$k$ selection itself is not differentiable, and an entry outside the top $k$ only re-enters when its logit grows through the gradients of other entries that share its embeddings.

Result: validation MAE {{adaptive_val_MAE}}, RMSE {{adaptive_val_RMSE}}. Test MAE {{adaptive_test_MAE}}, RMSE {{adaptive_test_RMSE}}, training time {{adaptive_time}} min, a {{adaptive_gain}}% MAE reduction over temporal-only, compared with {{fixed_gain}}% for the road graph.

![Left: fixed $D^{-1}(A+I)$, sparse and banded along the freeways. Middle: learned adaptive adjacency, dense with no diagonal band. The bright vertical stripes are hub sensors that many rows attend to. Right: adaptive part of the hybrid model, with the same character. Log colour scale, seed {{seed0}}.](figures/q6_adjacency_learned.png){width=100%}

## Q5(c) Hybrid adjacency

$$\hat A_{\text{hybrid}} = \alpha\,\hat A_{\text{fixed}} + (1-\alpha)\,\hat A_{\text{adaptive}},\qquad \alpha = \mathrm{sigmoid}(\theta).$$
Both parts are row-stochastic, so they are on the same scale and the mixture is row-stochastic too. $\alpha$ is learned. It starts at 0.5 ($\theta=0$) and ended at $\alpha = {{alpha}}$ (per seed: {{alpha_list}}).

Result: validation MAE {{hybrid_val_MAE}}, RMSE {{hybrid_val_RMSE}}. Test MAE {{hybrid_test_MAE}}, RMSE {{hybrid_test_RMSE}}, training time {{hybrid_time}} min.

**Comparison of the three graphs.**

| Adjacency | Test MAE | Test RMSE | MAE reduction vs temporal-only | Training time |
|---|---|---|---|---|
| Fixed (road distance) | {{fixed_test_MAE}} | {{fixed_test_RMSE}} | {{fixed_gain}}% | {{fixed_time}} min |
| Adaptive (learned) | {{adaptive_test_MAE}} | {{adaptive_test_RMSE}} | {{adaptive_gain}}% | {{adaptive_time}} min |
| Hybrid, learned $\alpha$ | {{hybrid_test_MAE}} | {{hybrid_test_RMSE}} | {{hybrid_gain}}% | {{hybrid_time}} min |
| Hybrid, fixed $\alpha = 0.5$ (seed {{seed0}} only) | {{a05_test_MAE}} | {{a05_test_RMSE}} | {{a05_gain}}% | {{a05_time}} min |

The learned graph clearly beats the road graph. The hybrid with learned $\alpha$ is {{hybrid_vs_adaptive}}. The reason is visible in $\alpha$ itself: training drives it from 0.5 down to about {{alpha}}, so the hybrid ends up using almost only its adaptive part. Given the choice, gradient descent decides the road graph is not worth much weight once a learned graph is available. Forcing $\alpha = 0.5$ {{a05_sentence}}

Per epoch, the graph models are about 20% slower than the temporal-only model ({{none_spe}} s vs {{fixed_spe}} to {{hybrid_spe}} s), mainly from the $716\times716$ product with a batch of $64\times716\times32$ features. The adaptive and hybrid graphs are no slower than the fixed graph, because the fixed graph is also stored densely. Their total training time is about three times longer only because the temporal-only model stopped early after about 9 epochs, while the graph models used all 20.

# Question 6: Experimental analysis and interpretation

## Q6(a) Forecasting comparison

All models: same encoder, head, splits, protocol and test period (Apr to Dec 2021). Errors in vehicles per 15 minutes. {{table_note}}

{{table_main}}

**Sanity baselines** (not trained, same masks and test windows): repeating the last observed value for all 4 steps gives test MAE {{naive_persist_mae}} (RMSE {{naive_persist_rmse}}), and the 2020 average for the same time of week gives {{naive_tow_mae}} ({{naive_tow_rmse}}). Every trained model beats both. The time-of-week average is especially poor on the test period because 2020 traffic, shaped by the lockdown, does not match 2021.

Test MAE / RMSE by horizon:

{{table_horizon}}

{{table_seeds_block}}

The temporal-only model stops early after about 9 epochs, because its validation error stops improving. The graph models were still improving slightly at the 20-epoch cap (8 of the 9 graph runs reached their best validation error at epoch 18 to 20). Longer training would lower their error a little further, so the gaps in the table are, if anything, conservative.

![Validation MAE per epoch.](figures/q6_curves.png){width=70%}

![Predicted vs actual flow, 15 minutes ahead, four test sensors, 14 to 16 June 2021.](figures/q6_pred_vs_actual_15min.png){width=100%}

![Predicted vs actual flow, 60 minutes ahead, same sensors and days.](figures/q6_pred_vs_actual_60min.png){width=100%}

![Zoom on one morning peak, 60 minutes ahead.](figures/q6_zoom.png){width=80%}

All four models follow the daily cycle closely, and most of the remaining error is at the peaks and at sudden changes. The plots show two typical differences.

* **Morning ramp.** At the 60-minute horizon the temporal-only and fixed-graph forecasts lag the weekday morning ramp-up and undershoot it. Over the {{ramp_n}} sensors whose predictions we saved, 60-minute forecasts for weekday targets between 05:00 and 09:00 (seed {{seed0}}) have MAE {{ramp_none}} and mean error {{ramp_none_bias}} for temporal-only, {{ramp_fixed}} and {{ramp_fixed_bias}} for the fixed graph, but {{ramp_adaptive}} and {{ramp_adaptive_bias}} for the adaptive graph and {{ramp_hybrid}} and {{ramp_hybrid_bias}} for the hybrid. The per-horizon table shows the same pattern for all sensors: the advantage of the learned graph grows with the horizon (15 min: {{none_h1}} vs {{adaptive_h1}}, 60 min: {{none_h4}} vs {{adaptive_h4}}).
* **Isolated drops.** In the zoom, sensor {{zoom_sensor}} drops from {{zoom_0745}} to {{zoom_0800}} vehicles at 08:00 and is back at {{zoom_0815}} at 08:15, a one-step dip no model can foresee. The forecast for 09:00 (actual {{zoom_0900}}), issued at 08:00 with the dip as the most recent input, shows the difference. The temporal-only model repeats the dip one hour later ({{zoom_p_none}}), the fixed graph partly does ({{zoom_p_fixed}}), the hybrid less so ({{zoom_p_hybrid}}), and the adaptive model stays close to the true value ({{zoom_p_adaptive}}). With information from other sensors that did not dip, the graph models can treat it as a local, short-lived event.

Test errors are higher than validation errors for every model. The test period (April to December 2021) has much higher traffic than the training year (mean flow {{flow_test}} vs {{flow_train}} vehicles per 15 minutes, because 2020 included the COVID-19 lockdown), and absolute errors grow with flow. The ranking of the models is the same on validation and test, except that the adaptive and hybrid models, which are within seed variation of each other, swap places.

## Q6(b) Graph structure analysis

For each sensor we take its 10 strongest learned out-neighbours (largest $\hat A_{ij}$, $j\neq i$) from the adaptive model, seed {{seed0}} ({{ga_pairs}} pairs between sensors with 2020 data). "Change correlation" is the correlation of the 15-minute changes of two sensors' deviations from their weekly profiles in 2020 (Q3(c)). It measures shared short-term fluctuations, not the shared daily cycle.

| Measure | Top-10 learned edges | Road-graph edges | Random pairs |
|---|---|---|---|
| Share that are road-graph edges | {{ga_road_pct}} | 100% | {{ga_base_pct}} |
| Median straight-line distance (km) | {{ga_median_km_top}} | {{ga_median_km_road_edges}} | {{ga_median_km_random}} |
| Within 5 km | {{ga_frac_top_within_5km}} | | {{ga_frac_random_within_5km}} |
| Same freeway and direction | {{ga_same_fwy_top}} | | {{ga_same_fwy_random}} |
| Mean change correlation | {{ga_corr_top}} | {{ga_corr_road_edges}} | {{ga_corr_random}} |

![Left: strongest learned out-neighbour of every sensor, blue if it is also a road-graph edge. Right: distances of road-graph edges, top-10 learned edges and random pairs.](figures/q6_learned_edges.png){width=100%}

**1. Are the strongest learned connections also road-graph edges?** Almost never. Only {{ga_road_pct}} of the top-10 learned edges are road-graph edges, close to the {{ga_base_pct}} expected for random pairs. The middle panel of the adjacency figure looks nothing like the banded road graph. The weights are concentrated (the 10 strongest entries of a row hold {{ga_top10_mass_pct}} of its weight on average, against $10/716 = 0.014$ for a uniform row) and they go to a few hub sensors: the three most-attended sensors are {{ga_hubs}}, where the number in brackets is how many sensors have that hub among their 10 strongest connections (about 10 would be expected at random).

**2. Are learned connections geographically close?** No. The median distance of a top-10 learned edge is {{ga_median_km_top}} km, almost the same as for random pairs ({{ga_median_km_random}} km), while road-graph edges are {{ga_median_km_road_edges}} km apart. Only {{ga_frac_top_within_5km}} of learned edges are within 5 km (random: {{ga_frac_random_within_5km}}). The distance histogram of learned edges follows the random-pair curve.

**3. Does the learned graph capture dependencies absent from the physical graph?** It uses almost only connections absent from the road graph ({{ga_road_pct}} overlap), and those connections help: the adaptive model has {{adaptive_gain}}% lower test MAE than temporal-only, against {{fixed_gain}}% for the road graph. But they are not pairs of sensors whose short-term fluctuations move together. Their change correlation ({{ga_corr_top}}) is barely above random pairs ({{ga_corr_random}}) and far below road-graph edges ({{ga_corr_road_edges}}). Across all pairs, the learned weight and the change correlation correlate at only {{ga_pearson_weight_vs_corr_all_pairs}}. Learned edges are slightly more likely to join two sensors on the same freeway and direction ({{ga_same_fwy_top}} vs {{ga_same_fwy_random}} for random pairs), and to join sensors with similar weekly profiles (profile correlation {{ga_profile_corr_top}} vs {{ga_profile_corr_random}}), so they partly reflect shared commuting patterns. The adaptive part of the hybrid model learned a graph with the same properties ({{gh_road_pct}} road overlap, median {{gh_median_km_top}} km).

Our interpretation is that the learned graph acts mostly as a **network-wide context channel**, not as a set of local links. Each sensor reads a weighted summary of a common set of hub sensors. That summary tells the model where in the daily and weekly cycle the network is, and whether traffic is generally heavier or lighter than usual. The model has no time-of-day or day-of-week input, and two hours of one sensor's history often cannot tell these apart, which matters most at the longer horizons. This is consistent with the larger gains at 60 minutes and with how little the learned edges care about distance. The road graph only offers local averages, which carry real short-term information (high change correlation) but no global context.

**4. Why can a learned graph do better or worse than the fixed graph?**

* *Better:* it is optimised for the forecasting loss, so it can pick whatever relationships help. These include long-range and global ones that a distance threshold excludes, such as context from hubs or similar commuting corridors. It can also down-weight road links that carry little signal, such as the opposite-carriageway links seen in Q3(c), and it is not limited to the 0.01 threshold on driving distance.
* *Worse:* it adds $2Nd_e$ parameters and must learn from data what the road graph gives for free. This needs enough data. With little data, or for sensors that are rarely observed, it can overfit spurious correlations. It also has no inductive bias towards locality, so it may miss propagation patterns that are rare in training (incidents). Finally, it is a single static matrix learned from 2020, a year whose traffic was unusual, so its relations may not transfer to 2021.

## Q6(c) Critical discussion

**Does lower error mean the learned graph represents road connectivity?** No, and our results show it directly. The adaptive model has the lowest error, yet its strongest connections are essentially unrelated to the road network: their overlap with road edges and their distances are close to those of random pairs (Q6(b)). Forecast error measures how useful a matrix is to *this* model for *this* loss, not whether it is physically true. The softmax graph is an attention pattern that can encode global context, shared daily profiles, or compensation for features the model lacks (here, time of day). Many different matrices, including ones with no physical meaning, can give similar error, and different seeds learn different matrices ({{seed_overlap_sentence}}). Reading a learned graph as a road map needs separate evidence, such as overlap with known links, consistency across seeds, or a causal test such as cutting an edge and checking the effect.

**Why might a hybrid outperform either alone?** The two graphs carry complementary information. The road graph gives a reliable local prior, the short-range propagation that is physically guaranteed and has the highest change correlation, for free. The adaptive graph adds long-range and context relations the distance threshold cannot express. A mixture can use both, and the road prior can also regularise the learned part, which helps when data is limited, for sensors with few observations, and early in training. In our runs the learned $\alpha$ went to about {{alpha}} and the hybrid ended up {{hybrid_vs_adaptive_short}}: with a full year of training data and a residual skip, the adaptive graph learned enough on its own that the road prior added little. Forcing $\alpha = 0.5$ was worse than letting the model choose (Q5(c)). A hybrid is more likely to win with less data, larger graphs, or models with several graph layers where a good local prior matters more.

**Limits of a static graph.** Traffic dependencies change with time of day, day of week, season, incidents, roadworks and long-term shifts (2020 lockdown vs 2021). In free flow, neighbouring sensors are only weakly coupled. In congestion, a queue spreads upstream and couples sensors kilometres apart, and the direction and speed of that spread change during the day. A single $\hat A$, fixed or learned, must average all of these regimes into one compromise matrix. It cannot express "use the upstream sensor during the evening peak but not at night", and it cannot react to an incident that cuts a link. Our learned graph, fit to 2020 traffic, is then applied unchanged to the busier 2021 test period. Remedies include time-dependent or input-dependent adjacency (attention computed from the current window, as in DGCRN or ASTGCN), adjacency conditioned on time-of-day embeddings, several graphs mixed with time-varying weights, or periodic re-estimation of the graph. All of these cost more computation and are harder to interpret.

# Question 7: Reproducibility and reporting

**Code:** {{github}}. Layout:

| Path | Content |
|---|---|
| `scripts/q1_q2.py` | Q1 and Q2 checks and heatmaps |
| `scripts/prepare_data.py` | extracts SD, 2020 and 2021, 15-minute sums, adjacency submatrix |
| `scripts/q3_analysis.py` | missing values, weekly plot, graph statistics, neighbourhoods |
| `pa3/data.py`, `pa3/models.py`, `pa3/train.py` | windows and splits, models, training and evaluation |
| `run_all.sh` | trains the four models for the given seeds |
| `scripts/analyze.py` | Q6 tables, plots and graph analysis |
| `scripts/naive_baselines.py`, `scripts/verify_checkpoints.py` | sanity baselines, checkpoint re-evaluation |
| `tests/` | {{n_tests}} automated tests (see below) |
| `report/build.py` | fills every number in this report from `results/` and builds the PDF |
| `results/runs/*.json`, `logs/` | per-run configuration, per-epoch logs, metrics, hardware, timings |

**Reproduce.**

```
./setup_env.sh && source .venv/bin/activate
# LargeST files from Kaggle (liuxu77/largest): ca_his_raw_2020.h5, ca_his_raw_2021.h5,
# ca_meta.csv, ca_rn_adj.npy, renamed to 2020.h5, 2021.h5, metadata.csv, adjacency.npy in data/raw/
python scripts/prepare_data.py --raw_dir data/raw
python scripts/q1_q2.py && python scripts/q3_analysis.py
./run_all.sh {{seeds_space}}
python -m pa3.train --graph hybrid --alpha 0.5 --seed 0 --tag _a05    # fixed-alpha ablation
python scripts/analyze.py --seeds {{seeds_space}}
python scripts/naive_baselines.py
python scripts/verify_checkpoints.py      # re-evaluates every checkpoint against the logged metrics
PA3_RAW_DIR=data/raw python -m pytest tests
python report/build.py
```

**Hyperparameters.** $d = 32$, temporal kernel 3 (two layers), one graph layer, head $32\to 32\to 4$, $d_e = 10$, no top-$k$, hybrid $\alpha$ learned (initial 0.5), Adam with learning rate $10^{-3}$, batch 64, gradient clipping 5, at most 20 epochs, patience 4, masked MAE loss, per-sensor z-score from 2020, input forward-fill limit 4 steps. Seeds {{seeds}}.

**Validation.** The test suite (`pytest tests`, {{n_tests}} tests, all passing) independently re-checks the Q1 and Q2 answers, compares {{n_resample_checks}} resampled values with the raw 5-minute files, checks that the sensor order of the flow tensor and the adjacency matches the LargeST metadata, that no window crosses a split boundary, that replacing all 2021 data with random numbers leaves the normalisation statistics and every 2020 input unchanged (no leakage), that forward filling only uses past values, that $\hat A$ is row-stochastic and the fixed graph is never updated, that the temporal-only model is exactly the graph model with $W_g=0$, that the masked metrics match a direct NumPy computation, and that training is bit-for-bit reproducible for a fixed seed. `scripts/verify_checkpoints.py` reloads all {{n_ckpt}} saved checkpoints and reproduces every logged validation and test MAE and RMSE to within $10^{-3}$.

**Hardware.** {{hw}}. No GPU was used. Training times are wall-clock time of the training epochs only (validation excluded), as listed in Q6(a).

**Data provenance.** LargeST CA release (CC BY-NC 4.0), files obtained from the HuggingFace mirror {{mirror}}, SHA-256 verified against the mirror. The SD graph statistics (716 nodes, 17,319 edges) match the LargeST paper.
