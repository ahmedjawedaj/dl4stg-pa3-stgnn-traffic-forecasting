# Question 4: Spatio-temporal traffic forecasting

## Q4(a) Forecasting windows

For each prediction time $t$ the input is $X_t = [x_{t-7},\dots,x_t] \in \mathbb R^{8\times 716}$ (2 hours) and the target is $Y_t = [x_{t+1},\dots,x_{t+4}]\in\mathbb R^{4\times 716}$ (1 hour). A window belongs to a split only if **all 12 of its timestamps** lie inside that split, so no window crosses a boundary.

| Split | Period | Windows | First input step | Last target step |
|---|---|---|---|---|
| Train | Jan to Dec 2020 | {{n_train}} | {{train_first}} | {{train_last}} |
| Validation | Jan to Mar 2021 | {{n_val}} | {{val_first}} | {{val_last}} |
| Test | Apr to Dec 2021 | {{n_test}} | {{test_first}} | {{test_last}} |

**Normalisation.** Each sensor is z-scored with the mean and standard deviation of its observed 2020 values only, $\tilde x = (x-\mu_i)/\sigma_i$. Models predict in normalised units, and the prediction is mapped back with $\hat y = \sigma_i \hat{\tilde y} + \mu_i$ before the loss, so the loss and all metrics are in vehicles per 15 minutes. Missing values are handled as described in Q3(a).

## Q4(b) Architecture

All models are in `pa3/models.py`, with hidden size $d = 32$.

1. **Temporal encoder** $Z = f_{\text{temporal}}(X)$: two 1D convolutions over time (kernel 3, no padding, ReLU, $1\to 32 \to 32$ channels, length $8 \to 6 \to 4$), shared by all sensors. The remaining $4\times 32$ features of each sensor are flattened and projected to $d$ by a linear layer with ReLU, giving $Z\in\mathbb R^{B\times N\times d}$.
2. **Spatial graph convolution** $H = \mathrm{ReLU}(\hat A Z W_g)$ with $W_g\in\mathbb R^{d\times d}$. The product with $\hat A$ runs over the node dimension for each batch item (`einsum("ij,bjd->bid")`).
3. **Forecast head** $\hat Y = f_{\text{head}}(H)$: a two-layer MLP ($32\to 32\to 4$, ReLU) applied to every sensor.

**Training protocol (identical for every model).** Masked MAE loss on observed targets, Adam (learning rate $10^{-3}$), batch size 64, gradient clipping at norm 5, at most 20 epochs, early stopping after 4 epochs without a better validation MAE, and the checkpoint with the best validation MAE is evaluated on test. Each model is trained with seeds {{seeds}} (`torch`, `numpy` and `random` seeded, and the seed also fixes the shuffling order). Everything runs on CPU.

**Metrics** on observed targets, over all sensors, test windows and the 4 horizons:
$$\text{MAE} = \frac{1}{|\Omega|}\sum_{(t,h,i)\in\Omega}|\hat y_{t,h,i}-y_{t,h,i}|,\qquad \text{RMSE} = \sqrt{\frac{1}{|\Omega|}\sum_{(t,h,i)\in\Omega}(\hat y_{t,h,i}-y_{t,h,i})^2}.$$

## Q4(c) Temporal-only baseline vs the graph model

The temporal-only model uses the same encoder and head with no graph layer, $\hat Y = f_{\text{head}}(Z)$. Results are means over {{nseeds}} seeds, in vehicles per 15 minutes:

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---|---|---|---|
| Temporal-only | {{none_val_MAE}} | {{none_val_RMSE}} | {{none_test_MAE}} | {{none_test_RMSE}} |
| STGNN, fixed road graph | {{fixed_val_MAE}} | {{fixed_val_RMSE}} | {{fixed_test_MAE}} | {{fixed_test_RMSE}} |

**With the specified architecture, the graph layer makes forecasts worse.** The fixed-graph model has {{fixed_loss}}% higher test MAE and is worse than the temporal-only model on {{fixed_worse_frac}} of the sensors with test data (seed {{seed0}}). The loss grows with the number of road neighbours:

| Road-graph neighbours | Sensors with test data | Change in test MAE vs temporal-only |
|---|---|---|
{{gain_by_degree}}

The cause is that the head only sees $H$. With $\hat A = D^{-1}(A+I)$, row $i$ is an average over sensor $i$ and its neighbours, and the median sensor has {{deg_med}} neighbours, so sensor $i$ keeps on average only {{self_w}} of its own features. The forecast for sensor $i$ becomes mostly a forecast of its neighbourhood, which is the over-smoothing problem of Q3(c) after a single layer. Sensors with no neighbours, for which $\hat A$ is the identity, lose almost nothing.

**Spatial information does help when each sensor keeps its own features.** To show this, we also trained the same graph models with a skip connection, $\hat Y = f_{\text{head}}(Z + H)$, with everything else unchanged. This is not the specified architecture and is reported only for this comparison:

{{table_skip}}

With the skip, the road graph lowers test MAE by {{skip_fixed_gain}}% compared with the temporal-only model. So neighbours do carry useful information, as the correlations in Q3(c) suggest, but the graph term has to be added to the sensor's own features, not replace them.

# Question 5: Fixed versus learnable adjacency

All three variants use the same encoder, head, splits and protocol. Only $\hat A$ changes.

## Q5(a) Fixed adjacency

The SD road-distance matrix is directed (Q3(b)), so the symmetric $D^{-1/2}(A+I)D^{-1/2}$ does not apply. We use the row (random-walk) normalisation
$$\hat A_{\text{fixed}} = D^{-1}(A + I),\qquad D_{ii} = \sum_j (A+I)_{ij},$$
after setting the diagonal of $A$ to zero (LargeST stores 1 there). Each row sums to 1, so the scale of $Z$ is preserved, and messages follow the stored edges ($j\to i$ when $A_{ij}\neq 0$). $\hat A_{\text{fixed}}$ is a constant buffer, never updated.

Result: validation MAE {{fixed_val_MAE}}, RMSE {{fixed_val_RMSE}}. Test MAE {{fixed_test_MAE}}, RMSE {{fixed_test_RMSE}}. Training time {{fixed_time}} min ({{fixed_spe}} s per epoch).

## Q5(b) Learnable adjacency

Two embedding matrices $E_1, E_2\in\mathbb R^{716\times 10}$ ($d_e = 10$) are initialised from $\mathcal N(0,1)$ and trained by backpropagation with the rest of the model:
$$\hat A_{\text{adaptive}} = \mathrm{Softmax}_{\text{row}}\big(\mathrm{ReLU}(E_1E_2^\top)\big).$$
Each row sums to 1, like $\hat A_{\text{fixed}}$, and any sensor can attend to any other. This adds $2\times 716\times 10 = 14{,}320$ parameters. We did not sparsify: the dense $716\times 716$ matrix is cheap to build and use.

Result: validation MAE {{adaptive_val_MAE}}, RMSE {{adaptive_val_RMSE}}. Test MAE {{adaptive_test_MAE}}, RMSE {{adaptive_test_RMSE}}. Training time {{adaptive_time}} min.

The learned graph is clearly better than the road graph ({{adaptive_vs_fixed}}% lower test MAE), but still worse than the temporal-only model. Part of its advantage over the road graph is that it can learn how much weight each sensor gives itself: the mean diagonal weight grew to {{self_w_adapt}}, against {{self_w_uniform}} for a uniform row at the start. It was still improving at the 20-epoch cap.

![Left: fixed $D^{-1}(A+I)$, sparse and banded along the freeways. Right: learned adaptive adjacency, dense and without the diagonal band. Bright vertical stripes are hub sensors that many rows attend to. Log colour scale, seed {{seed0}}.](figures/q6_adjacency_learned.png){width=100%}

## Q5(c) Hybrid adjacency

$$\hat A_{\text{hybrid}} = \alpha\,\hat A_{\text{fixed}} + (1-\alpha)\,\hat A_{\text{adaptive}},\qquad \alpha = \mathrm{sigmoid}(\theta).$$
Both parts are row-stochastic, so they are on the same scale and the mixture is row-stochastic too. $\alpha$ is learned. It starts at 0.5 and ended at {{alpha_list}} for the three seeds.

Result: validation MAE {{hybrid_val_MAE}}, RMSE {{hybrid_val_RMSE}}. Test MAE {{hybrid_test_MAE}}, RMSE {{hybrid_test_RMSE}}. Training time {{hybrid_time}} min.

**Comparison.** Both learned graphs beat the fixed graph by a wide margin. Between them, {{hybrid_vs_adaptive}}. Training moves $\alpha$ from 0.5 to about {{alpha}}, so the model relies mostly on the learned graph but keeps some weight on the road graph. With the skip connection in Q4(c) the order is the same, fixed worst and the two learned graphs within seed variation of each other, and all three beat the temporal-only model.

# Question 6: Experimental analysis and interpretation

## Q6(a) Forecasting comparison

Test period April to December 2021 for every model. Errors in vehicles per 15 minutes, mean ± standard deviation over seeds {{seeds}}. Training time is the wall-clock time of the training epochs of the seed-{{seed0}} run.

{{table_main}}

![Predicted vs actual flow, 60 minutes ahead, four test sensors, 14 to 16 June 2021 (seed {{seed0}}).](figures/q6_pred_vs_actual.png){width=100%}

![Sensor {{zoom_sensor}} on the morning of 15 June 2021, 60 minutes ahead.](figures/q6_zoom.png){width=80%}

All models follow the daily cycle. The differences are in the level and in sudden changes:

* **Neighbourhood bias.** The graph models often predict the level of the neighbourhood instead of the sensor. At sensor 1125559 (SR56-E) the fixed-graph forecast rises early each morning, following the busier I-15 sensors it is linked to. At sensor 1108475 (I5-S) the adaptive and hybrid forecasts sit above the actual daytime flow. The temporal-only model has no such bias.
* **Isolated dips.** Sensor {{zoom_sensor}} drops from {{zoom_0745}} to {{zoom_0800}} vehicles at 08:00 and is back at {{zoom_0815}} at 08:15. The forecast for 09:00 (actual {{zoom_0900}}), made at 08:00 with the dip as the latest input, shows the other side of mixing. The temporal-only model repeats the dip ({{zoom_p_none}}), while the fixed ({{zoom_p_fixed}}), hybrid ({{zoom_p_hybrid}}) and adaptive ({{zoom_p_adaptive}}) models stay closer to the true value, because the other sensors did not dip.

Test errors are higher than validation errors for every model. The test period has more traffic than the training year (mean flow {{flow_test}} vs {{flow_train}} vehicles per 15 minutes, because 2020 included the COVID-19 lockdown), and absolute errors grow with flow.

## Q6(b) Graph structure analysis

For each sensor we take its 10 strongest learned out-neighbours from the adaptive model (seed {{seed0}}, {{ga_pairs}} pairs between sensors with 2020 data). "Change correlation" is the correlation of 15-minute changes in two sensors' deviations from their weekly profiles in 2020 (Q3(c)). It measures shared short-term fluctuations, not the shared daily cycle.

| Measure | Top-10 learned edges | Road-graph edges | Random pairs |
|---|---|---|---|
| Share that are road-graph edges | {{ga_road_pct}} | 100% | {{ga_base_pct}} |
| Median straight-line distance (km) | {{ga_median_km_top}} | {{ga_median_km_road_edges}} | {{ga_median_km_random}} |
| Same freeway and direction | {{ga_same_fwy_top}} | | {{ga_same_fwy_random}} |
| Mean change correlation | {{ga_corr_top}} | {{ga_corr_road_edges}} | {{ga_corr_random}} |

![Left: strongest learned out-neighbour of every sensor, blue if it is also a road-graph edge. Right: distances of road-graph edges, top-10 learned edges and random pairs.](figures/q6_learned_edges.png){width=100%}

**1. Are the strongest learned connections in the road graph?** Mostly not. {{ga_road_pct}} of the top-10 learned edges are road-graph edges, about twice the {{ga_base_pct}} expected at random, but far from all. The weights are concentrated: the 10 strongest entries of a row hold {{ga_top10_mass_pct}} of its weight on average, and they often go to a few hub sensors: {{ga_hubs}}.

**2. Are learned connections geographically close?** Only slightly. The median distance of a top-10 learned edge is {{ga_median_km_top}} km, against {{ga_median_km_random}} km for random pairs and {{ga_median_km_road_edges}} km for road-graph edges. The distance histogram of learned edges is close to the random-pair curve.

**3. Does the learned graph capture dependencies absent from the road graph?** It uses mostly such connections, and they carry some real signal. Their change correlation ({{ga_corr_top}}) is above random pairs ({{ga_corr_random}}) though below road-graph edges ({{ga_corr_road_edges}}). They are more than twice as likely as random pairs to join two sensors on the same freeway and direction ({{ga_same_fwy_top}} vs {{ga_same_fwy_random}}), and they join sensors with similar weekly profiles (profile correlation {{ga_profile_corr_top}} vs {{ga_profile_corr_random}}). So they partly reflect shared corridors and commuting patterns rather than physical links. The hub structure also suggests a network-wide context channel: many sensors read the same few sensors, which tells the model where in the daily cycle the network is. The model has no time-of-day input, so this is useful information.

**4. Why can a learned graph do better or worse than the fixed graph?**

* *Better:* it is trained for the forecasting loss. It can pick long-range or global relations that a distance threshold excludes, down-weight road links that carry little signal (such as the opposite-carriageway links in Q3(c)), and learn how much weight each sensor gives itself. That last point matters most here, given the over-smoothing in Q4(c).
* *Worse:* it adds $2Nd_e$ parameters and must learn from data what the road graph gives for free. It needs more data and epochs, can fit spurious correlations, and has no bias towards locality, so it may miss propagation patterns that are rare in training (incidents). It is also one static matrix learned from 2020, an unusual year.

## Q6(c) Critical discussion

**Does lower error mean the learned graph represents road connectivity?** No. The adaptive model has much lower error than the fixed one, yet most of its strongest connections are not road links and are about as far apart as random pairs (Q6(b)). Forecast error measures how useful a matrix is to this model for this loss, not whether it is physically true. The softmax graph can encode shared daily profiles, global context, or simply a larger self-weight. Different seeds learn different matrices with similar error ({{seed_overlap_sentence}}). Reading a learned graph as a road map needs separate evidence, such as overlap with known links, consistency across seeds, or a test such as cutting an edge and measuring the effect.

**Why might a hybrid outperform either alone?** The two graphs carry complementary information. The road graph gives a reliable local prior for free: short-range propagation, with the highest change correlation. The adaptive graph adds long-range and context relations and can adjust the self-weight. A mixture can use both, and the road prior can regularise the learned part when data is limited. In our runs the hybrid was {{hybrid_vs_adaptive_short}}, and $\alpha$ settled at about {{alpha}}: the model kept a quarter of its weight on the road graph, so the local prior added something on top of the learned relations. The gain is small, and with the skip connection it is within seed variation. A hybrid should matter more with less data, larger graphs, or several graph layers.

**Limits of a static graph.** Traffic dependencies change with time of day, day of week, season, incidents and roadworks. In free flow, neighbouring sensors are only weakly coupled. In congestion, a queue spreads upstream and couples sensors kilometres apart, and its direction and speed change during the day. A single $\hat A$, fixed or learned, has to average these regimes into one compromise. It cannot express "use the upstream sensor during the evening peak but not at night", or react to an incident that cuts a link. Our learned graph, fitted to 2020, is also applied unchanged to the busier 2021 test period. Time-dependent or input-dependent adjacency (attention computed from the current window, as in ASTGCN or DGCRN), or graphs conditioned on time of day, address this at the cost of more computation and less interpretability.

# Question 7: Reproducibility and reporting

**Code:** {{github}}.

| Path | Content |
|---|---|
| `scripts/q1_q2.py` | Q1 and Q2 checks and heatmaps |
| `scripts/prepare_data.py` | extracts SD for 2020 and 2021, 15-minute sums, adjacency submatrix |
| `scripts/q3_analysis.py` | missing values, weekly plot, graph statistics, neighbourhoods |
| `pa3/data.py`, `pa3/models.py`, `pa3/train.py` | windows and splits, models, training and evaluation |
| `run_all.sh` | trains the four models for the given seeds |
| `scripts/analyze.py` | Q6 table, plots and graph analysis |
| `scripts/verify_checkpoints.py`, `tests/` | checkpoint re-evaluation, {{n_tests}} automated tests |
| `report/build.py` | fills every number in this report from `results/` and builds the PDF |
| `results/runs/*.json`, `logs/` | per-run configuration, per-epoch logs, metrics, hardware, timings |

**Reproduce.**

```
./setup_env.sh && source .venv/bin/activate
# LargeST files from Kaggle (liuxu77/largest) in data/raw/: ca_his_raw_2020.h5 as 2020.h5,
# ca_his_raw_2021.h5 as 2021.h5, ca_meta.csv as metadata.csv, ca_rn_adj.npy as adjacency.npy
python scripts/prepare_data.py --raw_dir data/raw
python scripts/q1_q2.py
python scripts/q3_analysis.py
./run_all.sh 0 1 2
python scripts/analyze.py --seeds 0 1 2
python report/build.py
```

The skip-connection runs of Q4(c) use `python -m pa3.train --graph <fixed|adaptive|hybrid> --seed <s> --skip --tag _skip`.

**Hyperparameters.** $d = 32$, temporal kernel 3 (two layers), one graph layer, head $32\to 32\to 4$, $d_e = 10$, hybrid $\alpha$ learned (initial 0.5), Adam with learning rate $10^{-3}$, batch 64, gradient clipping 5, at most 20 epochs, patience 4, masked MAE loss, per-sensor z-score from 2020, input forward-fill limit 4 steps. Seeds {{seeds}}. Epochs run: {{epochs_note}}.

**Validation.** `pytest tests` ({{n_tests}} tests, all passing) re-checks the Q1 and Q2 answers, compares 1,200 resampled values with the raw 5-minute files, and checks the sensor order of the flow tensor and the adjacency against the LargeST metadata. It also checks that no window crosses a split boundary, that replacing all 2021 data with random numbers changes neither the normalisation statistics nor any 2020 input, that $\hat A$ is row-stochastic and the fixed graph is never updated, that the forward pass matches the formulas above, that the masked metrics match a direct NumPy computation, and that training is exactly reproducible for a fixed seed. `scripts/verify_checkpoints.py` reloads all {{n_ckpt}} saved checkpoints and reproduces every logged validation and test MAE and RMSE to within $10^{-3}$.

**Hardware.** {{hw}}. No GPU was used.

**Data provenance.** LargeST CA release (CC BY-NC 4.0), from the HuggingFace mirror {{mirror}}, SHA-256 verified against the mirror. The SD graph statistics (716 nodes, 17,319 edges) match the LargeST paper.
