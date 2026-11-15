# Question 1: What does a sum remember?

Throughout, a neighbourhood with $a$ slowing, $s$ steady and $b$ speeding-up sensors has $n_v = a+s+b$ and $S_v = b-a$.

## Q1(a) Affine messages

With $\phi(x) = wx + c$ the aggregated message is
$$\sum_{u\in N(v)} \phi(x_u) = w\sum_{u\in N(v)} x_u + c\,|N(v)| = w\,S_v + c\,n_v .$$
The layer output is $h_v = \rho(w S_v + c\, n_v)$. The argument of $\rho$ depends on the neighbourhood only through the pair $(n_v, S_v)$. If two neighbourhoods have the same $n_v$ and $S_v$, $\rho$ receives the same number for every choice of $w$ and $c$, so $h_v$ is identical for every $\rho$. No choice of parameters can separate them.

The pair $(n_v, S_v)$ does not fix the counts. From $a+s+b=n_v$ and $b-a=S_v$ we get $s = n_v - S_v - 2b$, so one degree of freedom is lost and the number of steady neighbours is not determined.

Example: the neighbourhood drawn in the handout, $A = \{+1,-1,0,0,-1\}$, has $n_v=5$, $S_v=-1$ and two steady neighbours. $B=\{-1,-1,-1,+1,+1\}$ also has $n_v=5$, $S_v=-1$ but zero steady neighbours. Both give the aggregate $-w + 5c$. We checked this for 1000 random $(w,c)$ in `scripts/q1_q2.py` (all equal).

## Q1(b) A nonlinear scalar message that keeps all three counts

Let $B = K+1$ and choose
$$\phi(-1) = 1,\qquad \phi(0) = B = K+1,\qquad \phi(+1) = B^2 = (K+1)^2 .$$
Then
$$M_v=\sum_{u\in N(v)}\phi(x_u) = a + s\,B + b\,B^2 .$$
This is the number with base-$B$ digits $(b, s, a)$. Each count is at most $|N(v)| \le K < B$, so every digit is a valid base-$B$ digit, no carry can happen, and base-$B$ representations are unique. The counts are recovered by
$$a = M_v \bmod B,\qquad s = \lfloor M_v / B\rfloor \bmod B,\qquad b = \lfloor M_v/B^2 \rfloor .$$
The base must be at least $K+1$. With base $K$, a neighbourhood of $K$ slowing sensors gives $K\cdot 1 = K$, the same as a single steady sensor ($1\cdot K$), so the counts collide.

**NumPy check, $K=8$** ($\phi = 1, 9, 81$): we enumerated every triple $(a,s,b)$ of non-negative integers with $1 \le a+s+b \le 8$. There are {{q1b_triples}} triples, they produce {{q1b_distinct}} distinct sums ({{q1b_collisions}} collisions, largest sum {{q1b_max}}), and decoding every sum returned the original triple.

**Counting only non-steady neighbours.** Use $\phi(x) = x^2$ (equivalently $|x|$): $\phi(-1)=\phi(+1)=1$, $\phi(0)=0$. Then $\sum_u \phi(x_u) = a + b$ exactly, for any neighbourhood size, because the sum just adds one per non-steady sensor and needs no bound $K$.

## Q1(c) Sum, mean and max of one-hot messages

With one-hot messages the three aggregators give $\text{Sum} = (a,s,b)$, $\text{Mean} = (a,s,b)/n_v$ and $\text{Max} = (\mathbb 1[a>0], \mathbb 1[s>0], \mathbb 1[b>0])$. We confirmed every row below by brute force over all {{q1c_hoods}} allowed neighbourhoods (1 to 8 neighbours).

| Task | Method | Possible? | Reason |
|-----------|-----|------|------------------------------------|
| (i) Count steady neighbours | Sum | Yes | $\rho$ returns coordinate 2, which is $s$. |
| | Mean | No | $\{0\}$ and $\{0,0\}$ both give $(0,1,0)$. Answers 1 and 2. |
| | Max | No | $\{0\}$ and $\{0,0\}$ both give $(0,1,0)$. Answers 1 and 2. |
| (ii) More than half slowing | Sum | Yes | $n_v$ is the sum of the three coordinates. Test $2a > a+s+b$. |
| | Mean | Yes | Coordinate 1 is $a/n_v$. Test $a/n_v > 1/2$. |
| | Max | No | $\{-1,0\}$ and $\{-1,-1,0\}$ both give $(1,1,0)$. Answers No (1 of 2) and Yes (2 of 3). |
| (iii) Any neighbour slowing | Sum | Yes | Test coordinate 1: $a > 0$. |
| | Mean | Yes | Test coordinate 1: $a/n_v > 0$, which holds exactly when $a>0$. |
| | Max | Yes | Coordinate 1 is $\mathbb 1[a>0]$, which is the answer. |

Sum keeps the full multiset of counts, so it can do all three tasks. Mean keeps proportions but loses $n_v$, so it fails on absolute counts. Max keeps only which trends are present, so it fails whenever the answer depends on how many.

# Question 2: Which way is the slowdown moving?

## Q2(a)(i) Least-squares fit of the symmetric layer

Input: $x_{20}(t)=1$, all other readings 0. Target: $x_{19}(t+1)=1$, all others 0. Only sensors 19, 20 and 21 have a non-zero prediction:

| Sensor | Prediction $\hat x_i(t+1)$ | Target $x_i(t+1)$ |
|---|---|---|
| 19 | $a\cdot 0 + b(x_{18}+x_{20}) = b$ | 1 |
| 20 | $a\cdot 1 + b(0+0) = a$ | 0 |
| 21 | $a\cdot 0 + b(x_{20}+x_{22}) = b$ | 0 |

The sum of squared errors is $L(a,b) = (b-1)^2 + a^2 + b^2$. Setting the derivatives to zero:
$$\frac{\partial L}{\partial a} = 2a = 0 \Rightarrow a = 0,\qquad \frac{\partial L}{\partial b} = 2(b-1) + 2b = 0 \Rightarrow b = \tfrac12 .$$
The minimum error is $L = \tfrac14 + 0 + \tfrac14 = \tfrac12$. The layer cannot tell 19 from 21, so the best it can do is split the slowdown equally between them. `np.linalg.lstsq` gives the same values ($a={{q2_a}}$, $b={{q2_b}}$, error {{q2_sse}}).

## Q2(a)(ii) Rolling the fitted layer forward

With $a=0$, $b=\tfrac12$ the layer is $\hat x_i(t+1) = \tfrac12(x_{i-1}(t)+x_{i+1}(t))$, a discrete diffusion step.

![Smooth bump rolled out for 20 steps. Left: true motion. Right: the fitted symmetric layer applied to its own predictions. Same colour scale.](figures/q2a_heatmaps.png){width=95%}

The true bump moves one sensor against traffic per step and keeps its shape, so after 20 steps it is centred at sensor 0. The fitted layer never moves the bump: its centre stays at sensor {{q2_centre}}, and it spreads out symmetrically, with the peak falling from 1.00 to {{q2_peak20}} while the total slowdown stays constant ({{q2_mass}}), because $a + 2b = 1$.

## Q2(b) Weighting the two directions separately

Choose $a = 0$, $w_- = 0$, $w_+ = 1$. Substituting:
$$\hat x_i(t+1) = 0\cdot x_i(t) + 0\cdot x_{i-1}(t) + 1\cdot x_{i+1}(t) = x_{i+1}(t) = x_i(t+1).$$
This is exact for every input, which we confirmed on 200 random snapshots. In (a) the layer had to give sensors $i-1$ and $i+1$ the same weight, so the isolated slowdown produced the same prediction at 19 and 21. With separate weights the layer knows which neighbour is ahead and can copy from it only, which is exactly the motion rule.

As a 1D convolution: kernel $[0, 0, 1]$ multiplying $[x_{i-1}, x_i, x_{i+1}]$, one input and one output channel, no bias. The ring is handled with circular padding of one sensor on each side (sensor 39 is placed before sensor 0 and sensor 0 after sensor 39), e.g. `nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular")`. PyTorch convolutions are cross-correlations, so the kernel is used in this order without flipping. We checked that this convolution reproduces `np.roll(x, -1)` exactly.

## Q2(c) Using history with equal neighbour weights

**(i) Backward motion.** The rule $x_i(t) = x_{i+1}(t-1)$ says the reading at sensor $i$ at time $t-1$ has moved to sensor $i-1$ at time $t$: $x_i(t-1) = x_{i-1}(t)$. Substituting into the model:
$$\hat x_i(t+1) = a\,x_i(t) + b\,x_{i-1}(t) + b\,x_{i+1}(t) + c\,x_{i-1}(t) = a\,x_i(t) + (b+c)\,x_{i-1}(t) + b\,x_{i+1}(t).$$
The target is $x_i(t+1) = x_{i+1}(t)$, so we need $a = 0$, $b = 1$ and $b + c = 0$, giving
$$a = 0,\qquad b = 1,\qquad c = -1,\qquad \hat x_i(t+1) = x_{i-1}(t) + x_{i+1}(t) - x_i(t-1).$$
The history term cancels the neighbour behind the pattern, leaving only the one ahead.

**(ii) Forward motion.** Now $x_i(t+1) = x_{i-1}(t)$ and $x_i(t) = x_{i-1}(t-1)$, so $x_i(t-1) = x_{i+1}(t)$. With the same coefficients:
$$\hat x_i(t+1) = x_{i-1}(t) + x_{i+1}(t) - x_{i+1}(t) = x_{i-1}(t) = x_i(t+1).$$
The same coefficients are still exact. Both cases were confirmed numerically on 200 random snapshot pairs.

**(iii) Why history helps.** A single snapshot of a symmetric pattern looks the same whether it moves left or right. The previous snapshot records where the pattern came from. Subtracting $x_i(t-1)$ removes the contribution of the neighbour the pattern has just left (the trailing side), so what remains is the neighbour on the leading side, whichever direction that is. The rule $x_i(t+1) + x_i(t-1) = x_{i-1}(t) + x_{i+1}(t)$ is a discrete wave equation, whose solutions are exactly patterns travelling at one sensor per step in either direction. The spatial weights stay symmetric, and the time ordering of the two snapshots supplies the direction.

# Question 3: Dataset preparation and graph construction

## Q3(a) Data extraction and resampling

**Source.** LargeST CA release: raw 5-minute flow for 2020 and 2021 (`2020.h5`, `2021.h5`, all 8600 California sensors), sensor metadata (`ca_meta.csv`) and the road-distance adjacency (`ca_rn_adj.npy`). Kaggle could not be reached from the compute environment, so the identical files were downloaded from a HuggingFace mirror ({{mirror}}). The download was checked against the mirror's SHA-256 hashes. The SD graph we extract has 716 nodes and 17,319 edges, the same as Tables 1 and 4 of the LargeST paper, which confirms the metadata and adjacency match the official release.

**Extraction.** Following the official `generate_sd_dataset.ipynb`, the SD sensors are the 716 rows of the metadata with `District == 11`, in metadata order. The flow columns are selected by sensor `ID`, and the adjacency rows and columns by the `ID2` index of the same rows, so node $i$ is the same sensor in the flow tensor and in $A_{SD}$. The raw files are read column-wise with `h5py`, so the full 8600-sensor matrices never have to fit in memory.

**Resampling.** PeMS 5-minute "flow" is the number of vehicles counted during the 5-minute interval, summed over lanes. It is an interval count, not a rate, so the 15-minute value is the **sum** of the three 5-minute counts, in vehicles per 15 minutes:
$$X^{15}_t = \sum_{k=0}^{2} X^{5}_{3t+k}.$$
If one or two of the three readings are missing, we use $3\times$ the mean of the observed readings, so a partly observed bin is not biased low. A bin with no readings stays missing. Every fully observed 15-minute value in the result is an integer, as expected for summed vehicle counts. (The official LargeST script `process_ca_his.ipynb` averages instead, giving vehicles per 5 minutes, so their numbers are one third of ours, and it fills missing values with 0, which we avoid below.) The result is $X \in \mathbb R^{T\times N}$ with $T = {{T15}}$ steps (2020-01-01 00:00 to 2021-12-31 23:45) and $N=716$. Observed 15-minute flow has mean {{flow_mean}}, median {{flow_median}} and maximum {{flow_max}} vehicles.

**Missing values.**

| Statistic | Value |
|----------------------------------|--------------|
| 5-minute readings missing | {{m5}} |
| 15-minute bins with no reading | {{m15}} (2020: {{m15_2020}}, 2021: {{m15_2021}}) |
| 15-minute bins partly observed (1 or 2 of 3 readings) | {{m_partial}} |
| Sensors with no reading in 2020 and 2021 | {{never}} |
| Sensors with no reading in 2021 (so none in validation or test) | {{never21}} |
| Sensors more than 50% missing | {{over50}} |
| Timestamps where every sensor is missing | {{allmiss_t}} |
| Missing bins that belong to gaps of 1 hour or less | {{short_gap_frac}} |

![Missing data. Left: share of missing bins per day. Right: missingness per sensor.](figures/q3a_missing.png){width=95%}

Missingness is dominated by whole sensors that are offline for months, not by scattered short gaps: {{sens_lt5}} sensors miss less than 5% of their bins, while {{sens_ge60}} sensors miss 60% or more. The steps in the daily curve are groups of sensors going offline in late March 2020 and on 1 October 2020. The two one-day spikes (8 March 2020 and 14 March 2021) are the daylight-saving switch: the hour from 02:00 to 03:00 does not exist on those days, so every sensor is missing for four bins. These are exactly the 8 timestamps where all sensors are missing. We handle missing values as follows.

1. **Targets.** Missing values are never filled for the loss or the metrics. Every loss and every MAE/RMSE is computed only over observed targets (a mask), so the models are never trained or scored on made-up values.
2. **Inputs.** A gap is forward-filled with the last observed value for at most 4 steps (1 hour). This only uses past values, so it is valid at prediction time and leaks nothing across splits. Longer gaps are set to the sensor's training-set mean, which is 0 after normalisation.
3. **Normalisation statistics** (per-sensor mean and standard deviation) come only from observed 2020 values. The {{bad_stats}} sensors with no 2020 data use the global 2020 mean and standard deviation.

![Traffic flow for three sensors over one week in February 2020 (before the COVID-19 lockdown).](figures/q3a_week.png){width=100%}

The weekly plot shows the structure the models must learn: a strong daily cycle, a morning and evening peak on weekdays, a single broad peak at weekends, and short-term fluctuations on top that differ between sensors.

## Q3(b) Traffic graph

| Property | Value |
|--------------|----------------------------------|
| Nodes | {{adj_nodes}} |
| Non-zero entries (with the 716 self-loops on the diagonal) | {{adj_nnz}} |
| Non-zero edges, self-loops excluded | {{adj_edges}} |
| Density (off-diagonal) | {{adj_density}} |
| Symmetric | No. Only {{adj_recip}} of the edges also appear in the reverse direction, and the largest $|A_{ij}-A_{ji}|$ is {{adj_asym}} |
| Edge weights | {{adj_wmin}} to {{adj_wmax}} |
| Out-degree (neighbours per row) | mean {{deg_mean}}, median {{deg_med}}, max {{deg_max}}, {{deg_iso}} sensors with none |

![Left: adjacency heatmap (log colour scale, zeros white). Right: number of neighbours per sensor.](figures/q3b_adjacency.png){width=100%}

**What $A_{ij}\neq 0$ means.** LargeST builds the matrix with a thresholded Gaussian kernel of road-network distance: $A_{ij} = \exp(-d_{ij}^2/\sigma^2)$ if this is at least $r$, else 0, where $d_{ij}$ is the driving distance from sensor $i$ to sensor $j$ (computed with the OSRM routing engine, only for sensor pairs within 4 km of each other) and $\sigma$ is the standard deviation of all distances. A non-zero entry therefore means sensor $j$ can be reached from sensor $i$ along the road network within a short driving distance, and its value (between 0.01 and 1, matching the paper's threshold $r = 0.01$) grows as that distance shrinks. Driving distance is not symmetric on a freeway network (one-way carriageways, ramps), which is why $A \neq A^\top$. The diagonal bands in the heatmap are consecutive sensors along the same freeway, and the off-diagonal blocks are interchanges between freeways.

## Q3(c) Graph neighbourhoods

We selected sensor {{nb_id}} on {{nb_fwy}}, whose number of neighbours equals the median. Its one-hop neighbourhood $N(i) = \{j: A_{ij}\neq 0\}$ has {{nb_one}} sensors. Paths of at most two edges reach {{nb_two}} sensors, {{nb_two_only}} of them not already one-hop. Both sets lie on {{nb_fwys}}, around the SR-56 / I-15 interchange.

![One-hop (blue) and two-hop only (orange) neighbours of sensor {{nb_id}}.](figures/q3c_neighbourhood.png){width=62%}

**Why neighbours help.** Vehicles passing sensor $i$ were recently at upstream sensors and will soon pass downstream ones, so congestion forming nearby shows up at $i$ a few steps later, as the ring-road example in Q2 showed. In 2020 data, the 15-minute changes of the selected sensor's deviation from its weekly profile correlate on average {{corr1}} with its one-hop neighbours and {{corr2}} with two-hop-only neighbours, against {{corr3}} with sensors further away. So neighbours carry short-term information that the sensor's own daily pattern does not. Raw flows correlate highly with everyone ({{rawcorr1}} vs {{rawcorr3}}) because all sensors share the same daily cycle.

**One-hop vs two-hop aggregation.** One graph convolution mixes each sensor with its direct neighbours ($\hat A Z$). Stacking a second layer mixes in neighbours of neighbours ($\hat A^2 Z$), so information can travel twice as far on the road network, here from about {{nb_one}} to about {{nb_two}} sensors. Two-hop information can capture congestion arriving from further away, but it is also averaged over many more sensors, so each individual signal is weaker.

**Too many layers.** Yes, performance can drop. Each layer of a normalised graph convolution is a smoothing step, like the diffusion in Q2(a)(ii). After many layers, the representations of all sensors in a connected region converge towards the same weighted average (over-smoothing), so sensor-specific detail is lost. Deep GNNs also aggregate exponentially growing neighbourhoods into fixed-size vectors (over-squashing), and are harder to train. In this graph, opposite carriageways are also connected (the neighbours above include SR56-E and SR56-W), so deep mixing blends traffic flowing in opposite directions. For a 1-hour horizon, nearby sensors matter most, so one or two layers are usually enough.
