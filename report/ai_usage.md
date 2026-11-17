# AI-use disclosure

**How I approached the assignment.** I started by scoping the handout before writing any code. I counted the questions and separated the two theory questions (Q1 and Q2) from the five experimental ones (Q3 to Q7). I listed every fixed constraint: the San Diego subset, the 2020 and 2021 splits, 15-minute resampling, eight input steps and four output steps, hidden size 16 to 32, a fixed seed, and no use of validation or test data for anything learned. I then checked whether the work needed a GPU. Since the handout asks for a lightweight model and says a GPU is optional, I kept everything on CPU. I worked in the order of the handout: theory first, then the data and graph, then the models, then the analysis.

**Tools, outputs and roles.** Claude (Anthropic, Claude Opus 5.5) supplied the Q1 and Q2 derivations and numerical checks, the data pipeline, the model and training code, the experiment runs, the tests, the figures and the report drafts. I directed each stage, reviewed the results against the handout and asked for corrections when the work fell short. All measurements were produced on CPU ({{hw_short}}), as listed in Question 7.

**Reconstructed prompt sequence.** These prompts were written retrospectively to express the instructions and review questions for each stage. They are not a verbatim record of the original interactions. The work produced and my edits are summarised separately below.

**Step 1. Scope the assignment**

> Review the handout before changing anything. Count the questions, separate the derivations from the experiments, and list every fixed constraint on data, splits, model size and evaluation. Identify what can run on a CPU and whether any part needs a GPU, and propose a plan in the order of the handout.

**Step 2. Work through the theory**

> For Q1, show what a sum of affine messages can and cannot distinguish, construct a scalar message whose sum recovers all three counts, and check it in NumPy for every allowed neighbourhood when K = 8. For Q2, fit the symmetric layer by least squares, roll it forward against the true motion, and show how directional weights or one past snapshot recover the direction of travel.

**Step 3. Prepare the data and the graph**

> Extract the 716 San Diego sensors for 2020 and 2021 from LargeST and decide whether the 5-minute values are rates or counts before resampling. Report missing values and how each kind is handled. Make sure the adjacency rows follow the same sensor order as the flow tensor, and check the graph against the statistics published with the dataset.

**Step 4. Build the models under one protocol**

> Build the forecasting windows and chronological splits without crossing boundaries, and normalise with training statistics only. Implement the temporal encoder, one graph convolution and the forecast head exactly as specified, plus a temporal-only baseline. Train fixed, adaptive and hybrid adjacency with the same protocol and several seeds, and record training times.

**Step 5. Analyse the results**

> Compare the four models in the required table and plot predictions against actual flow. Test whether the strongest learned connections match the road graph and whether they are geographically close, and explain what a lower error does and does not show about the learned graph.

**Step 6. Validate before calling it finished**

> Do not treat the first complete draft as final. Write tests for resampling, sensor order, split boundaries, leakage and metrics. Reload every saved checkpoint and confirm it reproduces the reported numbers. Check every claim in the text against the data.

**Step 7. Audit against the handout**

> Map every request, requirement and constraint in the handout to the report and code. Flag anything that deviates from a stated formula, and remove anything the handout does not ask for. Keep the text concise and plain.

**Produced work, edits and verification.**

**Work produced.** AI assistance produced the derivations and numerical checks, the data pipeline, the models and training runs, the test suite, the analysis, the figures and the report drafts. Code, per-run logs and result files are in the repository.

**My decisions and checks.**

* I scoped the work and fixed the constraints before implementation started, including the decision to run everything on CPU.
* I did not accept the first complete draft as finished. I asked for a second round of validation, which added the test suite and the checkpoint re-evaluation. It also corrected the report: validation RMSE was missing, several claims had no numbers behind them, and the number of runs was miscounted.
* I then asked for a full audit against the handout. It found that the forecast head received $Z + H$ instead of the specified $H$. I kept the specified architecture as the main result, even though it performs worse than the temporal-only baseline. The nine graph models were retrained, and the earlier skip-connection runs appear only as a labelled comparison that explains the result.
* I had everything beyond the handout removed, and I required all text to follow my writing style.
* The final results use {{nseeds}} seeds. {{n_tests}} tests pass, and all {{n_ckpt}} saved checkpoints reproduce their reported metrics.
