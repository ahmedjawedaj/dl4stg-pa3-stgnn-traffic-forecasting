# Generative AI usage

Following the course AI policy, this section lists the prompts I gave (verbatim), what the tool produced, and the edits I made.

**Tool:** Claude (Anthropic, model Claude Opus 5.5), used through the Claude app with access to my assignment folder and a cloud CPU workspace.

## Prompt 1

```
i have added assignment 3 in assignments folder
please review it

tell me how many questions
whats the problem
potential solution

do you need a gpu ? what van be done on my mac? and whats your plan?

start
```

**Output.** The tool read the handout, summarised the seven questions and the intended solution for each, and noted that no GPU is needed because the handout asks for hidden size 16 to 32 on CPU. It then:

1. derived the Q1 and Q2 answers and wrote `scripts/q1_q2.py`, which checks them numerically and draws the Q2 heatmaps,
2. downloaded LargeST 2020 and 2021 (a HuggingFace mirror, because Kaggle was unreachable from its workspace), verified the checksums, and wrote `scripts/prepare_data.py` and `scripts/q3_analysis.py`,
3. wrote the `pa3` package (data windows, models, training loop) and `run_all.sh`, ran all four models with {{nseeds}} seeds on CPU plus the fixed-$\alpha$ ablation, and wrote `scripts/analyze.py`,
4. drafted this report (`report/*.md`) and `report/build.py`, which fills in every number from the result files.

## Prompt 2

```
Continue from where you left off.
```

**Output.** The tool waited for the remaining seed-2 and ablation runs, ran the three-seed analysis, rebuilt the report and copied the repository to my assignment folder.

## Prompt 3

```
i don't think its fully done
review again
validate
test

and ensure it is fully done
```

**Output.** The tool re-checked the report against every requirement in the handout, wrote the `tests/` suite ({{n_tests}} tests), `scripts/verify_checkpoints.py` (all {{n_ckpt}} checkpoints reproduce their logged metrics) and `scripts/naive_baselines.py`, and fact-checked the text against the data and the LargeST paper and code. This led to these corrections: validation RMSE added to the result tables, the cause of the 8 all-missing timestamps identified (daylight saving), the claims about the prediction plots backed with measured numbers, a miscounted number of training runs fixed, and the LargeST adjacency description completed (OSRM distances within 4 km, Tables 1 and 4).

## Edits I made

EDITS_PLACEHOLDER: describe here what you checked, changed or rewrote yourself (for example: re-derived Q1 and Q2 by hand, changed wording, re-ran the code, checked the plots). Remove this line before submitting.
