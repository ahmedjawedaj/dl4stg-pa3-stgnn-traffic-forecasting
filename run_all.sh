#!/usr/bin/env bash
# Trains the four models of Q4 and Q5, then the three skip-connection runs of Q4(c), with the same protocol.
# Usage: ./run_all.sh [seed ...]
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${@:-0}
mkdir -p logs results/runs
for s in $SEEDS
do
  for g in none fixed adaptive hybrid
  do
    ${PYTHON:-python3} -m pa3.train --graph "$g" --seed "$s" --epochs 20 --patience 4 --batch 64 --lr 1e-3 2>&1 | tee "logs/${g}_s${s}.log"
  done
  for g in fixed adaptive hybrid
  do
    ${PYTHON:-python3} -m pa3.train --graph "$g" --seed "$s" --epochs 20 --patience 4 --batch 64 --lr 1e-3 --skip --tag _skip 2>&1 | tee "logs/${g}_s${s}_skip.log"
  done
done
