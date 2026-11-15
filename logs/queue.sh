#!/usr/bin/env bash
cd "$(dirname "$0")/.."
./run_all.sh 1 2
python3 -m pa3.train --graph hybrid --alpha 0.5 --seed 0 --epochs 20 --patience 4 --batch 64 --lr 1e-3 --tag _a05 > logs/hybrid_s0_a05.log 2>&1
