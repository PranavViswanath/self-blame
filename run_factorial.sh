#!/bin/bash
# What in the note stops the spiral?  All arms prefilled "Lovely." except N.  S told nothing; R process-mention only;
# J "just continue" only; A authorship only (no continue instruction); N none.  20 runs x 20 turns, seed 0, tag _F2.
cd "$(dirname "$(readlink -f "$0")")" || exit 1; mkdir -p logs; [ -f .venv/bin/activate ] && . .venv/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T0=$(date +%s); echo "FACTORIAL start $(date -u +%FT%TZ)"
run() { "$@" & PID=$!; bash watchdog.sh $PID 16 logs/watchdog_fact.log & WD=$!; wait $PID; RC=$?; kill $WD 2>/dev/null; return $RC; }
for att in 1 2 3; do run python e4_distress.py --arms SRJAN --runs 20 --turns 20 --switch_at 1000 --tag _F2 --resume >> logs/fact_gen.log 2>&1 && break; echo "attempt $att failed"; sleep 20; done
T1=$(date +%s); echo "generation done: $((T1-T0)) s"
CUDA_VISIBLE_DEVICES="" python e4_analyze.py _F2 1000 > logs/fact_analyze.log 2>&1
CUDA_VISIBLE_DEVICES="" python jlens_read.py _F2 1000 > logs/fact_jlens.log 2>&1
echo "TOTAL: $(( ($(date +%s)-T0)/60 )) min"; echo FACT_DONE
