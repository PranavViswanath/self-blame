#!/bin/bash
# From-zero replication of the central experiment, timed.  Arms S (self) / D (disclosed) / N (none), 20 runs, 20 turns,
# NEW seed (1), tag _R1.  Then: behaviour + state readouts (e4_analyze) and Jacobian-lens readouts (jlens_read).
cd "$(dirname "$(readlink -f "$0")")" || exit 1; mkdir -p logs; [ -f .venv/bin/activate ] && . .venv/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T0=$(date +%s); echo "REPRO start $(date -u +%FT%TZ)"
run() { "$@" & PID=$!; bash watchdog.sh $PID 16 logs/watchdog_repro.log & WD=$!; wait $PID; RC=$?; kill $WD 2>/dev/null; return $RC; }
for att in 1 2 3; do run python e4_distress.py --arms SDN --runs 20 --turns 20 --switch_at 1000 --seed 1 --tag _R1 --resume >> logs/repro_gen.log 2>&1 && break; echo "attempt $att failed"; sleep 20; done
T1=$(date +%s); echo "generation+capture done: $((T1-T0)) s"
CUDA_VISIBLE_DEVICES="" python e4_analyze.py _R1 1000 > logs/repro_analyze.log 2>&1
T2=$(date +%s); echo "state/behaviour analysis done: $((T2-T1)) s"
CUDA_VISIBLE_DEVICES="" python jlens_read.py _R1 1000 > logs/repro_jlens.log 2>&1
T3=$(date +%s); echo "jacobian-lens readout done: $((T3-T2)) s"
echo "TOTAL from zero: $((T3-T0)) s = $(( (T3-T0)/60 )) min"
echo REPRO_DONE
