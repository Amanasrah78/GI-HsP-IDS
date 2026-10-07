#!/usr/bin/env bash
set -u

ROOT="$(pwd)"
PROTOCOL="configs/gi_hsp_v2_comparison_graphids.yaml"
BASE="results/gi_hsp_v2/comparisons/experiments"
MASTER="$BASE/graphids-remaining-2way-master.log"

mkdir -p "$BASE"

run_worker() {
    worker="$1"
    index=0

    for fold in 1 2 3 4; do
        for seed in $(seq 5 14); do

            # Already completed and validated.
            if [ "$fold" -eq 1 ] && [ "$seed" -eq 5 ]; then
                continue
            fi

            # Split jobs deterministically between worker 0 and worker 1.
            assigned=$(( index % 2 ))
            index=$(( index + 1 ))

            if [ "$assigned" -ne "$worker" ]; then
                continue
            fi

            name="mqttset-graphids-fold-${fold}-seed-${seed}"
            out="$BASE/$name"
            log="$BASE/${name}.log"

            if [ -f "$out/summary.json" ]; then
                echo "$(date -u '+%F %T UTC') WORKER=$worker SKIP completed $name"
                continue
            fi

            if [ -d "${out}.tmp" ]; then
                echo "$(date -u '+%F %T UTC') WORKER=$worker ERROR stale tmp exists: ${out}.tmp"
                return 1
            fi

            echo "$(date -u '+%F %T UTC') WORKER=$worker START $name"

            /usr/bin/time -p \
            .venv/bin/python -m models.proposed.run_gi_hsp_v2_graphids \
                --protocol "$PROTOCOL" \
                --fold "$fold" \
                --seed "$seed" \
                --device cpu \
                --output-directory "$out" \
                > "$log" 2>&1

            status=$?

            if [ "$status" -ne 0 ]; then
                echo "$(date -u '+%F %T UTC') WORKER=$worker FAILED $name status=$status"
                return "$status"
            fi

            if [ ! -f "$out/summary.json" ]; then
                echo "$(date -u '+%F %T UTC') WORKER=$worker ERROR no summary for $name"
                return 1
            fi

            echo "$(date -u '+%F %T UTC') WORKER=$worker DONE $name"
        done
    done
}

echo "============================================================"
echo "GraphIDS remaining 39-run matrix"
echo "Started: $(date -u '+%F %T UTC')"
echo "Repository: $ROOT"
echo "Protocol: $PROTOCOL"
echo "Maximum concurrent workers: 2"
echo "Fold 1 seed 5: already completed; skipped"
echo "============================================================"

run_worker 0 &
pid0=$!

run_worker 1 &
pid1=$!

wait "$pid0"
status0=$?

wait "$pid1"
status1=$?

echo "============================================================"
echo "Finished: $(date -u '+%F %T UTC')"
echo "Worker 0 status: $status0"
echo "Worker 1 status: $status1"
echo "============================================================"

if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
    exit 1
fi
