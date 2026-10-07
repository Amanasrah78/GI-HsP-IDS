#!/usr/bin/env bash
set -u

PROTOCOL="configs/gi_hsp_v2_comparison_graphids.yaml"
BASE="results/gi_hsp_v2/comparisons/experiments"

for fold in 1 2 3 4; do
    for seed in $(seq 5 14); do

        name="mqttset-graphids-fold-${fold}-seed-${seed}"
        out="$BASE/$name"
        log="$BASE/${name}.log"

        if [ -f "$out/summary.json" ]; then
            echo "$(date -u '+%F %T UTC') SKIP completed $name"
            continue
        fi

        if [ -d "${out}.tmp" ]; then
            echo "$(date -u '+%F %T UTC') ERROR stale tmp exists: ${out}.tmp"
            exit 1
        fi

        echo "$(date -u '+%F %T UTC') START $name"

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
            echo "$(date -u '+%F %T UTC') FAILED $name status=$status"
            exit "$status"
        fi

        if [ ! -f "$out/summary.json" ]; then
            echo "$(date -u '+%F %T UTC') ERROR missing summary $name"
            exit 1
        fi

        echo "$(date -u '+%F %T UTC') DONE $name"
    done
done

echo "$(date -u '+%F %T UTC') CATCHUP COMPLETE"
