#!/usr/bin/env bash

BASE="results/gi_hsp_v2/comparisons/experiments"
TOTAL=40
REFRESH=60

while true; do
    clear

    now="$(date '+%Y-%m-%d %H:%M:%S %Z')"

    # ---------------------------------------------------------
    # Overall completion
    # ---------------------------------------------------------
    completed=$(
        find "$BASE" \
            -maxdepth 2 \
            -type f \
            -name summary.json \
            -path '*/mqttset-graphids-fold-*-seed-*/*' \
            2>/dev/null \
        | wc -l
    )

    # Get active Python GraphIDS workers only.
    mapfile -t workers < <(
        ps -eo pid=,ppid=,etimes=,%cpu=,%mem=,rss=,args= \
        | grep '\.venv/bin/python -m models\.proposed\.run_gi_hsp_v2_graphids' \
        | grep -v grep
    )

    running=${#workers[@]}
    pending=$(( TOTAL - completed - running ))

    if (( pending < 0 )); then
        pending=0
    fi

    percent=$(
        awk -v c="$completed" -v t="$TOTAL" \
            'BEGIN { printf "%.1f", (100*c/t) }'
    )

    # ---------------------------------------------------------
    # System statistics
    # ---------------------------------------------------------
    load="$(cut -d' ' -f1-3 /proc/loadavg)"
    cores="$(nproc)"

    mem_info=$(
        free -h \
        | awk '/^Mem:/ {
            printf "%s used / %s total; %s available",
            $3, $2, $7
        }'
    )

    echo "=============================================================="
    echo "               GraphIDS 40-run Monitor"
    echo "=============================================================="
    echo "Time:              $now"
    echo
    echo "Overall progress:  $completed / $TOTAL completed  ($percent%)"
    echo "Currently running: $running"
    echo "Pending:           $pending"
    echo
    echo "System load:       $load"
    echo "CPU cores:         $cores"
    echo "Memory:            $mem_info"
    echo "=============================================================="

    # ---------------------------------------------------------
    # Active workers
    # ---------------------------------------------------------
    if (( running == 0 )); then
        echo
        echo "No GraphIDS workers currently running."
    else
        worker_number=0

        for line in "${workers[@]}"; do
            worker_number=$((worker_number + 1))

            pid="$(awk '{print $1}' <<< "$line")"
            elapsed_seconds="$(awk '{print $3}' <<< "$line")"
            cpu="$(awk '{print $4}' <<< "$line")"
            mem="$(awk '{print $5}' <<< "$line")"
            rss_kb="$(awk '{print $6}' <<< "$line")"

            args="$(cut -d' ' -f7- <<< "$line")"

            fold="$(
                sed -n 's/.*--fold \([0-9][0-9]*\).*/\1/p' \
                <<< "$args"
            )"

            seed="$(
                sed -n 's/.*--seed \([0-9][0-9]*\).*/\1/p' \
                <<< "$args"
            )"

            name="mqttset-graphids-fold-${fold}-seed-${seed}"
            log="$BASE/${name}.log"

            elapsed="$(
                printf '%02dh %02dm %02ds' \
                    $((elapsed_seconds / 3600)) \
                    $(((elapsed_seconds % 3600) / 60)) \
                    $((elapsed_seconds % 60))
            )"

            rss_mb="$(
                awk -v kb="$rss_kb" \
                    'BEGIN { printf "%.1f", kb/1024 }'
            )"

            latest_event=""
            if [[ -f "$log" ]]; then
                latest_event=$(
                    grep '"status": "epoch_completed"' "$log" \
                    | tail -1
                )
            fi

            epoch="?"
            best_epoch="?"
            ap="?"

            if [[ -n "$latest_event" ]]; then
                epoch="$(
                    sed -n \
                    's/.*"epoch": \([0-9][0-9]*\).*/\1/p' \
                    <<< "$latest_event"
                )"

                best_epoch="$(
                    sed -n \
                    's/.*"best_epoch": \([0-9][0-9]*\).*/\1/p' \
                    <<< "$latest_event"
                )"

                ap="$(
                    sed -n \
                    's/.*"validation_average_precision": \([^,}]*\).*/\1/p' \
                    <<< "$latest_event"
                )"
            fi

            echo
            echo "Worker $worker_number"
            echo "--------------------------------------------------------------"
            echo "PID:               $pid"
            echo "Fold / Seed:       $fold / $seed"
            echo "Epoch:             $epoch / 100"
            echo "Best epoch:        $best_epoch"
            echo "Validation AP:     $ap"
            echo "Elapsed:           $elapsed"
            echo "CPU:               ${cpu}%"
            echo "Process memory:    ${mem}% (${rss_mb} MiB RSS)"
        done
    fi

    # ---------------------------------------------------------
    # Most recently completed experiments
    # ---------------------------------------------------------
    echo
    echo "=============================================================="
    echo "Most recent completed runs"
    echo "=============================================================="

    find "$BASE" \
        -maxdepth 2 \
        -type f \
        -name summary.json \
        -path '*/mqttset-graphids-fold-*-seed-*/*' \
        -printf '%T@ %h\n' \
        2>/dev/null \
    | sort -nr \
    | head -5 \
    | cut -d' ' -f2- \
    | sed 's#.*/##'

    # ---------------------------------------------------------
    # Batch-controller events
    # ---------------------------------------------------------
    MASTER="$BASE/graphids-remaining-2way-master.log"

    if [[ -f "$MASTER" ]]; then
        echo
        echo "=============================================================="
        echo "Latest matrix events"
        echo "=============================================================="
        grep -E 'START|DONE|FAILED|ERROR|Finished' "$MASTER" \
            | tail -8
    fi

    echo
    echo "=============================================================="
    echo "Next refresh in ${REFRESH} seconds — Ctrl+C exits monitor"
    echo "=============================================================="

    sleep "$REFRESH"
done
