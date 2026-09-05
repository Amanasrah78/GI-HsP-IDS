#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <experiment_id> [duration_seconds]"
    exit 1
fi

EXPERIMENT_ID="$1"
DURATION="${2:-30}"

PCAP_FILE="capture/pcap/${EXPERIMENT_ID}.pcap"

echo "========================================"
echo "GI-HsP benign MQTT experiment"
echo "Experiment ID : $EXPERIMENT_ID"
echo "Duration      : ${DURATION}s"
echo "========================================"

echo
echo "[Stage 1/2] Capturing traffic..."
./scripts/capture_benign_mqtt.sh \
    "$EXPERIMENT_ID" \
    "$DURATION"

echo
echo "[Stage 2/2] Processing capture..."
./scripts/process_pcap.sh \
    "$PCAP_FILE" \
    "$EXPERIMENT_ID"

echo
echo "========================================"
echo "Experiment complete: $EXPERIMENT_ID"
echo "========================================"
