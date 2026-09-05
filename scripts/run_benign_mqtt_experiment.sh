#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <experiment_id> [duration_seconds]"
    exit 1
fi

EXPERIMENT_ID="$1"
DURATION="${2:-30}"

PCAP_FILE="capture/pcap/${EXPERIMENT_ID}.pcap"
MANIFEST_FILE="experiments/${EXPERIMENT_ID}.yaml"

echo "========================================"
echo "GI-HsP benign MQTT experiment"
echo "Experiment ID : $EXPERIMENT_ID"
echo "Duration      : ${DURATION}s"
echo "========================================"

echo
echo "[Stage 1/3] Capturing traffic..."
./scripts/capture_benign_mqtt.sh \
    "$EXPERIMENT_ID" \
    "$DURATION"

echo
echo "[Stage 2/3] Processing capture..."
./scripts/process_pcap.sh \
    "$PCAP_FILE" \
    "$EXPERIMENT_ID"

echo
echo "[Stage 3/3] Writing experiment manifest..."
python3 scripts/write_experiment_manifest.py \
    "$EXPERIMENT_ID" \
    "$PCAP_FILE" \
    --duration "$DURATION" \
    --output "$MANIFEST_FILE"

echo
echo "========================================"
echo "Experiment complete: $EXPERIMENT_ID"
echo "Manifest: $MANIFEST_FILE"
echo "========================================"
