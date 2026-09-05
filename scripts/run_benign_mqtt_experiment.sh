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

if [ -e "$PCAP_FILE" ] || [ -e "$MANIFEST_FILE" ]; then
    echo "ERROR: Experiment ID already exists: $EXPERIMENT_ID"
    echo "Refusing to overwrite existing experiment artifacts."
    exit 1
fi

echo "========================================"
echo "GI-HsP benign MQTT experiment"
echo "Experiment ID : $EXPERIMENT_ID"
echo "Duration      : ${DURATION}s"
echo "========================================"

echo
echo "[Stage 1/5] Capturing traffic..."
./scripts/capture_benign_mqtt.sh \
    "$EXPERIMENT_ID" \
    "$DURATION"

echo
echo "[Stage 2/5] Processing capture..."
./scripts/process_pcap.sh \
    "$PCAP_FILE" \
    "$EXPERIMENT_ID"

echo
echo "[Stage 3/5] Writing experiment manifest..."
python3 scripts/write_experiment_manifest.py \
    "$EXPERIMENT_ID" \
    "$PCAP_FILE" \
    --duration "$DURATION" \
    --output "$MANIFEST_FILE"

echo
echo "[Stage 4/5] Writing experiment summary..."
./scripts/write_experiment_summary.sh \
    "$EXPERIMENT_ID"

echo
echo "[Stage 5/5] Validating experiment..."
python3 scripts/validate_experiment.py \
    "$EXPERIMENT_ID"

echo
echo "========================================"
echo "Experiment complete and validated"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Manifest     : $MANIFEST_FILE"
echo "========================================"
