#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <experiment_id> [duration_seconds]"
    exit 1
fi

EXPERIMENT_ID="$1"
DURATION="${2:-60}"

PCAP_FILE="capture/pcap/${EXPERIMENT_ID}.pcap"
HASH_FILE="capture/pcap/${EXPERIMENT_ID}.sha256"
TIMING_FILE="capture/pcap/${EXPERIMENT_ID}.timing.json"
MANIFEST_FILE="experiments/${EXPERIMENT_ID}.yaml"
CSV_FILE="results/processed/${EXPERIMENT_ID}.csv"
MQTT_CSV_FILE="results/processed/${EXPERIMENT_ID}.mqtt_publish.csv"
PACKET_CSV_FILE="results/processed/${EXPERIMENT_ID}.packets.csv"
WINDOWS_FILE="results/processed/${EXPERIMENT_ID}.windows.jsonl"
SEQUENCES_FILE="results/processed/${EXPERIMENT_ID}.sequences.jsonl"
SUMMARY_FILE="results/processed/${EXPERIMENT_ID}.summary.json"
GRAPH_FILE="graph/output/${EXPERIMENT_ID}.json"
DYNAMIC_GRAPH_FILE="graph/output/${EXPERIMENT_ID}.dynamic.json"
ZEEK_DIR="results/raw/${EXPERIMENT_ID}"

ATTACK_COMMAND="nmap -sT -Pn -p 1883 172.30.0.10 172.30.0.11"

if [ -e "$PCAP_FILE" ] \
    || [ -e "$HASH_FILE" ] \
    || [ -e "$TIMING_FILE" ] \
    || [ -e "$MANIFEST_FILE" ] \
    || [ -e "$CSV_FILE" ] \
    || [ -e "$MQTT_CSV_FILE" ] \
    || [ -e "$PACKET_CSV_FILE" ] \
    || [ -e "$WINDOWS_FILE" ] \
    || [ -e "$SEQUENCES_FILE" ] \
    || [ -e "$SUMMARY_FILE" ] \
    || [ -e "$GRAPH_FILE" ] \
    || [ -e "$DYNAMIC_GRAPH_FILE" ] \
    || [ -e "$ZEEK_DIR" ]; then
    echo "ERROR: Experiment ID already exists: $EXPERIMENT_ID"
    echo "Refusing to overwrite existing experiment artifacts."
    exit 1
fi

echo "========================================"
echo "GI-HsP Nmap reconnaissance experiment"
echo "Experiment ID : $EXPERIMENT_ID"
echo "Duration      : ${DURATION}s"
echo "========================================"

echo
echo "[Stage 1/5] Capturing traffic and injecting HsP attack..."
./scripts/capture_hsp_nmap.sh \
    "$EXPERIMENT_ID" \
    "$DURATION"

echo
echo "[Stage 2/5] Writing experiment manifest..."
python3 scripts/write_experiment_manifest.py \
    "$EXPERIMENT_ID" \
    "$PCAP_FILE" \
    --duration "$DURATION" \
    --timing-file "$TIMING_FILE" \
    --output "$MANIFEST_FILE" \
    --class attack \
    --attack-goal reconnaissance \
    --hsp-family nmap \
    --attacker-container hsp-attacker \
    --attack-command "$ATTACK_COMMAND"

echo
echo "[Stage 3/5] Processing capture..."
./scripts/process_pcap.sh \
    "$PCAP_FILE" \
    "$EXPERIMENT_ID"

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
