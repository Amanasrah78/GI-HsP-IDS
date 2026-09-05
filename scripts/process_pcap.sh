#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <pcap_file> <experiment_id>"
    exit 1
fi

PCAP_FILE="$1"
EXPERIMENT_ID="$2"

ZEEK_DIR="results/raw/${EXPERIMENT_ID}/zeek"
CSV_FILE="results/processed/${EXPERIMENT_ID}.csv"
MQTT_CSV_FILE="results/processed/${EXPERIMENT_ID}.mqtt_publish.csv"
GRAPH_FILE="graph/output/${EXPERIMENT_ID}.json"
DYNAMIC_GRAPH_FILE="graph/output/${EXPERIMENT_ID}.dynamic.json"
TIMING_FILE="capture/pcap/${EXPERIMENT_ID}.timing.json"

if [ ! -f "$TIMING_FILE" ]; then
    echo "ERROR: Missing timing file: $TIMING_FILE"
    exit 1
fi

MEASUREMENT_START_TS="$(jq -r '.measurement_start_ts' "$TIMING_FILE")"
MEASUREMENT_END_TS="$(jq -r '.measurement_end_ts' "$TIMING_FILE")"

echo "[1/5] Running Zeek..."
rm -rf "$ZEEK_DIR"
./scripts/run-zeek.sh "$PCAP_FILE" "$ZEEK_DIR"

echo "[2/5] Converting conn.log to CSV..."
python3 preprocessing/zeek_conn_to_csv.py \
    "$ZEEK_DIR/conn.log" \
    "$CSV_FILE"

echo "[3/5] Building communication graph..."
python3 graph/build_graph.py \
    "$CSV_FILE" \
    "$GRAPH_FILE"

echo "[4/5] Converting mqtt_publish.log to CSV..."
python3 preprocessing/zeek_mqtt_publish_to_csv.py \
    "$ZEEK_DIR/mqtt_publish.log" \
    "$MQTT_CSV_FILE"

echo "[5/5] Building 5-second dynamic graph..."
python3 graph/build_dynamic_graph.py \
    "$MQTT_CSV_FILE" \
    5 \
    "$MEASUREMENT_START_TS" \
    "$MEASUREMENT_END_TS" \
    "$DYNAMIC_GRAPH_FILE"

echo
echo "Processing complete."
echo "Zeek logs    : $ZEEK_DIR"
echo "Flow CSV     : $CSV_FILE"
echo "MQTT CSV     : $MQTT_CSV_FILE"
echo "Graph JSON   : $GRAPH_FILE"
echo "Dynamic graph: $DYNAMIC_GRAPH_FILE"
