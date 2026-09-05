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
GRAPH_FILE="graph/output/${EXPERIMENT_ID}.json"

echo "[1/3] Running Zeek..."
rm -rf "$ZEEK_DIR"
./scripts/run-zeek.sh "$PCAP_FILE" "$ZEEK_DIR"

echo "[2/3] Converting conn.log to CSV..."
python3 preprocessing/zeek_conn_to_csv.py \
    "$ZEEK_DIR/conn.log" \
    "$CSV_FILE"

echo "[3/3] Building communication graph..."
python3 graph/build_graph.py \
    "$CSV_FILE" \
    "$GRAPH_FILE"

echo
echo "Processing complete."
echo "Zeek logs : $ZEEK_DIR"
echo "Flow CSV  : $CSV_FILE"
echo "Graph JSON: $GRAPH_FILE"
