#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_id>"
    exit 1
fi

EXPERIMENT_ID="$1"

PCAP_FILE="capture/pcap/${EXPERIMENT_ID}.pcap"
CSV_FILE="results/processed/${EXPERIMENT_ID}.csv"
MQTT_CSV_FILE="results/processed/${EXPERIMENT_ID}.mqtt_publish.csv"
PACKET_CSV_FILE="results/processed/${EXPERIMENT_ID}.packets.csv"
WINDOWS_FILE="results/processed/${EXPERIMENT_ID}.windows.jsonl"
SEQUENCES_FILE="results/processed/${EXPERIMENT_ID}.sequences.jsonl"
GRAPH_FILE="graph/output/${EXPERIMENT_ID}.json"
DYNAMIC_GRAPH_FILE="graph/output/${EXPERIMENT_ID}.dynamic.json"
SUMMARY_FILE="results/processed/${EXPERIMENT_ID}.summary.json"

FLOW_COUNT=$(( $(wc -l < "$CSV_FILE") - 1 ))
MQTT_EVENT_COUNT=$(( $(wc -l < "$MQTT_CSV_FILE") - 1 ))
PACKET_COUNT=$(( $(wc -l < "$PACKET_CSV_FILE") - 1 ))
WINDOW_COUNT=$(wc -l < "$WINDOWS_FILE")

if [ -f "$SEQUENCES_FILE" ]; then
    SEQUENCE_COUNT=$(wc -l < "$SEQUENCES_FILE")
else
    SEQUENCE_COUNT=0
fi

NODE_COUNT=$(jq '.nodes | length' "$GRAPH_FILE")
EDGE_COUNT=$(jq '.edges | length' "$GRAPH_FILE")
SNAPSHOT_COUNT=$(jq '.snapshot_count' "$DYNAMIC_GRAPH_FILE")
PCAP_BYTES=$(stat -c '%s' "$PCAP_FILE")

jq -n \
  --arg experiment_id "$EXPERIMENT_ID" \
  --argjson flow_count "$FLOW_COUNT" \
  --argjson mqtt_event_count "$MQTT_EVENT_COUNT" \
  --argjson packet_count "$PACKET_COUNT" \
  --argjson window_count "$WINDOW_COUNT" \
  --argjson sequence_count "$SEQUENCE_COUNT" \
  --argjson node_count "$NODE_COUNT" \
  --argjson edge_count "$EDGE_COUNT" \
  --argjson snapshot_count "$SNAPSHOT_COUNT" \
  --argjson pcap_bytes "$PCAP_BYTES" \
  '{
    experiment_id: $experiment_id,
    flow_count: $flow_count,
    mqtt_publish_event_count: $mqtt_event_count,
    packet_count: $packet_count,
    training_window_count: $window_count,
    training_sequence_count: $sequence_count,
    graph_node_count: $node_count,
    graph_edge_count: $edge_count,
    dynamic_snapshot_count: $snapshot_count,
    pcap_bytes: $pcap_bytes
  }' > "$SUMMARY_FILE"

echo "Wrote summary to $SUMMARY_FILE"
