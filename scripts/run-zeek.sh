#!/usr/bin/env bash
set -euo pipefail

PCAP_FILE="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  -v "$(realpath "$(dirname "$PCAP_FILE")"):/pcap:ro" \
  -v "$(realpath "$OUTPUT_DIR"):/output" \
  zeek/zeek:lts \
  bash -c "cd /output && zeek -C -r /pcap/$(basename "$PCAP_FILE")"
