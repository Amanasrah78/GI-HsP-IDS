#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="${1:-benign-mqtt-auto}"
DURATION="${2:-30}"

COMPOSE_FILE="docker/mqtt/compose.yml"
INTERFACE="br-c73cfc820324"
PCAP_FILE="capture/pcap/${EXPERIMENT_ID}.pcap"
HASH_FILE="capture/pcap/${EXPERIMENT_ID}.sha256"

echo "Experiment : $EXPERIMENT_ID"
echo "Duration   : ${DURATION}s"
echo "PCAP       : $PCAP_FILE"

TCPDUMP_PID=""

cleanup() {
    docker compose -f "$COMPOSE_FILE" stop         iot-client-1 iot-client-2 >/dev/null 2>&1 || true

    if [ -n "$TCPDUMP_PID" ] && kill -0 "$TCPDUMP_PID" 2>/dev/null; then
        sudo kill -INT "$TCPDUMP_PID" 2>/dev/null || true
        wait "$TCPDUMP_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

docker compose -f "$COMPOSE_FILE" stop iot-client-1 iot-client-2

rm -f "$PCAP_FILE" "$HASH_FILE"

# Authenticate sudo before launching tcpdump in the background.
sudo -v

sudo tcpdump \
  -i "$INTERFACE" \
  -nn \
  -Z "$(id -un)" \
  port 1883 \
  -w "$PCAP_FILE" &

TCPDUMP_PID=$!

sleep 2

docker compose -f "$COMPOSE_FILE" start iot-client-1 iot-client-2

sleep "$DURATION"

docker compose -f "$COMPOSE_FILE" stop iot-client-1 iot-client-2

sudo kill -INT "$TCPDUMP_PID"
wait "$TCPDUMP_PID" || true

sha256sum "$PCAP_FILE" > "$HASH_FILE"

echo
echo "Capture complete:"
ls -lh "$PCAP_FILE"
echo
echo "SHA-256:"
cat "$HASH_FILE"
