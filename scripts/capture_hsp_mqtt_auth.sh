#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="$1"
DURATION="${2:-60}"

COMPOSE_FILE="docker/mqtt/compose.yml"
INTERFACE="br-c73cfc820324"

PCAP_FILE="capture/pcap/${EXPERIMENT_ID}.pcap"
HASH_FILE="capture/pcap/${EXPERIMENT_ID}.sha256"
TIMING_FILE="capture/pcap/${EXPERIMENT_ID}.timing.json"

ATTACK_COMMAND="repeated mosquitto_pub authentication failures against 172.30.0.12:1883"

TCPDUMP_PID=""

cleanup() {
    docker compose -f "$COMPOSE_FILE" stop \
        iot-client-1 iot-client-2 iot-subscriber-1 \
        >/dev/null 2>&1 || true

    if [ -n "$TCPDUMP_PID" ] && kill -0 "$TCPDUMP_PID" 2>/dev/null; then
        sudo kill -INT "$TCPDUMP_PID" 2>/dev/null || true
        wait "$TCPDUMP_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

docker compose -f "$COMPOSE_FILE" stop \
    iot-client-1 iot-client-2 iot-subscriber-1

rm -f "$PCAP_FILE" "$HASH_FILE" "$TIMING_FILE"

sudo -v

sudo tcpdump \
    -i "$INTERFACE" \
    -nn \
    -Z "$(id -un)" \
    port 1883 \
    -w "$PCAP_FILE" &

TCPDUMP_PID=$!

sleep 2

MEASUREMENT_START_TS="$(date +%s.%N)"

docker compose -f "$COMPOSE_FILE" start \
    iot-subscriber-1 iot-client-1 iot-client-2

sleep 10

ATTACK_END=$((SECONDS + DURATION - 10))

while [ "$SECONDS" -lt "$ATTACK_END" ]; do
    docker exec hsp-attacker mosquitto_pub \
        -h 172.30.0.12 \
        -p 1883 \
        -u gi-hsp-user \
        -P wrong-password \
        -t test/auth \
        -m test \
        >/dev/null 2>&1 || true

    sleep 1
done

MEASUREMENT_END_TS="$(date +%s.%N)"

python3 - "$TIMING_FILE" \
    "$MEASUREMENT_START_TS" \
    "$MEASUREMENT_END_TS" <<'PYTIMING'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
start_ts = float(sys.argv[2])
end_ts = float(sys.argv[3])

data = {
    "measurement_start_ts": start_ts,
    "measurement_end_ts": end_ts,
    "measurement_start_utc": datetime.fromtimestamp(
        start_ts, tz=timezone.utc
    ).isoformat(),
    "measurement_end_utc": datetime.fromtimestamp(
        end_ts, tz=timezone.utc
    ).isoformat(),
}

output.write_text(
    json.dumps(data, indent=2) + "\n",
    encoding="utf-8",
)
PYTIMING

sudo kill -INT "$TCPDUMP_PID"
wait "$TCPDUMP_PID" || true
TCPDUMP_PID=""

docker compose -f "$COMPOSE_FILE" stop \
    iot-client-1 iot-client-2 iot-subscriber-1

sha256sum "$PCAP_FILE" > "$HASH_FILE"

echo
echo "Capture complete:"
ls -lh "$PCAP_FILE"
echo
echo "SHA-256:"
cat "$HASH_FILE"
echo
echo "Attack command:"
echo "$ATTACK_COMMAND"
