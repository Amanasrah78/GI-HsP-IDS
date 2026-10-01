#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

protocol="configs/gi_hsp_v2_generated_hsp_expanded.yaml"
python=".venv/bin/python"

sudo -n true || {
    echo "ERROR: sudo credentials are unavailable"
    exit 1
}

while true
do
    sudo -n true || exit 1
    sleep 45
done >/dev/null 2>&1 &

sudo_keepalive_pid=$!

cleanup()
{
    kill "$sudo_keepalive_pid" 2>/dev/null || true
}

trap cleanup EXIT

for block in 03 04 05 06 07 08 09 10
do
    for slot in 01 02 03 04 05 06
    do
        experiment_id="hsp-expanded-b${block}-s${slot}"
        completion=\
"results/gi_hsp_v2/generated_hsp_expanded/evidence/${experiment_id}.completion.json"

        if [ -f "$completion" ]; then
            "$python" - "$completion" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text())

assert record["success"] is True
assert record["validation_returncode"] == 0
assert record["validation_failures"] == []
PY
            printf '{"status":"skipped_complete","experiment_id":"%s"}\n' \
                "$experiment_id"
            continue
        fi

        printf '{"status":"starting","experiment_id":"%s"}\n' \
            "$experiment_id"

        "$python" -u -m \
            preprocessing.gi_hsp_v2.capture_generated_hsp_expanded \
            "$experiment_id" \
            --protocol "$protocol"

        "$python" - "$completion" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text())

assert record["success"] is True
assert record["validation_returncode"] == 0
assert record["validation_failures"] == []
PY

        printf '{"status":"completed","experiment_id":"%s"}\n' \
            "$experiment_id"
    done
done

printf '%s\n' \
    '{"status":"matrix_completed","blocks":"03-10","capture_count":48}'
