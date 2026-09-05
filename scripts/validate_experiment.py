import argparse
import hashlib
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    args = parser.parse_args()

    experiment_id = args.experiment_id

    manifest_path = Path(f"experiments/{experiment_id}.yaml")
    pcap_path = Path(f"capture/pcap/{experiment_id}.pcap")
    csv_path = Path(f"results/processed/{experiment_id}.csv")
    graph_path = Path(f"graph/output/{experiment_id}.json")
    conn_path = Path(f"results/raw/{experiment_id}/zeek/conn.log")

    required = {
        "manifest": manifest_path,
        "pcap": pcap_path,
        "zeek_conn": conn_path,
        "flow_csv": csv_path,
        "graph_json": graph_path,
    }

    ok = True

    for name, path in required.items():
        if path.exists():
            print(f"[OK]   {name}: {path}")
        else:
            print(f"[FAIL] {name}: {path}")
            ok = False

    if not manifest_path.exists() or not pcap_path.exists():
        raise SystemExit(1)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    expected_hash = manifest["capture"]["sha256"]
    actual_hash = sha256_file(pcap_path)

    if expected_hash == actual_hash:
        print("[OK]   PCAP SHA-256 matches manifest")
    else:
        print("[FAIL] PCAP SHA-256 mismatch")
        ok = False

    if manifest["experiment_id"] == experiment_id:
        print("[OK]   Experiment ID matches manifest")
    else:
        print("[FAIL] Experiment ID mismatch")
        ok = False

    if not ok:
        raise SystemExit(1)

    print(f"\nExperiment '{experiment_id}' validation PASSED")


if __name__ == "__main__":
    main()
