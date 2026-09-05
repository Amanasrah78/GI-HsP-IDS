import argparse
import csv
import json
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument(
        "--feature-config",
        default="configs/flow_features.yaml",
    )
    args = parser.parse_args()

    experiment_id = args.experiment_id

    flow_csv = Path(
        f"results/processed/{experiment_id}.csv"
    )
    dynamic_graph = Path(
        f"graph/output/{experiment_id}.dynamic.json"
    )
    manifest_path = Path(
        f"experiments/{experiment_id}.yaml"
    )
    feature_config_path = Path(args.feature_config)

    manifest = load_yaml(manifest_path)
    feature_config = load_yaml(feature_config_path)

    with flow_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        flow_rows = list(csv.DictReader(f))

    with dynamic_graph.open(
        "r",
        encoding="utf-8",
    ) as f:
        graph_data = json.load(f)

    print("experiment_id:", experiment_id)
    print("label:", manifest["label"])
    print("flow_rows:", len(flow_rows))
    print("graph_snapshots:", graph_data["snapshot_count"])
    print(
        "numeric_features:",
        feature_config["numeric_features"],
    )
    print(
        "categorical_features:",
        feature_config["categorical_features"],
    )


if __name__ == "__main__":
    main()
