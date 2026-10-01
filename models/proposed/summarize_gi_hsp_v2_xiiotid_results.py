import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from models.proposed.gi_hsp_v2_result_aggregation import (
    aggregate_repeated_run_summaries,
)


def load_result(path):
    path = Path(path)

    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON result: {path}") from exc

    required = {
        "architecture",
        "fold",
        "graph_view",
        "metrics",
        "seed",
        "window_count",
    }
    missing = required - set(result)

    if missing:
        raise ValueError(
            f"External result is missing fields: {sorted(missing)}"
        )

    metrics = result["metrics"]

    if int(metrics["sample_count"]) != int(result["window_count"]):
        raise ValueError(
            f"Sample count does not match window count: {path}"
        )

    confusion_total = sum(
        int(value)
        for row in metrics["confusion_matrix"]
        for value in row
    )

    if confusion_total != int(result["window_count"]):
        raise ValueError(
            f"Confusion matrix does not match window count: {path}"
        )

    return result


def aggregate_scenario_recall(results):
    runs_by_seed = defaultdict(list)

    for result in results:
        runs_by_seed[int(result["seed"])].append(result)

    scenario_names = {
        scenario
        for result in results
        for scenario in result["metrics"][
            "per_attack_scenario_recall"
        ]
    }
    output = {}

    for scenario in sorted(scenario_names):
        counts = set()
        seed_means = []

        for seed in sorted(runs_by_seed):
            recalls = []

            for result in runs_by_seed[seed]:
                values = result["metrics"][
                    "per_attack_scenario_recall"
                ]

                if scenario not in values:
                    raise ValueError(
                        f"Scenario {scenario!r} is absent for seed {seed}"
                    )

                counts.add(int(values[scenario]["positive_count"]))
                recalls.append(float(values[scenario]["recall"]))

            seed_means.append(mean(recalls))

        if len(counts) != 1:
            raise ValueError(
                f"Scenario counts differ for {scenario!r}"
            )

        output[scenario] = {
            "positive_window_count": next(iter(counts)),
            "mean": mean(seed_means),
            "std": (
                stdev(seed_means)
                if len(seed_means) > 1
                else 0.0
            ),
            "minimum": min(seed_means),
            "maximum": max(seed_means),
        }

    return output


def summarize_paths(paths):
    paths = [Path(path) for path in paths]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate result paths are not allowed")

    results = [load_result(path) for path in paths]

    window_counts = {
        int(result["window_count"])
        for result in results
    }

    if len(window_counts) != 1:
        raise ValueError("External window counts differ across runs")

    transformed = [
        {
            "architecture": result["architecture"],
            "fold": int(result["fold"]),
            "graph_view": result["graph_view"],
            "seed": int(result["seed"]),
            "test_metrics": result["metrics"],
        }
        for result in results
    ]

    aggregate = aggregate_repeated_run_summaries(transformed)
    scenario_recall = aggregate_scenario_recall(results)
    positive_count = sum(
        values["positive_window_count"]
        for values in scenario_recall.values()
    )
    window_count = next(iter(window_counts))

    if positive_count > window_count:
        raise ValueError(
            "Positive-window count exceeds external window count"
        )

    aggregate.pop("positive_window_count_per_seed", None)
    aggregate.pop("positive_window_evaluations", None)
    aggregate.update({
        "dataset": "x-iiotid",
        "evaluation_role": (
            "external_cross_dataset_mqtt_stress_test"
        ),
        "external_window_count": window_count,
        "external_positive_window_count": positive_count,
        "external_negative_window_count": (
            window_count - positive_count
        ),
        "per_attack_scenario_recall": scenario_recall,
        "source_summaries": [str(path) for path in paths],
    })

    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated GI-HSP V2 X-IIoTID evaluations."
        )
    )
    parser.add_argument(
        "result_paths",
        nargs="+",
        help="Paths to xiiotid_test_metrics.json files.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    result = summarize_paths(args.result_paths)
    rendered = json.dumps(result, indent=2, sort_keys=True)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")

    print(rendered)


if __name__ == "__main__":
    main()
