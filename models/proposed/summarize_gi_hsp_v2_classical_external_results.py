import argparse
import json
from pathlib import Path
from statistics import mean, stdev


AGGREGATED_METRICS = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "auprc",
    "auroc",
)


def load_result(path):
    path = Path(path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON result: {path}") from exc

    if not isinstance(result, dict):
        raise ValueError(f"External result must be an object: {path}")
    required = {
        "architecture",
        "dataset",
        "fold",
        "graph_view",
        "metrics",
        "model_refit_on_external_dataset",
        "seed",
        "window_count",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(
            f"External result is missing fields: {sorted(missing)}"
        )
    return result


def summarize_values(values):
    values = [float(value) for value in values]
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_grouped_recall(results, field):
    groups = {
        group
        for result in results
        for group in result[field]
    }
    output = {}

    for group in sorted(groups):
        counts = set()
        recalls = []

        for result in results:
            if group not in result[field]:
                raise ValueError(
                    f"Recall group {group!r} is absent from one fold"
                )
            values = result[field][group]
            counts.add(int(values["positive_count"]))
            recalls.append(float(values["recall"]))

        if len(counts) != 1:
            raise ValueError(
                f"Positive counts differ for recall group {group!r}"
            )
        output[group] = {
            "positive_window_count": next(iter(counts)),
            "recall": summarize_values(recalls),
        }

    return output


def aggregate_classical_external_results(results):
    if not results:
        raise ValueError("At least one external result is required")

    folds = [int(result["fold"]) for result in results]
    if len(folds) != len(set(folds)):
        raise ValueError("External results contain duplicate folds")

    architectures = {result["architecture"] for result in results}
    datasets = {result["dataset"] for result in results}
    graph_views = {result["graph_view"] for result in results}
    seeds = {int(result["seed"]) for result in results}
    window_counts = {int(result["window_count"]) for result in results}

    for values, message in (
        (architectures, "model architectures"),
        (datasets, "external datasets"),
        (graph_views, "graph views"),
        (seeds, "estimator seeds"),
        (window_counts, "external window counts"),
    ):
        if len(values) != 1:
            raise ValueError(f"Results mix {message}")

    if any(
        result["model_refit_on_external_dataset"] is not False
        for result in results
    ):
        raise ValueError("An external result reports model refitting")

    sample_count = next(iter(window_counts))
    for result in results:
        if int(result["metrics"]["sample_count"]) != sample_count:
            raise ValueError("Metric and result sample counts differ")

    metrics = {
        metric: summarize_values(
            result["metrics"][metric]
            for result in results
        )
        for metric in AGGREGATED_METRICS
    }
    scenario_results = [
        {
            **result,
            "scenario_recall": result["metrics"][
                "per_attack_scenario_recall"
            ],
        }
        for result in results
    ]

    output = {
        "aggregation_unit": "fold_macro_deterministic_reference",
        "source_run_count": len(results),
        "architecture": next(iter(architectures)),
        "dataset": next(iter(datasets)),
        "graph_view": next(iter(graph_views)),
        "folds": sorted(folds),
        "seed": next(iter(seeds)),
        "external_window_count": sample_count,
        "external_window_evaluations": sample_count * len(results),
        "metrics": metrics,
        "per_attack_scenario_recall": aggregate_grouped_recall(
            scenario_results,
            "scenario_recall",
        ),
    }

    goal_presence = [
        "per_attack_goal_recall" in result
        for result in results
    ]
    if any(goal_presence):
        if not all(goal_presence):
            raise ValueError(
                "Attack-goal recall must be present in every fold"
            )
        output["per_attack_goal_recall"] = aggregate_grouped_recall(
            results,
            "per_attack_goal_recall",
        )

    return output


def summarize_paths(paths):
    resolved = [Path(path) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate result paths are not allowed")

    output = aggregate_classical_external_results([
        load_result(path) for path in resolved
    ])
    output["source_results"] = [str(path) for path in resolved]
    return output


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate deterministic classical external fold results."
        )
    )
    parser.add_argument("result_paths", nargs="+")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = summarize_paths(arguments.result_paths)
    write_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
