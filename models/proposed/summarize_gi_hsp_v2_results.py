import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_result_aggregation import (
    aggregate_repeated_run_summaries,
    aggregate_run_summaries,
)


def load_summary(path):
    path = Path(path)

    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON summary: {path}") from exc

    if not isinstance(summary, dict):
        raise ValueError(f"Run summary must be an object: {path}")

    return summary


def summarize_paths(paths, repeated=False):
    resolved_paths = [Path(path) for path in paths]

    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Duplicate summary paths are not allowed")

    summaries = [
        load_summary(path)
        for path in resolved_paths
    ]
    aggregate = (
        aggregate_repeated_run_summaries
        if repeated
        else aggregate_run_summaries
    )
    result = aggregate(summaries)
    result["source_summaries"] = [
        str(path)
        for path in resolved_paths
    ]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate comparable GI-HSP V2 experiment summaries."
    )
    parser.add_argument(
        "summary_paths",
        nargs="+",
        help="Paths to compatible summary.json files.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--repeated",
        action="store_true",
        help="Aggregate fold-macro results across random seeds.",
    )
    arguments = parser.parse_args()

    result = summarize_paths(
        arguments.summary_paths,
        repeated=arguments.repeated,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)

    if arguments.output:
        output_path = Path(arguments.output)

        if output_path.exists():
            raise FileExistsError(
                f"Output file already exists: {output_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")

    print(rendered)


if __name__ == "__main__":
    main()
