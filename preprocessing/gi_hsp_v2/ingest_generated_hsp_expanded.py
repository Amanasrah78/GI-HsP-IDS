import argparse
import json
from pathlib import Path

from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
    sha256_file,
)
from preprocessing.gi_hsp_v2.ingest_generated_hsp import (
    ingest_generated_hsp,
)


DEFAULT_PROCESSING_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)


def ingest_expanded_hsp(processing_protocol_path=DEFAULT_PROCESSING_PROTOCOL):
    processing = load_verified_processing_protocol(processing_protocol_path)
    capture_protocol = processing["capture_protocol_value"]
    capture_ids = [
        record["experiment_id"] for record in capture_protocol["schedule"]
    ]
    result = ingest_generated_hsp(
        processing["capture_protocol"],
        protocol=capture_protocol,
        capture_ids=capture_ids,
        output_path=processing["canonical_store"],
        dataset_name=processing["dataset"],
    )
    result["processing_protocol_path"] = str(Path(processing_protocol_path))
    result["processing_protocol_sha256"] = processing[
        "processing_protocol_sha256"
    ]
    result["verified_completion_count"] = processing["artifact_summary"][
        "capture_count"
    ]
    summary_path = Path(f"{processing['canonical_store']}.summary.json")
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Ingest the completed expanded generated-HsP study."
    )
    parser.add_argument(
        "--processing-protocol", default=DEFAULT_PROCESSING_PROTOCOL
    )
    args = parser.parse_args()
    result = ingest_expanded_hsp(args.processing_protocol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
