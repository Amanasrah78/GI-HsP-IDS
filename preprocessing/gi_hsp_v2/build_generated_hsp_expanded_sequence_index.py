import argparse
import json

from preprocessing.gi_hsp_v2.build_generated_hsp_sequence_index import (
    build_sequence_index as build_base_sequence_index,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    open_sequence_index,
    set_metadata,
)


DEFAULT_PROCESSING_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)


def build_expanded_sequence_index(
    processing_protocol_path=DEFAULT_PROCESSING_PROTOCOL,
):
    processing = load_verified_processing_protocol(processing_protocol_path)
    capture_protocol = processing["capture_protocol_value"]
    capture_ids = [
        record["experiment_id"] for record in capture_protocol["schedule"]
    ]
    result = build_base_sequence_index(
        processing["capture_protocol"],
        output_path=processing["sequence_index"],
        protocol=capture_protocol,
        capture_ids=capture_ids,
        database_path=processing["canonical_store"],
    )
    connection = open_sequence_index(processing["sequence_index"])
    try:
        set_metadata(
            connection,
            "processing_protocol",
            {
                "path": str(processing_protocol_path),
                "sha256": processing["processing_protocol_sha256"],
            },
        )
        connection.commit()
    finally:
        connection.close()
    result["processing_protocol_sha256"] = processing[
        "processing_protocol_sha256"
    ]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build the expanded generated-HsP evaluation index."
    )
    parser.add_argument(
        "--processing-protocol", default=DEFAULT_PROCESSING_PROTOCOL
    )
    args = parser.parse_args()
    result = build_expanded_sequence_index(args.processing_protocol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
