import csv
import io

import pytest

from preprocessing.gi_hsp_v2.cse_cic_ids2018_adapter import (
    DATASET_NAME,
)
from preprocessing.gi_hsp_v2.ingest_cse_cic_ids2018 import (
    DEFAULT_PROTOCOL,
    REQUIRED_COLUMNS,
    load_protocol,
    normalized_headers,
)


def reader_for(header):
    return csv.DictReader(
        io.StringIO(header + "\n")
    )


def test_frozen_protocol_loads():
    protocol, digest = load_protocol(DEFAULT_PROTOCOL)

    assert protocol["dataset"] == DATASET_NAME
    assert protocol["status"] == "frozen_before_ingestion"
    assert len(digest) == 64


def test_protocol_records_identity_subset_scope():
    protocol, _ = load_protocol(DEFAULT_PROTOCOL)

    assert protocol["scope"]["included_file_count"] == 1
    assert protocol["scope"]["excluded_file_count"] == 9
    assert protocol["scope"]["exclusion_rule"] == (
        "missing_endpoint_identity_fields"
    )


def test_protocol_records_temporal_policy():
    protocol, _ = load_protocol(DEFAULT_PROTOCOL)
    temporal = protocol["temporal_representation"]

    assert temporal["bin_seconds"] == 5
    assert temporal["sequence_length"] == 10
    assert temporal["window_length_seconds"] == 50
    assert temporal["evaluation_stride_seconds"] == 50
    assert protocol["labels"]["window_policy"] == (
        "any_attack_flow"
    )


def test_headers_are_trimmed():
    header = ",".join(
        f" {name} "
        for name in sorted(REQUIRED_COLUMNS)
    )
    reader = reader_for(header)
    result = normalized_headers(reader, "source.csv")

    assert set(result) == REQUIRED_COLUMNS


def test_missing_required_header_is_rejected():
    fields = sorted(REQUIRED_COLUMNS - {"Src IP"})
    reader = reader_for(",".join(fields))

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        normalized_headers(reader, "source.csv")


def test_duplicate_trimmed_header_is_rejected():
    fields = sorted(REQUIRED_COLUMNS)
    header = ",".join(fields + [" Src IP "])
    reader = reader_for(header)

    with pytest.raises(
        ValueError,
        match="duplicate headers",
    ):
        normalized_headers(reader, "source.csv")
