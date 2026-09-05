import copy
import json
import tempfile
import unittest
from pathlib import Path

from preprocessing.build_training_sequences import (
    build_sequences,
    default_output_path,
    load_sequence_records,
    load_window_records,
    validate_sequence_records,
    validate_window_records,
    write_sequences,
)


def make_record(index):
    start = 100.0 + index * 5.0

    return {
        "schema_version": 1,
        "experiment_id": "experiment-a",
        "window_seconds": 5.0,
        "window_index": index,
        "start_ts": start,
        "end_ts": start + 5.0,
        "packet_features": {
            "packet_count": index + 1,
        },
        "graph": {
            "node_count": 3,
            "active_node_indices": [0, 1],
            "edges": [],
        },
        "label": {
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        },
    }


class LoadWindowRecordsTests(unittest.TestCase):
    def test_loads_jsonl_and_ignores_blank_lines(self):
        records = [make_record(0), make_record(1)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.jsonl"
            path.write_text(
                json.dumps(records[0])
                + "\n\n"
                + json.dumps(records[1])
                + "\n"
            )

            loaded = load_window_records(path)

        self.assertEqual(loaded, records)

    def test_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.jsonl"
            path.write_text('{"invalid":\n')

            with self.assertRaises(ValueError):
                load_window_records(path)

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.jsonl"
            path.write_text("")

            with self.assertRaises(ValueError):
                load_window_records(path)


class ValidateWindowRecordsTests(unittest.TestCase):
    def test_accepts_contiguous_compatible_windows(self):
        validate_window_records([
            make_record(0),
            make_record(1),
        ])

    def test_rejects_incompatible_records(self):
        base = [make_record(0), make_record(1)]

        cases = {}

        records = copy.deepcopy(base)
        records[1]["schema_version"] = 2
        cases["schema"] = records

        records = copy.deepcopy(base)
        records[1]["experiment_id"] = "experiment-b"
        cases["experiment"] = records

        records = copy.deepcopy(base)
        records[1]["window_seconds"] = 10.0
        cases["window_seconds"] = records

        records = copy.deepcopy(base)
        records[1]["label"]["class"] = "attack"
        cases["label"] = records

        records = copy.deepcopy(base)
        records[1]["graph"]["node_count"] = 4
        cases["node_count"] = records

        records = copy.deepcopy(base)
        records[1]["window_index"] = 2
        cases["window_index"] = records

        records = copy.deepcopy(base)
        records[1]["start_ts"] += 0.1
        records[1]["end_ts"] += 0.1
        cases["time_gap"] = records

        records = copy.deepcopy(base)
        records[1]["end_ts"] += 0.1
        cases["duration"] = records

        for name, records in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_window_records(records)


class BuildSequencesTests(unittest.TestCase):
    def test_builds_overlapping_unpadded_sequences(self):
        records = [
            make_record(index)
            for index in range(5)
        ]

        sequences = build_sequences(
            records,
            sequence_length=3,
            stride=1,
        )

        self.assertEqual(len(sequences), 3)
        self.assertEqual(
            [
                sequence["start_window_index"]
                for sequence in sequences
            ],
            [0, 1, 2],
        )
        self.assertEqual(
            [
                sequence["end_window_index"]
                for sequence in sequences
            ],
            [2, 3, 4],
        )
        self.assertEqual(
            [
                sequence["sequence_index"]
                for sequence in sequences
            ],
            [0, 1, 2],
        )
        self.assertTrue(
            all(
                len(sequence["steps"]) == 3
                for sequence in sequences
            )
        )
        self.assertTrue(
            all(
                "label" not in step
                for sequence in sequences
                for step in sequence["steps"]
            )
        )

    def test_respects_stride(self):
        records = [
            make_record(index)
            for index in range(5)
        ]

        sequences = build_sequences(
            records,
            sequence_length=2,
            stride=2,
        )

        self.assertEqual(
            [
                sequence["start_window_index"]
                for sequence in sequences
            ],
            [0, 2],
        )

    def test_rejects_invalid_parameters(self):
        records = [
            make_record(0),
            make_record(1),
        ]

        cases = [
            {
                "sequence_length": 0,
                "stride": 1,
            },
            {
                "sequence_length": 1,
                "stride": 0,
            },
            {
                "sequence_length": 3,
                "stride": 1,
            },
        ]

        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    build_sequences(
                        records,
                        **parameters,
                    )


class SequenceOutputTests(unittest.TestCase):
    def test_derives_default_output_path(self):
        input_path = Path(
            "results/processed/"
            "experiment-a.windows.jsonl"
        )

        self.assertEqual(
            default_output_path(input_path),
            Path(
                "results/processed/"
                "experiment-a.sequences.jsonl"
            ),
        )

    def test_writes_sequences_atomically(self):
        records = [
            make_record(0),
            make_record(1),
        ]
        sequences = build_sequences(
            records,
            sequence_length=2,
            stride=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory) / "sequences.jsonl"
            )

            write_sequences(
                output_path,
                sequences,
            )

            parsed = [
                json.loads(line)
                for line in output_path.read_text().splitlines()
            ]

            self.assertEqual(parsed, sequences)
            self.assertFalse(
                Path(str(output_path) + ".tmp").exists()
            )


class ValidateSequenceRecordsTests(unittest.TestCase):
    def test_loads_sequence_jsonl(self):
        records = [make_record(index) for index in range(2)]
        sequences = build_sequences(
            records,
            sequence_length=2,
            stride=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"
            write_sequences(path, sequences)

            self.assertEqual(
                load_sequence_records(path),
                sequences,
            )

    def test_rejects_empty_sequence_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "no records",
            ):
                load_sequence_records(path)

    def test_accepts_exact_regeneration(self):
        records = [make_record(index) for index in range(4)]
        sequences = build_sequences(
            records,
            sequence_length=3,
            stride=1,
        )

        validate_sequence_records(
            sequences,
            records,
            sequence_length=3,
            stride=1,
        )

    def test_rejects_modified_sequence(self):
        records = [make_record(index) for index in range(3)]
        sequences = build_sequences(
            records,
            sequence_length=2,
            stride=1,
        )
        sequences[0]["steps"][0]["packet_features"][
            "packet_count"
        ] = 999

        with self.assertRaisesRegex(
            ValueError,
            "do not match",
        ):
            validate_sequence_records(
                sequences,
                records,
                sequence_length=2,
                stride=1,
            )


if __name__ == "__main__":
    unittest.main()
