import copy
import json
import tempfile
import unittest
from pathlib import Path

from preprocessing.build_training_sequences import (
    load_window_records,
    validate_window_records,
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


if __name__ == "__main__":
    unittest.main()
