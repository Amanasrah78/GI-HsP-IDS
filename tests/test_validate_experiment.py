import copy
import tempfile
import unittest
from pathlib import Path

from preprocessing.build_training_sequences import (
    build_sequences,
    write_sequences,
)
from scripts.validate_experiment import (
    validate_training_sequences,
    validate_training_windows,
)


LABEL = {
    "class": "benign",
    "attack_goal": "none",
    "hsp_family": "none",
}


def make_dynamic_graph():
    return {
        "window_seconds": 5.0,
        "snapshots": [
            {
                "window_index": 0,
                "start_ts": 100.0,
                "end_ts": 105.0,
                "nodes": [{"id": "a"}],
                "edges": [],
            },
            {
                "window_index": 1,
                "start_ts": 105.0,
                "end_ts": 110.0,
                "nodes": [{"id": "a"}, {"id": "b"}],
                "edges": [],
            },
        ],
    }


def make_records(count=2):
    records = []

    for index in range(count):
        start = 100.0 + index * 5.0
        records.append({
            "schema_version": 1,
            "experiment_id": "experiment-001",
            "window_seconds": 5.0,
            "window_index": index,
            "start_ts": start,
            "end_ts": start + 5.0,
            "packet_features": {"packet_count": index},
            "graph": {
                "node_count": 2,
                "active_node_indices": [0],
                "edges": [],
            },
            "label": LABEL,
        })

    return records


class ValidateTrainingWindowsTests(unittest.TestCase):
    def test_accepts_matching_artifacts(self):
        validate_training_windows(
            make_records(),
            "experiment-001",
            LABEL,
            make_dynamic_graph(),
        )

    def test_rejects_requested_experiment_mismatch(self):
        with self.assertRaisesRegex(
            ValueError,
            "requested experiment",
        ):
            validate_training_windows(
                make_records(),
                "experiment-002",
                LABEL,
                make_dynamic_graph(),
            )

    def test_rejects_manifest_label_mismatch(self):
        label = copy.deepcopy(LABEL)
        label["class"] = "attack"

        with self.assertRaisesRegex(
            ValueError,
            "manifest",
        ):
            validate_training_windows(
                make_records(),
                "experiment-001",
                label,
                make_dynamic_graph(),
            )

    def test_rejects_snapshot_count_mismatch(self):
        graph = make_dynamic_graph()
        graph["snapshots"].pop()

        with self.assertRaisesRegex(
            ValueError,
            "Window count",
        ):
            validate_training_windows(
                make_records(),
                "experiment-001",
                LABEL,
                graph,
            )

    def test_rejects_boundary_mismatch(self):
        graph = make_dynamic_graph()
        graph["snapshots"][1]["start_ts"] += 0.1

        with self.assertRaisesRegex(
            ValueError,
            "boundary",
        ):
            validate_training_windows(
                make_records(),
                "experiment-001",
                LABEL,
                graph,
            )

    def test_rejects_stable_node_axis_mismatch(self):
        records = make_records()
        records[0]["graph"]["node_count"] = 3
        records[1]["graph"]["node_count"] = 3

        with self.assertRaisesRegex(
            ValueError,
            "node count",
        ):
            validate_training_windows(
                records,
                "experiment-001",
                LABEL,
                make_dynamic_graph(),
            )


class ValidateTrainingSequencesTests(unittest.TestCase):
    def test_accepts_absent_sequences_for_short_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"

            self.assertFalse(
                validate_training_sequences(
                    path,
                    make_records(),
                )
            )

    def test_rejects_stale_sequences_for_short_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"
            path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "insufficient window count",
            ):
                validate_training_sequences(
                    path,
                    make_records(),
                )

    def test_rejects_missing_sequences_when_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"

            with self.assertRaisesRegex(
                ValueError,
                "required but missing",
            ):
                validate_training_sequences(
                    path,
                    make_records(10),
                )

    def test_accepts_exact_production_sequences(self):
        records = make_records(11)
        sequences = build_sequences(
            records,
            sequence_length=10,
            stride=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"
            write_sequences(path, sequences)

            self.assertTrue(
                validate_training_sequences(
                    path,
                    records,
                )
            )

    def test_rejects_mismatched_production_sequences(self):
        records = make_records(10)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.jsonl"
            path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "do not match",
            ):
                validate_training_sequences(
                    path,
                    records,
                )


if __name__ == "__main__":
    unittest.main()
