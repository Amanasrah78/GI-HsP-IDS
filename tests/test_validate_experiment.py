import copy
import unittest

from scripts.validate_experiment import (
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


def make_records():
    records = []

    for index in range(2):
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


if __name__ == "__main__":
    unittest.main()
