import unittest

from models.proposed.input_assembly import assemble_sequence_inputs
from models.proposed.input_contract import (
    NODE_FEATURE_NAMES,
    PACKET_FEATURE_NAMES,
)


def make_sequence_record():
    steps = []

    for window_index in range(10):
        packet_features = {
            name: index + window_index
            for index, name in enumerate(PACKET_FEATURE_NAMES)
        }

        node_features = [
            {
                name: feature_index + node_index
                for feature_index, name in enumerate(NODE_FEATURE_NAMES)
            }
            for node_index in range(2)
        ]

        steps.append({
            "window_index": window_index,
            "start_ts": float(window_index),
            "end_ts": float(window_index + 1),
            "packet_features": packet_features,
            "graph": {
                "node_count": 2,
                "active_node_indices": [0, 1],
                "node_features": node_features,
                "edges": [
                    {
                        "source_index": 0,
                        "target_index": 1,
                        "event_count": 2,
                        "payload_bytes": 100,
                    }
                ],
            },
        })

    return {
        "schema_version": 1,
        "window_schema_version": 2,
        "experiment_id": "example",
        "window_seconds": 5.0,
        "sequence_index": 0,
        "sequence_length": 10,
        "start_window_index": 0,
        "end_window_index": 9,
        "start_ts": 0.0,
        "end_ts": 10.0,
        "steps": steps,
        "label": {
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        },
    }


class ModelInputAssemblyTests(unittest.TestCase):
    def test_assembles_expected_shapes(self):
        result = assemble_sequence_inputs(make_sequence_record())

        self.assertEqual(len(result["packet_features"]), 10)
        self.assertEqual(len(result["packet_features"][0]), 10)

        self.assertEqual(len(result["node_features"]), 10)
        self.assertEqual(len(result["node_features"][0]), 2)
        self.assertEqual(len(result["node_features"][0][0]), 7)

        self.assertEqual(len(result["edges"]), 10)
        self.assertEqual(result["node_count"], 2)

    def test_uses_contract_feature_order(self):
        result = assemble_sequence_inputs(make_sequence_record())

        self.assertEqual(
            result["packet_features"][0],
            [float(index) for index in range(10)],
        )
        self.assertEqual(
            result["node_features"][0][0],
            [float(index) for index in range(7)],
        )

    def test_preserves_edges_and_label(self):
        record = make_sequence_record()
        result = assemble_sequence_inputs(record)

        self.assertEqual(
            result["edges"][0][0],
            {
                "source_index": 0,
                "target_index": 1,
                "event_count": 2,
                "payload_bytes": 100,
            },
        )
        self.assertEqual(result["label"], record["label"])

    def test_rejects_wrong_sequence_length(self):
        record = make_sequence_record()
        record["sequence_length"] = 9

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected sequence length",
        ):
            assemble_sequence_inputs(record)

    def test_rejects_changing_node_axis(self):
        record = make_sequence_record()
        record["steps"][1]["graph"]["node_count"] = 3

        with self.assertRaisesRegex(
            ValueError,
            "Graph node axis changes within sequence",
        ):
            assemble_sequence_inputs(record)

    def test_rejects_wrong_node_feature_count(self):
        record = make_sequence_record()
        record["steps"][0]["graph"]["node_features"].pop()

        with self.assertRaisesRegex(
            ValueError,
            "Node feature count does not match node count",
        ):
            assemble_sequence_inputs(record)


class ModelInputAdjacencyTests(unittest.TestCase):
    def test_builds_directed_event_count_adjacency(self):
        record = {
            "sequence_length": 10,
            "label": {
                "class": "benign",
                "attack_goal": "none",
                "hsp_family": "none",
            },
            "steps": [],
        }

        for _ in range(10):
            record["steps"].append({
                "packet_features": {
                    "packet_count": 1,
                    "frame_bytes": 2,
                    "tcp_payload_bytes": 3,
                    "mean_frame_len": 4,
                    "mean_tcp_payload_len": 5,
                    "mean_interarrival_seconds": 6,
                    "std_interarrival_seconds": 7,
                    "max_interarrival_seconds": 8,
                    "suspected_retransmission_count": 9,
                    "previous_segment_not_captured_count": 10,
                },
                "graph": {
                    "node_count": 3,
                    "node_features": [
                        {
                            "active": 1,
                            "in_neighbor_count": 0,
                            "out_neighbor_count": 1,
                            "in_event_count": 0,
                            "out_event_count": 2,
                            "in_payload_bytes": 0,
                            "out_payload_bytes": 200,
                        }
                        for _ in range(3)
                    ],
                    "edges": [
                        {
                            "source_index": 0,
                            "target_index": 1,
                            "event_count": 2,
                            "payload_bytes": 200,
                        },
                        {
                            "source_index": 2,
                            "target_index": 0,
                            "event_count": 1,
                            "payload_bytes": 100,
                        },
                    ],
                },
            })

        assembled = assemble_sequence_inputs(record)

        self.assertEqual(
            assembled["adjacency"][0],
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
