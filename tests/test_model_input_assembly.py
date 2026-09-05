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


if __name__ == "__main__":
    unittest.main()
