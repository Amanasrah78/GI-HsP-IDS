import unittest

from models.proposed.input_contract import (
    DEFAULT_SEQUENCE_LENGTH,
    NODE_FEATURE_NAMES,
    PACKET_FEATURE_NAMES,
)


class ModelInputContractTests(unittest.TestCase):
    def test_packet_feature_contract(self):
        self.assertEqual(len(PACKET_FEATURE_NAMES), 10)
        self.assertEqual(
            PACKET_FEATURE_NAMES,
            (
                "packet_count",
                "frame_bytes",
                "tcp_payload_bytes",
                "mean_frame_len",
                "mean_tcp_payload_len",
                "mean_interarrival_seconds",
                "std_interarrival_seconds",
                "max_interarrival_seconds",
                "suspected_retransmission_count",
                "previous_segment_not_captured_count",
            ),
        )

    def test_node_feature_contract(self):
        self.assertEqual(len(NODE_FEATURE_NAMES), 7)
        self.assertEqual(
            NODE_FEATURE_NAMES,
            (
                "active",
                "in_neighbor_count",
                "out_neighbor_count",
                "in_event_count",
                "out_event_count",
                "in_payload_bytes",
                "out_payload_bytes",
            ),
        )

    def test_default_sequence_length(self):
        self.assertEqual(DEFAULT_SEQUENCE_LENGTH, 10)


if __name__ == "__main__":
    unittest.main()
