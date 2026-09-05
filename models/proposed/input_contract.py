PACKET_FEATURE_NAMES = (
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
)

NODE_FEATURE_NAMES = (
    "active",
    "in_neighbor_count",
    "out_neighbor_count",
    "in_event_count",
    "out_event_count",
    "in_payload_bytes",
    "out_payload_bytes",
)

DEFAULT_SEQUENCE_LENGTH = 10
