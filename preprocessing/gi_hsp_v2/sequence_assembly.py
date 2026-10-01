import math

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    validate_graph_view,
)
from preprocessing.gi_hsp_v2.flow_step_features import (
    flow_feature_vector,
)
from preprocessing.gi_hsp_v2.topology_step_features import (
    assemble_topology_step,
)


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")

    return int(number)


def _validate_window(window):
    required = {
        "capture_id",
        "source_label",
        "binary_label",
        "start_second",
        "end_second_exclusive",
    }
    missing = required - set(window)

    if missing:
        raise ValueError(
            f"Window is missing fields: {sorted(missing)}"
        )

    start = _integer(
        window["start_second"],
        "start_second",
    )
    end = _integer(
        window["end_second_exclusive"],
        "end_second_exclusive",
    )

    if end <= start:
        raise ValueError(
            "end_second_exclusive must exceed start_second"
        )

    binary_label = _integer(
        window["binary_label"],
        "binary_label",
    )

    if binary_label not in (0, 1):
        raise ValueError("binary_label must be 0 or 1")

    return start, end, binary_label


def _group_records(
    records,
    window,
    start,
    end,
    binary_label,
):
    grouped = {
        second: []
        for second in range(start, end)
    }
    mixed_window_dataset = (
        "cse_cic_ids2018_identity_subset"
    )
    observed_binary_labels = set()
    uses_mixed_window_policy = False

    for record in records:
        if record["capture_id"] != window["capture_id"]:
            raise ValueError(
                "A flow record belongs to a different capture"
            )

        record_label = int(record["binary_label"])
        record_source_label = str(
            record["source_label"]
        )
        record_dataset = str(record.get("dataset") or "")

        if record_dataset == mixed_window_dataset:
            uses_mixed_window_policy = True

            expected_source_label = (
                "DDoS attacks-LOIC-HTTP"
                if record_label == 1
                else "Benign"
            )

            if record_source_label != expected_source_label:
                raise ValueError(
                    "A CSE-CIC flow source label disagrees "
                    "with its binary label"
                )

            observed_binary_labels.add(record_label)
        else:
            if record_label != binary_label:
                raise ValueError(
                    "A flow record label differs from "
                    "the window label"
                )

            if record_source_label != window["source_label"]:
                raise ValueError(
                    "A flow source label differs from "
                    "the window label"
                )

        second = math.floor(float(record["timestamp"]))

        if second < start or second >= end:
            raise ValueError(
                "A flow record lies outside the window"
            )

        grouped[second].append(record)

    if uses_mixed_window_policy:
        if not observed_binary_labels:
            raise ValueError(
                "A CSE-CIC window contains no flow labels"
            )

        expected_window_label = max(
            observed_binary_labels
        )

        if binary_label != expected_window_label:
            raise ValueError(
                "The CSE-CIC window label does not equal "
                "the maximum constituent flow label"
            )

        expected_window_source_label = (
            "DDoS attacks-LOIC-HTTP"
            if expected_window_label == 1
            else "Benign"
        )

        if (
            str(window["source_label"])
            != expected_window_source_label
        ):
            raise ValueError(
                "The CSE-CIC window source label disagrees "
                "with its aggregate binary label"
            )

    return grouped


def _global_node_ids(topology_steps, graph_view):
    if graph_view == "client_broker_role_collapsed":
        return ["client", "broker"]

    return sorted({
        node_id
        for step in topology_steps
        for node_id in step["node_ids"]
    })


def _zero_node_vector():
    return [
        0.0
        for _ in NODE_FEATURE_NAMES
    ]


def _remap_topology_step(step, global_node_ids):
    global_indices = {
        node_id: index
        for index, node_id in enumerate(global_node_ids)
    }

    node_vectors = [
        _zero_node_vector()
        for _ in global_node_ids
    ]

    for local_index, node_id in enumerate(step["node_ids"]):
        global_index = global_indices[node_id]
        features = step["node_features"][local_index]

        node_vectors[global_index] = [
            float(features[name])
            for name in NODE_FEATURE_NAMES
        ]

    edges = []

    for edge in step["edges"]:
        source_id = step["node_ids"][edge["source_index"]]
        destination_id = step["node_ids"][
            edge["destination_index"]
        ]

        edges.append(
            {
                "source_index": global_indices[source_id],
                "destination_index": global_indices[
                    destination_id
                ],
                "features": [
                    float(edge[name])
                    for name in EDGE_FEATURE_NAMES
                ],
            }
        )

    return node_vectors, edges


def assemble_temporal_sequence(
    records,
    window,
    graph_view,
    bin_seconds=1,
):
    graph_view = validate_graph_view(graph_view)
    start, end, binary_label = _validate_window(window)
    bin_seconds = _integer(bin_seconds, "bin_seconds")

    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")

    if (end - start) % bin_seconds != 0:
        raise ValueError(
            "Window duration must be divisible by bin_seconds"
        )

    records = list(records)

    grouped = _group_records(
        records,
        window,
        start,
        end,
        binary_label,
    )

    flow_steps = []
    topology_steps = []

    for step_start in range(start, end, bin_seconds):
        step_records = [
            record
            for second in range(
                step_start,
                step_start + bin_seconds,
            )
            for record in grouped[second]
        ]
        flow_steps.append(
            flow_feature_vector(
                step_records,
                step_start=step_start,
                bin_seconds=bin_seconds,
            )
        )
        topology_steps.append(
            assemble_topology_step(
                step_records,
                graph_view,
                step_start=step_start,
                bin_seconds=bin_seconds,
            )
        )

    node_ids = _global_node_ids(
        topology_steps,
        graph_view,
    )
    node_steps = []
    edge_steps = []

    for topology_step in topology_steps:
        node_vectors, edges = _remap_topology_step(
            topology_step,
            node_ids,
        )
        node_steps.append(node_vectors)
        edge_steps.append(edges)

    actual_active_seconds = sum(
        bool(grouped[second])
        for second in range(start, end)
    )

    if (
        "active_second_count" in window
        and _integer(
            window["active_second_count"],
            "active_second_count",
        )
        != actual_active_seconds
    ):
        raise ValueError(
            "Window active_second_count does not match its records"
        )

    if any(
        len(vector) != len(FLOW_FEATURE_NAMES)
        for vector in flow_steps
    ):
        raise RuntimeError("Flow feature width violates contract")

    return {
        "window_id": window.get("window_id"),
        "capture_id": window["capture_id"],
        "source_label": window["source_label"],
        "binary_label": binary_label,
        "start_second": start,
        "end_second_exclusive": end,
        "sequence_length": (end - start) // bin_seconds,
        "bin_seconds": bin_seconds,
        "active_second_count": actual_active_seconds,
        "graph_view": graph_view,
        "node_ids": node_ids,
        "flow_features": flow_steps,
        "node_features": node_steps,
        "edges": edge_steps,
    }
