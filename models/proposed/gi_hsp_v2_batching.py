import torch


def _validate_item(item, sequence_length, graph_view):
    flow_features = item["flow_features"]
    node_features = item["node_features"]
    edge_features = item["edge_features"]
    edge_mask = item.get("edge_mask")
    node_mask = item["node_mask"]

    if flow_features.ndim != 2:
        raise ValueError(
            "flow_features must have shape [time, features]"
        )

    if node_features.ndim != 3:
        raise ValueError(
            "node_features must have shape [time, nodes, features]"
        )

    if edge_features.ndim != 4:
        raise ValueError(
            "edge_features must have shape "
            "[time, nodes, nodes, features]"
        )

    if node_mask.ndim != 1:
        raise ValueError(
            "node_mask must have shape [nodes]"
        )

    if flow_features.shape[0] != sequence_length:
        raise ValueError(
            "Sequence lengths must match within a batch"
        )

    node_count = node_features.shape[1]

    if node_features.shape[0] != sequence_length:
        raise ValueError(
            "Node sequence length differs from flow sequence"
        )

    if edge_features.shape[:3] != (
        sequence_length,
        node_count,
        node_count,
    ):
        raise ValueError(
            "Edge tensor dimensions do not match node tensor"
        )

    if node_mask.shape[0] != node_count:
        raise ValueError(
            "Node mask length does not match node count"
        )

    if edge_mask is not None:
        if edge_mask.dtype != torch.bool:
            raise ValueError(
                "edge_mask must be boolean"
            )

        if edge_mask.shape != (
            sequence_length,
            node_count,
            node_count,
        ):
            raise ValueError(
                "Edge mask dimensions do not match edge tensor"
            )

    if item["graph_view"] != graph_view:
        raise ValueError(
            "Graph views must match within a batch"
        )


def collate_gi_hsp_v2(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")

    sequence_length = items[0]["flow_features"].shape[0]
    graph_view = items[0]["graph_view"]

    for item in items:
        _validate_item(
            item,
            sequence_length,
            graph_view,
        )

    max_node_count = max(
        item["node_features"].shape[1]
        for item in items
    )
    batch_size = len(items)
    first = items[0]

    node_feature_count = first["node_features"].shape[2]
    edge_feature_count = first["edge_features"].shape[3]

    flow_batch = torch.stack([
        item["flow_features"]
        for item in items
    ])

    node_batch = torch.zeros(
        (
            batch_size,
            sequence_length,
            max_node_count,
            node_feature_count,
        ),
        dtype=first["node_features"].dtype,
        device=first["node_features"].device,
    )

    edge_batch = torch.zeros(
        (
            batch_size,
            sequence_length,
            max_node_count,
            max_node_count,
            edge_feature_count,
        ),
        dtype=first["edge_features"].dtype,
        device=first["edge_features"].device,
    )

    edge_mask_batch = torch.zeros(
        (
            batch_size,
            sequence_length,
            max_node_count,
            max_node_count,
        ),
        dtype=torch.bool,
        device=first["edge_features"].device,
    )

    node_mask_batch = torch.zeros(
        (
            batch_size,
            max_node_count,
        ),
        dtype=torch.bool,
        device=first["node_mask"].device,
    )

    for batch_index, item in enumerate(items):
        node_count = item["node_features"].shape[1]

        node_batch[
            batch_index,
            :,
            :node_count,
            :,
        ] = item["node_features"]

        edge_batch[
            batch_index,
            :,
            :node_count,
            :node_count,
            :,
        ] = item["edge_features"]

        item_edge_mask = item.get("edge_mask")

        if item_edge_mask is None:
            item_edge_mask = torch.any(
                item["edge_features"] != 0,
                dim=-1,
            )

        edge_mask_batch[
            batch_index,
            :,
            :node_count,
            :node_count,
        ] = item_edge_mask

        node_mask_batch[
            batch_index,
            :node_count,
        ] = item["node_mask"]

    return {
        "flow_features": flow_batch,
        "node_features": node_batch,
        "edge_features": edge_batch,
        "edge_mask": edge_mask_batch,
        "node_mask": node_mask_batch,
        "targets": torch.stack([
            item["target"]
            for item in items
        ]),
        "graph_view": graph_view,
        "window_ids": [
            item["window_id"]
            for item in items
        ],
        "capture_ids": [
            item["capture_id"]
            for item in items
        ],
        "source_labels": [
            item["source_label"]
            for item in items
        ],
        "node_ids": [
            item["node_ids"]
            for item in items
        ],
    }
