import torch


def collate_gi_hsp_sequences(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")

    max_node_count = max(
        item["node_features"].shape[1]
        for item in items
    )

    packet_batch = []
    node_batch = []
    adjacency_batch = []
    payload_adjacency_batch = []
    node_masks = []
    targets = []

    for item in items:
        packet_features = item["packet_features"]
        node_features = item["node_features"]
        adjacency = item["adjacency"]
        payload_adjacency = item["payload_adjacency"]

        node_count = node_features.shape[1]

        if adjacency.shape[1:] != (node_count, node_count):
            raise ValueError(
                "Adjacency shape does not match node feature count"
            )

        if payload_adjacency.shape[1:] != (node_count, node_count):
            raise ValueError(
                "Payload adjacency shape does not match node feature count"
            )

        padded_nodes = torch.zeros(
            node_features.shape[0],
            max_node_count,
            node_features.shape[2],
            dtype=node_features.dtype,
            device=node_features.device,
        )
        padded_nodes[:, :node_count, :] = node_features

        padded_adjacency = torch.zeros(
            adjacency.shape[0],
            max_node_count,
            max_node_count,
            dtype=adjacency.dtype,
            device=adjacency.device,
        )
        padded_adjacency[:, :node_count, :node_count] = adjacency

        padded_payload_adjacency = torch.zeros(
            payload_adjacency.shape[0],
            max_node_count,
            max_node_count,
            dtype=payload_adjacency.dtype,
            device=payload_adjacency.device,
        )
        padded_payload_adjacency[
            :, :node_count, :node_count
        ] = payload_adjacency

        node_mask = torch.zeros(
            max_node_count,
            dtype=torch.bool,
            device=node_features.device,
        )
        node_mask[:node_count] = True

        packet_batch.append(packet_features)
        node_batch.append(padded_nodes)
        adjacency_batch.append(padded_adjacency)
        payload_adjacency_batch.append(padded_payload_adjacency)
        node_masks.append(node_mask)
        targets.append(int(item["target"]))

    return {
        "packet_features": torch.stack(packet_batch),
        "node_features": torch.stack(node_batch),
        "adjacency": torch.stack(adjacency_batch),
        "payload_adjacency": torch.stack(payload_adjacency_batch),
        "node_mask": torch.stack(node_masks),
        "targets": torch.tensor(
            targets,
            dtype=torch.long,
        ),
        "labels": [item["label"] for item in items],
        "experiment_ids": [
            item["experiment_id"]
            for item in items
        ],
        "sequence_indices": [
            item["sequence_index"]
            for item in items
        ],
    }
