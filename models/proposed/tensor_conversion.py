import torch


def sequence_inputs_to_tensors(
    assembled,
    device=None,
):
    packet_features = torch.tensor(
        [assembled["packet_features"]],
        dtype=torch.float32,
        device=device,
    )

    node_features = torch.tensor(
        [assembled["node_features"]],
        dtype=torch.float32,
        device=device,
    )

    adjacency = torch.tensor(
        [assembled["adjacency"]],
        dtype=torch.float32,
        device=device,
    )

    return {
        "packet_features": packet_features,
        "node_features": node_features,
        "adjacency": adjacency,
        "label": assembled["label"],
    }
