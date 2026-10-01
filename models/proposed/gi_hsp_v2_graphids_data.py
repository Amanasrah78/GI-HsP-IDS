import torch
from torch.utils.data import Dataset

from models.proposed.gi_hsp_v2_graphids import (
    _group_edge_embeddings,
)


class BenignWindowSubset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [
            index
            for index, target in enumerate(dataset.targets)
            if int(target) == 0
        ]

        if not self.indices:
            raise ValueError(
                "Training partition contains no benign windows"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def collate_graphids_windows(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")

    edge_features = []
    edge_indexes = []
    edge_windows = []
    targets = []
    window_ids = []
    capture_ids = []
    source_labels = []
    node_offset = 0

    for window_index, item in enumerate(items):
        if item.get("tensor_representation") != "sparse":
            raise ValueError(
                "GraphIDS requires sparse window tensors"
            )

        current_features = item["edge_features"]
        current_index = item["edge_index"]
        node_count = int(item["node_mask"].shape[0])

        if current_features.shape[0] == 0:
            raise ValueError(
                "GraphIDS window contains no occupied edges"
            )

        edge_features.append(current_features)
        edge_indexes.append(current_index + node_offset)
        edge_windows.append(
            torch.full(
                (current_features.shape[0],),
                window_index,
                dtype=torch.long,
            )
        )
        targets.append(item["target"])
        window_ids.append(item["window_id"])
        capture_ids.append(item["capture_id"])
        source_labels.append(item["source_label"])
        node_offset += node_count

    return {
        "edge_features": torch.cat(edge_features, dim=0),
        "edge_index": torch.cat(edge_indexes, dim=1),
        "edge_windows": torch.cat(edge_windows),
        "node_count": node_offset,
        "targets": torch.stack(targets),
        "window_ids": window_ids,
        "capture_ids": capture_ids,
        "source_labels": source_labels,
    }


def group_window_embeddings(
    embeddings,
    edge_windows,
    window_count,
    group_size,
    generator=None,
):
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape [items, features]"
        )
    if edge_windows.ndim != 1:
        raise ValueError(
            "edge_windows must have shape [items]"
        )
    if edge_windows.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "edge_windows must match embeddings"
        )

    groups = []
    masks = []
    mappings = []

    for window_index in range(window_count):
        selected = edge_windows == window_index
        if not torch.any(selected):
            continue

        window_embeddings = embeddings[selected]
        grouped, mask, _ = _group_edge_embeddings(
            window_embeddings,
            group_size,
            generator=generator,
            fixed_padding=False,
        )

        mapping = torch.full(
            mask.shape[:2],
            -1,
            dtype=torch.long,
            device=embeddings.device,
        )
        mapping[mask.any(dim=-1)] = window_index

        groups.append(grouped)
        masks.append(mask)
        mappings.append(mapping)

    if not groups:
        raise ValueError("No window contains edges")

    max_length = max(
        group.shape[1]
        for group in groups
    )

    padded_groups = []
    padded_masks = []
    padded_mappings = []

    for group, mask, mapping in zip(
        groups,
        masks,
        mappings,
    ):
        padding = max_length - group.shape[1]

        if padding:
            group = torch.cat(
                [
                    group,
                    group.new_zeros(
                        group.shape[0],
                        padding,
                        group.shape[2],
                    ),
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        mask.shape[0],
                        padding,
                        mask.shape[2],
                        dtype=torch.bool,
                        device=mask.device,
                    ),
                ],
                dim=1,
            )
            mapping = torch.cat(
                [
                    mapping,
                    torch.full(
                        (mapping.shape[0], padding),
                        -1,
                        dtype=torch.long,
                        device=mapping.device,
                    ),
                ],
                dim=1,
            )

        padded_groups.append(group)
        padded_masks.append(mask)
        padded_mappings.append(mapping)

    return (
        torch.cat(padded_groups, dim=0),
        torch.cat(padded_masks, dim=0),
        torch.cat(padded_mappings, dim=0),
    )

def maximum_window_scores(
    edge_errors,
    error_windows,
    window_count,
):
    if edge_errors.ndim != 1:
        raise ValueError("edge_errors must be one-dimensional")
    if error_windows.shape != edge_errors.shape:
        raise ValueError(
            "error_windows must align with edge_errors"
        )

    scores = []

    for window_index in range(window_count):
        selected = edge_errors[
            error_windows == window_index
        ]

        if selected.numel() == 0:
            raise ValueError(
                "Window contains no reconstruction errors"
            )

        scores.append(selected.max())

    return torch.stack(scores)
