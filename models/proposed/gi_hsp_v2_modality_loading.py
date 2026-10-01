import torch

from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from preprocessing.gi_hsp_v2.sequence_loader import (
    load_assembled_window,
)


FLOW_ONLY_ARCHITECTURES = frozenset({
    "flow_only",
    "flow_mlp",
    "flow_gru",
})

STEP_ACTIVE_INDEX = FLOW_FEATURE_NAMES.index(
    "step_active"
)


def flow_only_sequence_to_tensors(
    sequence,
    normalizer=None,
):
    sequence_length = int(sequence["sequence_length"])

    flow_features = torch.tensor(
        sequence["flow_features"],
        dtype=torch.float32,
    )

    if flow_features.ndim != 2:
        raise ValueError(
            "Flow features must have shape [time, features]"
        )

    if tuple(flow_features.shape) != (
        sequence_length,
        len(FLOW_FEATURE_NAMES),
    ):
        raise ValueError(
            "Flow tensor dimensions do not match the "
            "feature contract"
        )

    if not torch.isfinite(flow_features).all():
        raise ValueError(
            "Flow features must be finite"
        )

    flow_active = (
        flow_features[..., STEP_ACTIVE_INDEX] > 0
    )

    if not torch.any(flow_active):
        raise ValueError(
            "A flow-only sequence must contain an active step"
        )

    if normalizer is not None:
        if not normalizer.flow.fitted:
            raise RuntimeError(
                "Flow normalization statistics are not fitted"
            )

        flow_features = normalizer.flow.transform(
            flow_features,
            flow_active,
        )

    # Existing flow-only model signatures accept all modalities,
    # although they use only flow_features. Constant-size placeholders
    # preserve that interface without allocating topology-dependent
    # tensors.
    node_features = torch.zeros(
        (
            sequence_length,
            1,
            len(NODE_FEATURE_NAMES),
        ),
        dtype=torch.float32,
    )
    edge_features = torch.zeros(
        (
            sequence_length,
            1,
            1,
            len(EDGE_FEATURE_NAMES),
        ),
        dtype=torch.float32,
    )
    edge_mask = torch.zeros(
        (
            sequence_length,
            1,
            1,
        ),
        dtype=torch.bool,
    )

    return {
        "flow_features": flow_features,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "node_mask": torch.ones(
            1,
            dtype=torch.bool,
        ),
        "target": torch.tensor(
            int(sequence["binary_label"]),
            dtype=torch.long,
        ),
        "window_id": sequence.get("window_id"),
        "capture_id": sequence["capture_id"],
        "source_label": sequence["source_label"],
        "graph_view": sequence["graph_view"],
        "node_ids": ["__flow_only_placeholder__"],
        "tensor_representation": "flow_only",
    }


class GIHSPV2FlowOnlySequenceDataset(
    GIHSPV2SequenceDataset
):
    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        self._ensure_connections()
        window_id = self.window_ids[index]

        sequence = load_assembled_window(
            self._flow_connection,
            self._index_connection,
            dataset=self.dataset,
            window_id=window_id,
            graph_view=self.graph_view,
            bin_seconds=self.bin_seconds,
        )

        return flow_only_sequence_to_tensors(
            sequence,
            normalizer=self.normalizer,
        )
