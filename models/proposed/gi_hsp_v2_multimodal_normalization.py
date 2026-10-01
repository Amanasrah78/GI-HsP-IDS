from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_normalization import (
    StreamingFeatureNormalizer,
)


class GIHSPV2Normalizer:
    STATE_SCHEMA_VERSION = 1

    def __init__(self, epsilon=1.0e-8):
        self.flow = StreamingFeatureNormalizer(
            FLOW_FEATURE_NAMES,
            binary_feature_names=("step_active",),
            epsilon=epsilon,
        )
        self.node = StreamingFeatureNormalizer(
            NODE_FEATURE_NAMES,
            binary_feature_names=("active",),
            epsilon=epsilon,
        )
        self.edge = StreamingFeatureNormalizer(
            EDGE_FEATURE_NAMES,
            binary_feature_names=(),
            epsilon=epsilon,
        )

        self.flow_presence_index = (
            FLOW_FEATURE_NAMES.index("step_active")
        )
        self.node_presence_index = (
            NODE_FEATURE_NAMES.index("active")
        )
        self.edge_presence_index = (
            EDGE_FEATURE_NAMES.index("flow_count")
        )

    @property
    def fitted(self):
        return (
            self.flow.fitted
            and self.node.fitted
            and self.edge.fitted
        )

    def _presence_masks(self, tensors):
        flow_features = tensors["flow_features"]
        node_features = tensors["node_features"]
        edge_features = tensors["edge_features"]

        flow_active = (
            flow_features[
                ...,
                self.flow_presence_index,
            ]
            > 0
        )
        node_active = (
            node_features[
                ...,
                self.node_presence_index,
            ]
            > 0
        )
        edge_active = tensors.get("edge_mask")

        if edge_active is None:
            edge_active = (
                edge_features[
                    ...,
                    self.edge_presence_index,
                ]
                > 0
            )

        return flow_active, node_active, edge_active

    def update(self, tensors):
        flow_active, node_active, edge_active = (
            self._presence_masks(tensors)
        )

        self.flow.update(
            tensors["flow_features"],
            flow_active,
        )
        self.node.update(
            tensors["node_features"],
            node_active,
        )
        self.edge.update(
            tensors["edge_features"],
            edge_active,
        )

    def finalize(self):
        empty = [
            name
            for name, normalizer in (
                ("flow", self.flow),
                ("node", self.node),
                ("edge", self.edge),
            )
            if normalizer.count == 0
        ]

        if empty:
            raise RuntimeError(
                "Cannot finalize empty modalities: "
                f"{empty}"
            )

        self.flow.finalize()
        self.node.finalize()
        self.edge.finalize()
        return self

    def transform(self, tensors):
        if not self.fitted:
            raise RuntimeError(
                "All modality statistics must be finalized"
            )

        flow_active, node_active, edge_active = (
            self._presence_masks(tensors)
        )
        transformed = dict(tensors)

        transformed["flow_features"] = self.flow.transform(
            tensors["flow_features"],
            flow_active,
        )
        transformed["node_features"] = self.node.transform(
            tensors["node_features"],
            node_active,
        )
        transformed["edge_features"] = self.edge.transform(
            tensors["edge_features"],
            edge_active,
        )

        return transformed

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError(
                "All modality statistics must be finalized"
            )

        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "flow": self.flow.state_dict(),
            "node": self.node.state_dict(),
            "edge": self.edge.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state):
        if (
            state.get("schema_version")
            != cls.STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported normalizer state schema version"
            )

        normalizer = cls()
        normalizer.flow = (
            StreamingFeatureNormalizer.from_state_dict(
                state["flow"]
            )
        )
        normalizer.node = (
            StreamingFeatureNormalizer.from_state_dict(
                state["node"]
            )
        )
        normalizer.edge = (
            StreamingFeatureNormalizer.from_state_dict(
                state["edge"]
            )
        )

        return normalizer
