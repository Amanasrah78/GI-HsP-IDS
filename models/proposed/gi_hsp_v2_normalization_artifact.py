import json
from pathlib import Path

from models.proposed.gi_hsp_v2_feature_contract import (
    validate_graph_view,
)
from models.proposed.gi_hsp_v2_multimodal_normalization import (
    GIHSPV2Normalizer,
)


ARTIFACT_SCHEMA_VERSION = 1


def load_normalization_artifact(
    path,
    expected_fold=None,
    expected_graph_view=None,
):
    path = Path(path)
    artifact = json.loads(path.read_text())

    if (
        artifact.get("schema_version")
        != ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported normalization artifact schema version"
        )

    required = {
        "dataset",
        "fold",
        "graph_view",
        "fit_partition",
        "fit_unit",
        "bin_seconds",
        "training_capture_ids",
        "input_sha256",
        "normalizer",
    }
    missing = required - set(artifact)

    if missing:
        raise ValueError(
            "Normalization artifact is missing fields: "
            f"{sorted(missing)}"
        )

    if artifact["fit_partition"] != "train":
        raise ValueError(
            "Normalization artifact was not fitted on training data"
        )

    if artifact["fit_unit"] != "unique_active_step":
        raise ValueError(
            "Unsupported normalization fitting unit"
        )

    try:
        bin_seconds = int(artifact["bin_seconds"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Normalization artifact bin_seconds must be an integer"
        ) from exc

    if bin_seconds <= 0:
        raise ValueError(
            "Normalization artifact bin_seconds must be positive"
        )

    graph_view = validate_graph_view(
        artifact["graph_view"]
    )
    fold = int(artifact["fold"])

    if expected_fold is not None:
        if fold != int(expected_fold):
            raise ValueError(
                "Normalization artifact fold does not match"
            )

    if expected_graph_view is not None:
        expected = validate_graph_view(
            expected_graph_view
        )

        if graph_view != expected:
            raise ValueError(
                "Normalization artifact graph view does not match"
            )

    normalizer = GIHSPV2Normalizer.from_state_dict(
        artifact["normalizer"]
    )

    if not normalizer.fitted:
        raise ValueError(
            "Normalization artifact is not finalized"
        )

    return normalizer, artifact
