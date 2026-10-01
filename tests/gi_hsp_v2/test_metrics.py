import pytest

from models.proposed.gi_hsp_v2_metrics import (
    binary_classification_metrics,
)


def test_known_binary_example():
    metrics = binary_classification_metrics(
        targets=[0, 0, 1, 1],
        probabilities=[0.1, 0.4, 0.35, 0.8],
    )

    assert metrics["sample_count"] == 4
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["mcc"] == pytest.approx(1.0 / 3.0 ** 0.5)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5.0 / 6.0)
    assert metrics["auprc_method"] == "average_precision"


def test_perfect_predictions_have_perfect_metrics():
    metrics = binary_classification_metrics(
        targets=[0, 0, 1, 1],
        probabilities=[0.0, 0.2, 0.8, 1.0],
    )

    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "auroc",
        "auprc",
    ):
        assert metrics[name] == pytest.approx(1.0)


def test_tied_scores_are_handled_deterministically():
    metrics = binary_classification_metrics(
        targets=[0, 1],
        probabilities=[0.5, 0.5],
    )

    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["auprc"] == pytest.approx(0.5)


def test_single_class_partition_reports_undefined_ranking_metrics():
    metrics = binary_classification_metrics(
        targets=[0, 0],
        probabilities=[0.1, 0.2],
    )

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None


def test_attack_recall_is_reported_per_scenario():
    metrics = binary_classification_metrics(
        targets=[0, 1, 1, 1],
        probabilities=[0.1, 0.8, 0.4, 0.7],
        scenarios=["benign", "flood", "flood", "malformed"],
    )

    assert metrics["per_attack_scenario_recall"] == {
        "flood": {
            "positive_count": 2,
            "true_positive": 1,
            "recall": 0.5,
        },
        "malformed": {
            "positive_count": 1,
            "true_positive": 1,
            "recall": 1.0,
        },
    }


def test_probability_equal_to_threshold_is_predicted_as_attack():
    metrics = binary_classification_metrics(
        targets=[1],
        probabilities=[0.5],
        threshold=0.5,
    )

    assert metrics["confusion_matrix"] == [[0, 0], [0, 1]]


@pytest.mark.parametrize(
    ("targets", "probabilities", "message"),
    [
        ([], [], "At least one prediction"),
        ([0], [0.1, 0.2], "equal length"),
        ([0, 2], [0.1, 0.9], "binary labels"),
        ([0, 1], [-0.1, 0.9], "between 0 and 1"),
        ([0, 1], [0.1, float("nan")], "finite values"),
    ],
)
def test_invalid_prediction_inputs_are_rejected(
    targets,
    probabilities,
    message,
):
    with pytest.raises(ValueError, match=message):
        binary_classification_metrics(
            targets=targets,
            probabilities=probabilities,
        )


def test_scenario_count_must_match_prediction_count():
    with pytest.raises(ValueError, match="scenarios and targets"):
        binary_classification_metrics(
            targets=[0, 1],
            probabilities=[0.1, 0.9],
            scenarios=["benign"],
        )
