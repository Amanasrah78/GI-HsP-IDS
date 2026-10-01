import math
from collections import defaultdict


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _validate_inputs(targets, probabilities, scenarios, threshold):
    targets = [int(value) for value in targets]
    probabilities = [float(value) for value in probabilities]

    if not targets:
        raise ValueError("At least one prediction is required")

    if len(targets) != len(probabilities):
        raise ValueError("targets and probabilities must have equal length")

    if any(target not in (0, 1) for target in targets):
        raise ValueError("targets must contain only binary labels 0 and 1")

    if any(
        not math.isfinite(probability)
        or probability < 0.0
        or probability > 1.0
        for probability in probabilities
    ):
        raise ValueError(
            "probabilities must be finite values between 0 and 1"
        )

    threshold = float(threshold)

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if scenarios is not None:
        scenarios = [str(value) for value in scenarios]

        if len(scenarios) != len(targets):
            raise ValueError(
                "scenarios and targets must have equal length"
            )

        if any(not scenario.strip() for scenario in scenarios):
            raise ValueError("scenario names must not be empty")

    return targets, probabilities, scenarios, threshold


def _auroc(targets, probabilities):
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count

    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(
        zip(probabilities, targets),
        key=lambda item: item[0],
    )

    positive_rank_sum = 0.0
    index = 0

    while index < len(ranked):
        end = index + 1

        while (
            end < len(ranked)
            and ranked[end][0] == ranked[index][0]
        ):
            end += 1

        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            target for _, target in ranked[index:end]
        )
        index = end

    return (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _average_precision(targets, probabilities):
    positive_count = sum(targets)

    if positive_count == 0:
        return None

    ranked = sorted(
        zip(probabilities, targets),
        key=lambda item: item[0],
        reverse=True,
    )

    true_positive = 0
    processed = 0
    precision_sum = 0.0
    index = 0

    while index < len(ranked):
        end = index + 1

        while (
            end < len(ranked)
            and ranked[end][0] == ranked[index][0]
        ):
            end += 1

        group_positive = sum(
            target for _, target in ranked[index:end]
        )
        true_positive += group_positive
        processed += end - index

        if group_positive:
            precision_sum += (
                group_positive * true_positive / processed
            )

        index = end

    return precision_sum / positive_count


def _scenario_recall(targets, predictions, scenarios):
    if scenarios is None:
        return {}

    counts = defaultdict(lambda: {"positive_count": 0, "true_positive": 0})

    for target, prediction, scenario in zip(
        targets,
        predictions,
        scenarios,
    ):
        if target != 1:
            continue

        counts[scenario]["positive_count"] += 1
        counts[scenario]["true_positive"] += int(prediction == 1)

    return {
        scenario: {
            "positive_count": values["positive_count"],
            "true_positive": values["true_positive"],
            "recall": _safe_divide(
                values["true_positive"],
                values["positive_count"],
            ),
        }
        for scenario, values in sorted(counts.items())
    }


def binary_classification_metrics(
    targets,
    probabilities,
    scenarios=None,
    threshold=0.5,
):
    targets, probabilities, scenarios, threshold = _validate_inputs(
        targets,
        probabilities,
        scenarios,
        threshold,
    )

    predictions = [
        int(probability >= threshold)
        for probability in probabilities
    ]

    true_negative = sum(
        target == 0 and prediction == 0
        for target, prediction in zip(targets, predictions)
    )
    false_positive = sum(
        target == 0 and prediction == 1
        for target, prediction in zip(targets, predictions)
    )
    false_negative = sum(
        target == 1 and prediction == 0
        for target, prediction in zip(targets, predictions)
    )
    true_positive = sum(
        target == 1 and prediction == 1
        for target, prediction in zip(targets, predictions)
    )

    precision = _safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = _safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    specificity = _safe_divide(
        true_negative,
        true_negative + false_positive,
    )
    f1 = _safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = _safe_divide(
        true_positive * true_negative
        - false_positive * false_negative,
        mcc_denominator,
    )

    return {
        "sample_count": len(targets),
        "threshold": threshold,
        "loss": None,
        "accuracy": _safe_divide(
            true_positive + true_negative,
            len(targets),
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": mcc,
        "auroc": _auroc(targets, probabilities),
        "auprc": _average_precision(targets, probabilities),
        "auprc_method": "average_precision",
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "per_attack_scenario_recall": _scenario_recall(
            targets,
            predictions,
            scenarios,
        ),
    }
