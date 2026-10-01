import math
from numbers import Real
from statistics import mean, stdev


DEFAULT_BIN_COUNT = 10
CALIBRATION_METRICS = (
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
)


def _binary_target(value, index):
    if isinstance(value, bool):
        return int(value)

    try:
        target = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Target at index {index} must be binary"
        ) from exc

    if target not in (0, 1) or (
        isinstance(value, Real) and float(value) != target
    ):
        raise ValueError(f"Target at index {index} must be binary")

    return target


def _probability(value, index):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"Probability at index {index} must be numeric"
        )

    probability = float(value)

    if not math.isfinite(probability):
        raise ValueError(
            f"Probability at index {index} must be finite"
        )

    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            f"Probability at index {index} must be in [0, 1]"
        )

    return probability


def _bin_count(value):
    if isinstance(value, bool):
        raise ValueError("bin_count must be a positive integer")

    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bin_count must be a positive integer") from exc

    if count <= 0 or (
        isinstance(value, Real) and float(value) != count
    ):
        raise ValueError("bin_count must be a positive integer")

    return count


def binary_calibration_metrics(
    targets,
    probabilities,
    bin_count=DEFAULT_BIN_COUNT,
):
    targets = list(targets)
    probabilities = list(probabilities)
    bin_count = _bin_count(bin_count)

    if not targets:
        raise ValueError("At least one prediction is required")

    if len(targets) != len(probabilities):
        raise ValueError(
            "targets and probabilities must have equal length"
        )

    targets = [
        _binary_target(value, index)
        for index, value in enumerate(targets)
    ]
    probabilities = [
        _probability(value, index)
        for index, value in enumerate(probabilities)
    ]
    sample_count = len(targets)
    bins = [
        {
            "index": index,
            "lower_bound": index / bin_count,
            "upper_bound": (index + 1) / bin_count,
            "count": 0,
            "probability_sum": 0.0,
            "target_sum": 0,
        }
        for index in range(bin_count)
    ]

    for target, probability in zip(targets, probabilities):
        index = min(int(probability * bin_count), bin_count - 1)
        values = bins[index]
        values["count"] += 1
        values["probability_sum"] += probability
        values["target_sum"] += target

    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    reported_bins = []

    for values in bins:
        count = values["count"]

        if count:
            mean_probability = values["probability_sum"] / count
            positive_rate = values["target_sum"] / count
            absolute_gap = abs(mean_probability - positive_rate)
            contribution = count / sample_count * absolute_gap
            expected_calibration_error += contribution
            maximum_calibration_error = max(
                maximum_calibration_error,
                absolute_gap,
            )
        else:
            mean_probability = None
            positive_rate = None
            absolute_gap = None
            contribution = 0.0

        reported_bins.append({
            "index": values["index"],
            "lower_bound": values["lower_bound"],
            "upper_bound": values["upper_bound"],
            "upper_bound_inclusive": values["index"] == bin_count - 1,
            "count": count,
            "mean_probability": mean_probability,
            "positive_rate": positive_rate,
            "absolute_gap": absolute_gap,
            "ece_contribution": contribution,
        })

    brier_score = mean(
        (probability - target) ** 2
        for target, probability in zip(targets, probabilities)
    )

    return {
        "sample_count": sample_count,
        "positive_count": sum(targets),
        "negative_count": sample_count - sum(targets),
        "binning": "equal_width",
        "bin_count": bin_count,
        "brier_score": brier_score,
        "expected_calibration_error": expected_calibration_error,
        "maximum_calibration_error": maximum_calibration_error,
        "reliability_bins": reported_bins,
    }


def aggregate_calibration_runs(
    runs,
    expected_seeds=tuple(range(5)),
    expected_folds=tuple(range(1, 5)),
):
    runs = list(runs)
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    expected_folds = tuple(int(fold) for fold in expected_folds)
    expected_pairs = {
        (seed, fold)
        for seed in expected_seeds
        for fold in expected_folds
    }
    indexed = {}

    for run in runs:
        pair = (int(run["seed"]), int(run["fold"]))

        if pair in indexed:
            raise ValueError(f"Duplicate seed-fold pair: {pair}")

        indexed[pair] = run

        for metric_name in CALIBRATION_METRICS:
            value = run["calibration"][metric_name]

            if not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(
                    f"Calibration metric {metric_name!r} must be finite"
                )

    observed_pairs = set(indexed)

    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        unexpected = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            "Calibration runs do not match the expected design; "
            f"missing={missing}, unexpected={unexpected}"
        )

    metric_output = {}

    for metric_name in CALIBRATION_METRICS:
        per_seed = []
        seed_means = []

        for seed in expected_seeds:
            per_fold = {
                str(fold): float(
                    indexed[(seed, fold)]["calibration"][metric_name]
                )
                for fold in expected_folds
            }
            seed_mean = mean(per_fold.values())
            seed_means.append(seed_mean)
            per_seed.append({
                "seed": seed,
                "mean": seed_mean,
                "per_fold": per_fold,
            })

        metric_output[metric_name] = {
            "mean": mean(seed_means),
            "std": stdev(seed_means) if len(seed_means) > 1 else 0.0,
            "minimum": min(seed_means),
            "maximum": max(seed_means),
            "per_seed": per_seed,
        }

    return {
        "aggregation_unit": "seed_macro_mean_across_folds",
        "run_count": len(runs),
        "seed_count": len(expected_seeds),
        "fold_count": len(expected_folds),
        "seeds": list(expected_seeds),
        "folds": list(expected_folds),
        "metrics": metric_output,
        "runs": sorted(
            runs,
            key=lambda run: (int(run["seed"]), int(run["fold"])),
        ),
    }
