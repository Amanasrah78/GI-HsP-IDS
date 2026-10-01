import math
from numbers import Real
from statistics import mean, stdev


PAIRED_METRICS = (
    "balanced_accuracy",
    "mcc",
)

REQUIRED_CONDITIONS = (
    "fused_identity",
    "fused_role_control",
    "flow_only",
    "topology_only",
)

EXPECTED_SEEDS = tuple(range(5))
EXPECTED_FOLDS = tuple(range(1, 5))
ZERO_TOLERANCE = 1e-12


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")

    return integer


def validate_condition_runs(
    runs,
    expected_seeds=EXPECTED_SEEDS,
    expected_folds=EXPECTED_FOLDS,
):
    runs = list(runs)

    if not runs:
        raise ValueError("At least one run is required")

    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    expected_folds = tuple(int(fold) for fold in expected_folds)
    expected_pairs = {
        (seed, fold)
        for seed in expected_seeds
        for fold in expected_folds
    }
    observed_pairs = []

    for position, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Run {position} must be a mapping")

        missing = {"condition", "seed", "fold", "metrics"} - set(run)

        if missing:
            raise ValueError(
                f"Run {position} is missing fields: {sorted(missing)}"
            )

        if not isinstance(run["metrics"], dict):
            raise ValueError(f"Run {position} metrics must be a mapping")

        pair = (
            _integer(run["seed"], f"run {position} seed"),
            _integer(run["fold"], f"run {position} fold"),
        )
        observed_pairs.append(pair)

        for metric_name, metric_value in run["metrics"].items():
            _finite_float(
                metric_value,
                f"run {position} metric {metric_name!r}",
            )

    duplicates = sorted({
        pair
        for pair in observed_pairs
        if observed_pairs.count(pair) > 1
    })

    if duplicates:
        raise ValueError(f"Duplicate seed-fold pairs: {duplicates}")

    observed_pair_set = set(observed_pairs)
    missing_pairs = sorted(expected_pairs - observed_pair_set)
    unexpected_pairs = sorted(observed_pair_set - expected_pairs)

    if missing_pairs:
        raise ValueError(f"Missing seed-fold pairs: {missing_pairs}")

    if unexpected_pairs:
        raise ValueError(f"Unexpected seed-fold pairs: {unexpected_pairs}")

    conditions = {str(run["condition"]) for run in runs}

    if len(conditions) != 1:
        raise ValueError(
            f"A condition run set must contain one condition: {sorted(conditions)}"
        )

    return runs


def index_runs(
    runs,
    expected_seeds=EXPECTED_SEEDS,
    expected_folds=EXPECTED_FOLDS,
):
    validated = validate_condition_runs(
        runs,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
    )

    return {
        (int(run["seed"]), int(run["fold"])): run
        for run in validated
    }


def _metric(run, metric_name):
    if metric_name not in run["metrics"]:
        raise ValueError(
            f"Run {run.get('source_path', '<unknown>')} is missing "
            f"metric {metric_name!r}"
        )

    return _finite_float(
        run["metrics"][metric_name],
        f"metric {metric_name!r}",
    )


def aggregate_seed_effects(
    fold_effects,
    expected_seeds=EXPECTED_SEEDS,
    expected_folds=EXPECTED_FOLDS,
    zero_tolerance=ZERO_TOLERANCE,
):
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    expected_folds = tuple(int(fold) for fold in expected_folds)
    expected_pairs = {
        (seed, fold)
        for seed in expected_seeds
        for fold in expected_folds
    }
    observed_pairs = set(fold_effects)

    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        unexpected = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            "Fold effects do not match the expected design; "
            f"missing={missing}, unexpected={unexpected}"
        )

    per_seed = []
    seed_means = []

    for seed in expected_seeds:
        per_fold = {
            str(fold): _finite_float(
                fold_effects[(seed, fold)],
                f"effect for seed {seed}, fold {fold}",
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

    positive_count = sum(
        value > zero_tolerance
        for value in seed_means
    )
    negative_count = sum(
        value < -zero_tolerance
        for value in seed_means
    )
    zero_count = len(seed_means) - positive_count - negative_count

    return {
        "mean": mean(seed_means),
        "std": stdev(seed_means) if len(seed_means) > 1 else 0.0,
        "minimum": min(seed_means),
        "maximum": max(seed_means),
        "positive_count": positive_count,
        "zero_count": zero_count,
        "negative_count": negative_count,
        "per_seed": per_seed,
    }


def _validate_condition_mapping(
    condition_runs,
    expected_seeds,
    expected_folds,
):
    if not isinstance(condition_runs, dict):
        raise ValueError("condition_runs must be a mapping")

    condition_names = set(condition_runs)
    required_names = set(REQUIRED_CONDITIONS)

    if condition_names != required_names:
        missing = sorted(required_names - condition_names)
        unexpected = sorted(condition_names - required_names)
        raise ValueError(
            "Condition set does not match the required design; "
            f"missing={missing}, unexpected={unexpected}"
        )

    indexes = {}

    for condition in REQUIRED_CONDITIONS:
        indexes[condition] = index_runs(
            condition_runs[condition],
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
        )

        recorded_names = {
            str(run["condition"])
            for run in condition_runs[condition]
        }

        if recorded_names != {condition}:
            raise ValueError(
                f"Runs stored under {condition!r} identify as "
                f"{sorted(recorded_names)}"
            )

    return indexes


def paired_architecture_effects(
    condition_runs,
    metric_names=PAIRED_METRICS,
    expected_seeds=EXPECTED_SEEDS,
    expected_folds=EXPECTED_FOLDS,
):
    indexes = _validate_condition_mapping(
        condition_runs,
        expected_seeds,
        expected_folds,
    )
    pairs = sorted(indexes["fused_identity"])
    output = {}

    for metric_name in metric_names:
        contrast_effects = {
            "fusion_vs_flow": {},
            "fusion_vs_topology": {},
            "fusion_vs_best_unimodal": {},
            "identity_vs_role": {},
        }

        for pair in pairs:
            identity = _metric(
                indexes["fused_identity"][pair],
                metric_name,
            )
            role = _metric(
                indexes["fused_role_control"][pair],
                metric_name,
            )
            flow = _metric(indexes["flow_only"][pair], metric_name)
            topology = _metric(
                indexes["topology_only"][pair],
                metric_name,
            )

            contrast_effects["fusion_vs_flow"][pair] = identity - flow
            contrast_effects["fusion_vs_topology"][pair] = (
                identity - topology
            )
            contrast_effects["fusion_vs_best_unimodal"][pair] = (
                identity - max(flow, topology)
            )
            contrast_effects["identity_vs_role"][pair] = identity - role

        output[metric_name] = {
            contrast: aggregate_seed_effects(
                effects,
                expected_seeds=expected_seeds,
                expected_folds=expected_folds,
            )
            for contrast, effects in contrast_effects.items()
        }

    return {
        "schema_version": 1,
        "aggregation_unit": "paired_seed_macro_mean_across_folds",
        "seeds": list(expected_seeds),
        "folds": list(expected_folds),
        "metrics": output,
    }


def paired_transfer_effects(
    source_condition_runs,
    target_condition_runs,
    metric_names=PAIRED_METRICS,
    expected_seeds=EXPECTED_SEEDS,
    expected_folds=EXPECTED_FOLDS,
):
    source_indexes = _validate_condition_mapping(
        source_condition_runs,
        expected_seeds,
        expected_folds,
    )
    target_indexes = _validate_condition_mapping(
        target_condition_runs,
        expected_seeds,
        expected_folds,
    )
    output = {}

    for condition in REQUIRED_CONDITIONS:
        source_index = source_indexes[condition]
        target_index = target_indexes[condition]

        if set(source_index) != set(target_index):
            raise ValueError(
                f"Source and target pairs differ for {condition}"
            )

        condition_output = {}

        for metric_name in metric_names:
            fold_effects = {}

            for pair in sorted(source_index):
                source_run = source_index[pair]
                target_run = target_index[pair]

                for field in ("architecture", "graph_view"):
                    if (
                        field in source_run
                        and field in target_run
                        and source_run[field] != target_run[field]
                    ):
                        raise ValueError(
                            f"Source and target {field} differ for "
                            f"{condition}, pair {pair}"
                        )

                fold_effects[pair] = (
                    _metric(target_run, metric_name)
                    - _metric(source_run, metric_name)
                )

            condition_output[metric_name] = aggregate_seed_effects(
                fold_effects,
                expected_seeds=expected_seeds,
                expected_folds=expected_folds,
            )

        output[condition] = condition_output

    return {
        "schema_version": 1,
        "aggregation_unit": "paired_seed_macro_mean_across_folds",
        "effect_direction": "target_minus_source",
        "seeds": list(expected_seeds),
        "folds": list(expected_folds),
        "conditions": output,
    }
