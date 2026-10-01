import itertools
import math
from numbers import Real
from statistics import mean, stdev

from scipy.stats import t


ZERO_TOLERANCE = 1e-12


def finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_seed_effects(seed_effects, expected_seeds=tuple(range(5))):
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    if len(expected_seeds) < 2:
        raise ValueError("At least two seeds are required")
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Expected seeds contain duplicates")
    if not isinstance(seed_effects, list):
        raise ValueError("per_seed must be a list")

    values_by_seed = {}
    for position, item in enumerate(seed_effects):
        if not isinstance(item, dict):
            raise ValueError(f"Seed effect {position} must be a mapping")
        missing = {"seed", "mean"} - set(item)
        if missing:
            raise ValueError(
                f"Seed effect {position} is missing: {sorted(missing)}"
            )
        seed = int(item["seed"])
        if seed in values_by_seed:
            raise ValueError(f"Duplicate seed effect: {seed}")
        values_by_seed[seed] = finite_float(
            item["mean"], f"effect for seed {seed}"
        )

    observed = tuple(sorted(values_by_seed))
    if observed != tuple(sorted(expected_seeds)):
        raise ValueError(
            "Seed effects do not match the expected design; "
            f"expected={sorted(expected_seeds)}, observed={list(observed)}"
        )
    return [values_by_seed[seed] for seed in expected_seeds]


def exact_sign_flip_test(values, zero_tolerance=ZERO_TOLERANCE):
    values = [finite_float(value, "paired effect") for value in values]
    if not values:
        raise ValueError("At least one paired effect is required")

    observed = abs(mean(values))
    extreme = 0
    permutation_count = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(mean(
            sign * value for sign, value in zip(signs, values)
        ))
        if permuted >= observed - zero_tolerance:
            extreme += 1

    return {
        "alternative": "two_sided",
        "statistic": observed,
        "p_value": extreme / permutation_count,
        "extreme_assignment_count": extreme,
        "sign_assignment_count": permutation_count,
        "assumption": "exchangeable_signs_under_symmetric_null",
    }


def t_confidence_interval(values, confidence_level=0.95):
    values = [finite_float(value, "paired effect") for value in values]
    if len(values) < 2:
        raise ValueError("At least two paired effects are required")
    confidence_level = finite_float(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    center = mean(values)
    sample_sd = stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    critical = float(t.ppf(
        0.5 + confidence_level / 2.0,
        df=len(values) - 1,
    ))
    margin = critical * standard_error
    return {
        "method": "student_t_interval_across_seed_effects",
        "confidence_level": confidence_level,
        "degrees_of_freedom": len(values) - 1,
        "critical_value": critical,
        "standard_error": standard_error,
        "lower": center - margin,
        "upper": center + margin,
    }


def standardized_paired_effect(values, zero_tolerance=ZERO_TOLERANCE):
    values = [finite_float(value, "paired effect") for value in values]
    if len(values) < 2:
        raise ValueError("At least two paired effects are required")
    center = mean(values)
    sample_sd = stdev(values)
    if sample_sd <= zero_tolerance:
        return {
            "cohens_dz": None,
            "hedges_gz": None,
            "status": (
                "all_effects_zero"
                if abs(center) <= zero_tolerance
                else "zero_variance_nonzero_effect"
            ),
        }
    dz = center / sample_sd
    degrees_of_freedom = len(values) - 1
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
    return {
        "cohens_dz": dz,
        "hedges_gz": correction * dz,
        "small_sample_correction": correction,
        "status": "defined",
    }


def holm_adjust(p_values):
    if not p_values:
        raise ValueError("At least one p-value is required")
    checked = {
        name: finite_float(value, f"p-value {name!r}")
        for name, value in p_values.items()
    }
    if any(not 0.0 <= value <= 1.0 for value in checked.values()):
        raise ValueError("p-values must lie in [0, 1]")

    ordered = sorted(checked, key=lambda name: (checked[name], name))
    count = len(ordered)
    adjusted = {}
    running_maximum = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * checked[name])
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted


def infer_contrast(values, confidence_level=0.95):
    values = [finite_float(value, "paired effect") for value in values]
    if len(values) < 2:
        raise ValueError("At least two seed effects are required")
    center = mean(values)
    sample_sd = stdev(values)
    return {
        "independent_unit": "training_seed",
        "seed_count": len(values),
        "seed_effects": values,
        "mean_effect": center,
        "sample_standard_deviation": sample_sd,
        "minimum_effect": min(values),
        "maximum_effect": max(values),
        "positive_count": sum(value > ZERO_TOLERANCE for value in values),
        "zero_count": sum(abs(value) <= ZERO_TOLERANCE for value in values),
        "negative_count": sum(value < -ZERO_TOLERANCE for value in values),
        "confidence_interval": t_confidence_interval(
            values, confidence_level=confidence_level
        ),
        "standardized_effect": standardized_paired_effect(values),
        "exact_sign_flip_test": exact_sign_flip_test(values),
    }
