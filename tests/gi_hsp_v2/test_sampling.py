from collections import Counter

import pytest

from models.proposed.gi_hsp_v2_sampling import (
    RotatingBalancedBatchSampler,
)


TARGETS = [0] * 10 + [1] * 3


def flattened(sampler):
    return [
        index
        for batch in sampler
        for index in batch
    ]


def test_epoch_is_balanced_and_uses_all_minority_samples():
    sampler = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
        seed=7,
    )
    indices = flattened(sampler)
    counts = Counter(
        TARGETS[index]
        for index in indices
    )

    assert counts == {0: 3, 1: 3}
    assert {
        index
        for index in indices
        if TARGETS[index] == 1
    } == {10, 11, 12}


def test_every_batch_has_equal_class_counts():
    sampler = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
        seed=7,
    )

    for batch in sampler:
        labels = Counter(
            TARGETS[index]
            for index in batch
        )
        assert labels[0] == labels[1]


def test_same_seed_and_epoch_are_reproducible():
    first = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
        seed=7,
    )
    second = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
        seed=7,
    )
    first.set_epoch(2)
    second.set_epoch(2)

    assert list(first) == list(second)


def test_majority_samples_rotate_without_early_reuse():
    sampler = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
        seed=7,
    )

    sampler.set_epoch(0)
    first = {
        index
        for index in flattened(sampler)
        if TARGETS[index] == 0
    }

    sampler.set_epoch(1)
    second = {
        index
        for index in flattened(sampler)
        if TARGETS[index] == 0
    }

    assert len(first) == 3
    assert len(second) == 3
    assert first.isdisjoint(second)


def test_length_includes_balanced_final_batch():
    sampler = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
    )

    assert len(sampler) == 2
    assert [len(batch) for batch in sampler] == [4, 2]


@pytest.mark.parametrize("batch_size", [0, 3])
def test_invalid_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError):
        RotatingBalancedBatchSampler(
            TARGETS,
            batch_size=batch_size,
        )


@pytest.mark.parametrize(
    "targets",
    [
        [0, 0, 0],
        [0, 1, 2],
    ],
)
def test_invalid_target_sets_are_rejected(targets):
    with pytest.raises(ValueError):
        RotatingBalancedBatchSampler(
            targets,
            batch_size=2,
        )


def test_negative_epoch_is_rejected():
    sampler = RotatingBalancedBatchSampler(
        TARGETS,
        batch_size=4,
    )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        sampler.set_epoch(-1)
