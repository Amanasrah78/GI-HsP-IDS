import random

import pytest
import torch

from models.proposed.run_gi_hsp_v2 import (
    resolve_seed,
    set_seed,
)


def test_configured_seed_is_used_without_override():
    assert resolve_seed(7) == 7


def test_explicit_seed_overrides_configuration():
    assert resolve_seed(7, override=11) == 11


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_seed(0, override=-1)


def test_set_seed_reproduces_python_and_torch_values():
    set_seed(13)
    first = (random.random(), torch.rand(3))

    set_seed(13)
    second = (random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])
