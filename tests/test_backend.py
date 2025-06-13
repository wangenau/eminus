# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"
"""Test backend class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import eminus
from eminus import backend as xp
from eminus import config


def test_singleton():
    """Check that backend is truly a singleton."""
    assert id(config) == id(eminus.config)


def test_numpy_backend():
    """Test the numpy backend."""
    backend = config.backend
    config.backend = "numpy"
    array = xp.arange(9, dtype=float).reshape((3, 3))

    assert isinstance(xp.pi, float)
    assert xp.sqrt(array).ndim == 2
    assert isinstance(xp.sqrt(array), np.ndarray)
    assert xp.linalg.norm(array, axis=0).ndim == 1
    assert isinstance(xp.linalg.norm(array, axis=0), np.ndarray)
    config.backend = backend  # Restore the default


def test_torch_backend():
    """Test the torch backend."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    import torch

    backend = config.backend
    config.backend = "torch"
    array = xp.arange(9, dtype=float).reshape((3, 3))

    assert isinstance(xp.pi, float)
    assert xp.sqrt(array).ndim == 2
    assert isinstance(xp.sqrt(array), torch.Tensor)
    assert xp.linalg.norm(array, axis=0).ndim == 1
    assert isinstance(xp.linalg.norm(array, axis=0), torch.Tensor)
    config.backend = backend  # Restore the default


def test_switching_and_equality():
    """Test backend switching and compare results."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    backend = config.backend
    config.backend = "torch"
    array = xp.arange(9, dtype=float).reshape((3, 3))
    sqrt_torch = xp.sqrt(array)
    norm_torch = xp.linalg.norm(array, axis=0)

    config.backend = "numpy"
    array = xp.arange(9, dtype=float).reshape((3, 3))
    sqrt_numpy = xp.sqrt(array)
    norm_numpy = xp.linalg.norm(array, axis=0)

    assert type(sqrt_torch) is not type(sqrt_numpy)
    assert_allclose(sqrt_torch, sqrt_numpy)
    assert type(norm_torch) is not type(norm_numpy)
    assert_allclose(norm_torch, norm_numpy)
    config.backend = backend  # Restore the default


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
