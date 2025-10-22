# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"
"""Test backend module."""

import numpy as np
import pytest

import eminus
from eminus import backend as xp
from eminus import config
from eminus.testing import assert_allclose, assert_array_equal


def test_singleton():
    """Check that backend is truly a singleton."""
    assert id(config) == id(eminus.config)


def test_numpy_backend():
    """Test the numpy backend."""
    backend = config.backend
    config.backend = "numpy"
    array = xp.arange(9).reshape(3, 3)

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
    array = xp.arange(9, dtype=float).reshape(3, 3)

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
    array = xp.arange(9, dtype=float).reshape(3, 3)
    sqrt_torch = xp.sqrt(array)
    norm_torch = xp.linalg.norm(array, axis=0)

    config.backend = "numpy"
    array = xp.arange(9).reshape(3, 3)
    sqrt_numpy = xp.sqrt(array)
    norm_numpy = xp.linalg.norm(array, axis=0)

    assert type(sqrt_torch) is not type(sqrt_numpy)
    assert_allclose(sqrt_torch, sqrt_numpy)
    assert type(norm_torch) is not type(norm_numpy)
    assert_allclose(norm_torch, norm_numpy)
    config.backend = backend  # Restore the default


def test_is_array():
    """Test the array check helper function."""
    pytest.importorskip("torch", reason="torch not installed, skip tests")
    backend = config.backend
    config.backend = "torch"
    tensor = xp.arange(9, dtype=float)
    assert xp.is_array(tensor)
    array = np.arange(9, dtype=complex)
    assert xp.is_array(array)

    config.backend = "numpy"
    assert not xp.is_array(tensor)  # We skip the Torch import for the NumPy backend
    assert xp.is_array(array)
    config.backend = backend  # Restore the default


def test_to_np():
    """Test the casting to CPU arrays."""
    array1 = xp.arange(9)
    array1 = xp.to_np(array1)
    assert str(array1.device) == "cpu"
    assert isinstance(array1, np.ndarray)
    array2 = xp.arange(9)
    array1, array2 = xp.to_np(array1), xp.to_np(array2)
    assert str(array1.device) == "cpu"
    assert str(array2.device) == "cpu"
    assert isinstance(array1, np.ndarray)
    assert isinstance(array2, np.ndarray)


def test_delete():
    """Test the delete implementation."""
    array = xp.arange(4)
    assert_array_equal(xp.delete(array, 1), [0, 2, 3])
    assert_array_equal(xp.delete(array, [1, 3]), [0, 2])
    assert_array_equal(xp.delete(array, [0, 1, 2, 3]), [])

    array = xp.arange(4).reshape(2, 2)  # axis=None returns a flattened array
    assert_array_equal(xp.delete(array, 1), [0, 2, 3])
    assert_array_equal(xp.delete(array, [1, 3], axis=None), [0, 2])
    assert_array_equal(xp.delete(array, [0, 1, 2, 3]), [])

    array = xp.arange(4).reshape(2, 2)
    assert_array_equal(xp.delete(array, 0, axis=0), [[2, 3]])
    assert_array_equal(xp.delete(array, 1, axis=1), [[0], [2]])

    array = xp.arange(9).reshape(3, 3)
    assert_array_equal(xp.delete(array, [0, 2], axis=0), [[3, 4, 5]])
    assert_array_equal(xp.delete(array, [1, 2], axis=1), [[0], [3], [6]])


@pytest.mark.parametrize("name", ["fftn", "ifftn", "sqrtm"])
def test_trivial_functions(name):
    """Check the availability of the more trivial backend functions implementations."""
    array = xp.arange(9).reshape(3, 3)
    func = getattr(xp, name)
    func(array)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
