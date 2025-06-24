# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"
"""Test testing module."""

import numpy as np

from eminus import backend as xp
from eminus.testing import assert_allclose, assert_array_equal


def test_assert_allclose():
    """Test the assert_allclose function."""
    x = xp.arange(4)
    n = np.arange(4)
    l = [0.1, 1.1, 2.1, 3.1]
    assert_allclose(x, x)
    assert_allclose(x, n, rtol=0)
    assert_allclose(x, l, atol=0.1)


def test_assert_array_equal():
    """Test the assert_array_equal function."""
    x = xp.arange(4)
    n = np.arange(4)
    l = [0, 1, 2, 3]
    assert_array_equal(x, x)
    assert_array_equal(x, n)
    assert_array_equal(x, l)
