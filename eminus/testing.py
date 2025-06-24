# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Array backend testing."""

import numpy as np

from . import backend as xp


def assert_allclose(actual, desired, *args, **kwargs):
    """Raises an AssertionError if two objects are not equal up to desired tolerance.

    Args:
        actual: Array obtained.
        desired: Array desired.
        *args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.
    """
    np.testing.assert_allclose(xp.to_np(actual), xp.to_np(desired), *args, **kwargs)


def assert_array_equal(actual, desired, *args, **kwargs):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
        actual: The actual object to check.
        desired: The desired, expected object.
        *args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.
    """
    np.testing.assert_array_equal(xp.to_np(actual), xp.to_np(desired), *args, **kwargs)
