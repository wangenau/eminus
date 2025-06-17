# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Array backend handling."""

import numpy as np
import scipy

from . import config


def __getattr__(name):
    """Access modules and functions of array backends by their name."""
    if config.backend == "torch":
        from array_api_compat import torch as xp
    else:
        xp = np
    return getattr(xp, name)


def is_array(value):
    """Check if the object is an NumPy array or Torch tensor."""
    if isinstance(value, np.ndarray):
        return True
    if config.backend == "torch":
        from array_api_compat import is_torch_array

        return is_torch_array(value)
    return False


def expm(A, *args, **kwargs):
    """Matrix exponential.

    Args:
        A: Matrix whose matrix exponential to evaluate.
        args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Value of the exp function at A.
    """
    if isinstance(A, np.ndarray):
        return scipy.linalg.expm(A, *args, **kwargs)
    from array_api_compat import array_namespace

    xp = array_namespace(A)
    return xp.linalg.matrix_exp(A, *args, **kwargs)


def sqrtm(A, *args, **kwargs):
    """Matrix square root.

    Args:
        A: Matrix whose square root to evaluate.
        args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Value of the sqrt function at A.
    """
    if isinstance(A, np.ndarray):
        return scipy.linalg.sqrtm(A, *args, **kwargs)
    from array_api_compat import array_namespace

    xp = array_namespace(A)
    return xp.asarray(scipy.linalg.sqrtm(A, *args, **kwargs), dtype=A.dtype)


def fftn(x, *args, **kwargs):
    """Compute the N-D discrete Fourier Transform.

    Args:
        x: Input array, can be complex.
        args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Value of the fftn function at x.
    """
    if isinstance(x, np.ndarray):
        return scipy.fft.fftn(x, *args, **kwargs)
    from array_api_compat import array_namespace

    xp = array_namespace(x)
    return xp.fft.fftn(x, *args, **kwargs)


def ifftn(x, *args, **kwargs):
    """Compute the N-D inverse discrete Fourier Transform.

    Args:
        x: Input array, can be complex.
        args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Value of the ifftn function at x.
    """
    if isinstance(x, np.ndarray):
        return scipy.fft.ifftn(x, *args, **kwargs)
    from array_api_compat import array_namespace

    xp = array_namespace(x)
    return xp.fft.ifftn(x, *args, **kwargs)
