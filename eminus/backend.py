# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Array backend handling.

Includes helper and compatibility functions.
For more information see https://wangenau.gitlab.io/eminus/backend.html.
"""

import pathlib
import sys

import numpy as np
import scipy

from . import config

if "stubtest" not in pathlib.Path(sys.argv[0]).name:
    # Do not overwrite getattr when stubtest is running
    def __getattr__(name):
        """Access modules and functions of array backends by their name."""
        if config.backend == "torch":
            from array_api_compat import torch as xp
        else:
            xp = np
        return getattr(xp, name)


# ### Helper functions ###


def is_array(value):
    """Check if the object is an NumPy array or Torch tensor.

    Args:
        value: Input array.

    Returns:
        If the value is an array supported by the available backends.
    """
    if isinstance(value, np.ndarray):
        return True
    if config.backend == "torch":
        from array_api_compat import is_torch_array

        return is_torch_array(value)
    return False


def to_np(*args):
    """Copy the arrays from the current device to CPU NumPy arrays.

    Args:
        args: Input arrays.

    Returns:
        Copied arrays on the CPU.
    """
    if config.backend == "torch":
        cpu_arr = tuple(np.asarray(i.cpu()) if not isinstance(i, np.ndarray) else i for i in args)
    else:
        cpu_arr = args
    # Single args should return single args, not tuple
    if len(cpu_arr) == 1:
        return cpu_arr[0]
    return cpu_arr


# ### Compatibility functions ###


def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted.

    Ref: https://gist.github.com/velikodniy/6efef837e67aee2e7152eb5900eb0258

    Args:
        arr: Input array.
        obj: Indicate indices of sub-arrays to remove along the specified axis.

    Keyword Args:
        axis: The axis along which to delete the subarray defined by obj. If `axis` is `None`, `obj`
        is applied to the flattened array.

    Returns:
        A copy of `arr` with the elements specified by `obj` removed. If `axis` is `None`, `out` is
        a flattened array.
    """
    if isinstance(arr, np.ndarray):
        return np.delete(arr, obj, axis)
    if axis is None:
        axis = 0
        arr = arr.ravel()
    skip = [i for i in range(arr.size(axis)) if i not in np.asarray(obj)]
    indices = [slice(None) if i != axis else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


def fftn(x, *args, **kwargs):
    """Compute the N-D discrete Fourier Transform.

    Use SciPy FFTs since they are faster, support parallelism, and are more accurate.
    They will upcast some complex arrays to complex256. Using NumPy FFTs will fail some tests.

    Args:
        x: Input array, can be complex.
        args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Value of the fftn function at x.
    """
    if isinstance(x, np.ndarray):
        return scipy.fft.fftn(x, *args, **kwargs, workers=config.threads)
    from array_api_compat import array_namespace

    xp = array_namespace(x)
    return xp.fft.fftn(x, *args, **kwargs)


def ifftn(x, *args, **kwargs):
    """Compute the N-D inverse discrete Fourier Transform.

    Use SciPy FFTs since they are faster, support parallelism, and are more accurate.
    They will upcast some complex arrays to complex256. Using NumPy FFTs will fail some tests.

    Args:
        x: Input array, can be complex.
        args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Value of the ifftn function at x.
    """
    if isinstance(x, np.ndarray):
        return scipy.fft.ifftn(x, *args, **kwargs, workers=config.threads)
    from array_api_compat import array_namespace

    xp = array_namespace(x)
    return xp.fft.ifftn(x, *args, **kwargs)


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
        return np.asarray(scipy.linalg.sqrtm(A, *args, **kwargs), dtype=complex)
    from array_api_compat import array_namespace

    xp = array_namespace(A)
    return xp.asarray(
        np.asarray(scipy.linalg.sqrtm(to_np(A), *args, **kwargs), dtype=complex), dtype=complex
    )
