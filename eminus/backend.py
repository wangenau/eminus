# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Array backend handling."""

import functools
import pathlib
import sys

import numpy as np
import scipy


from . import config


def sqrtm(A, *args, **kwargs):
    """sqrtm with Cupy support, convert to complex additionally."""
    from array_api_compat.common import array_namespace
    xp = array_namespace(A)
    return xp.asarray(scipy.linalg.sqrtm(A, *args, **kwargs), dtype=xp.complex128)


class Backend:
    """Backend class to handle requested namespaces conditionally."""

    def __getattr__(self, name):
        """Access modules and functions of array backends by their name."""
        WRAPPERS = {
            "sqrtm": sqrtm,
        }
        WRAPPER_KEYS = set(WRAPPERS)

        if name in WRAPPER_KEYS:
            return WRAPPERS[name]

        if config.backend == "torch":
            from array_api_compat import torch as xp
        else:
            xp = np
        try:
            return getattr(xp, name)
        except AttributeError:
            return getattr(np, name)

    @staticmethod
    def debug(func):  # noqa: C901
        """Decorator to convert input arrays to Torch tensors and return NumPy arrays in the end."""
        DEBUG = True

        @functools.wraps(func)
        def decorator(*args, **kwargs):  # noqa: C901
            torch_input = True
            if DEBUG and config.backend == "torch":
                import torch

                # Convert input args to Torch tensors if necessary
                args = list(args)
                for i in range(len(args)):
                    if isinstance(args[i], np.ndarray):
                        args[i] = torch.asarray(args[i])
                        torch_input = False
                    if hasattr(args[i], "convert"):
                        args[i].convert()
                        torch_input = False
                # Convert input kwargs to Torch tensors if necessary
                for key, value in kwargs.items():
                    if isinstance(value, np.ndarray):
                        kwargs[key] = torch.asarray(value)
                        torch_input = False
                    if hasattr(value, "convert"):
                        value.convert()
                        torch_input = False
            ret = func(*args, **kwargs)
            if DEBUG and config.backend == "torch" and not torch_input:
                # Convert return value to NumPy array if necessary
                if isinstance(ret, tuple):
                    ret = list(ret)
                    for i in range(len(ret)):
                        if isinstance(ret[i], torch.Tensor):
                            ret[i] = np.asarray(ret[i])
                elif isinstance(ret, torch.Tensor):
                    ret = np.asarray(ret)
                # Convert objects back to NumPy
                args = list(args)
                for i in range(len(args)):
                    if hasattr(args[i], "convert"):
                        args[i].convert("np")
                for value in kwargs.values():
                    if hasattr(value, "convert"):
                        value.convert("np")
            return ret

        return decorator

    @staticmethod
    def convert(value):
        """Convert input to the desired backend array; wrapper to keep track of changes."""
        if config.backend == "torch":
            from array_api_compat import torch as xp
        else:
            xp = np
        if isinstance(value, (list, tuple)):
            return [xp.asarray(i) if isinstance(i, np.ndarray) else i for i in value]
        return xp.asarray(value)

    @staticmethod
    def is_array(value):
        """Check if the object is an NumPy array or Torch tensor."""
        if isinstance(value, np.ndarray):
            return True
        if config.backend == "torch":
            import torch

            return isinstance(value, torch.Tensor)
        return False


# Do not initialize the class when Sphinx or stubtest is running
# Since we set the class instance to the module name Sphinx would only document
# the main docstring of the class without the properties
if (
    "sphinx-build" not in pathlib.Path(sys.argv[0]).name
    and "stubtest" not in pathlib.Path(sys.argv[0]).name
):
    sys.modules[__name__] = Backend()
else:

    def debug(func):
        """Decorator to convert input arrays to Torch tensors and return NumPy arrays in the end."""
        return func

    def convert(value):
        """Convert input to the desired backend array; wrapper to keep track of changes."""
        return value

    def is_array(value):
        """Check if the object is an NumPy array or Torch tensor."""
        return value
