# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Array backend handling."""

import pathlib
import sys

import numpy as np
import scipy

from . import config


class Backend:
    """Backend class to handle requested namespaces conditionally."""

    def __getattr__(self, name):
        """Access modules and functions of array backends by their name."""
        if config.backend == "torch":
            from array_api_compat import torch as xp
        else:
            xp = np
        return getattr(xp, name)

    @staticmethod
    def is_array(value):
        """Check if the object is an NumPy array or Torch tensor."""
        if isinstance(value, np.ndarray):
            return True
        if config.backend == "torch":
            import torch

            return isinstance(value, torch.Tensor)
        return False

    def expm(self, A, *args, **kwargs):
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
        return self.linalg.matrix_exp(A, *args, **kwargs)

    def sqrtm(self, A):
        """Matrix square root.

        Args:
            A: Matrix whose square root to evaluate.

        Returns:
            Value of the sqrt function at A.
        """
        if isinstance(A, np.ndarray):
            return scipy.linalg.sqrtm(A)
        return self.asarray(scipy.linalg.sqrtm(A), dtype=A.dtype)


# Do not initialize the class when Sphinx or stubtest is running
# Since we set the class instance to the module name Sphinx would only document
# the main docstring of the class without the properties
if (
    "sphinx-build" not in pathlib.Path(sys.argv[0]).name
    and "stubtest" not in pathlib.Path(sys.argv[0]).name
):
    sys.modules[__name__] = Backend()
else:

    def is_array(value):
        """Check if the object is an NumPy array or Torch tensor."""
        return value

    def expm(A, *args, **kwargs):
        """Matrix exponential."""
        return A

    def sqrtm(A):
        """Matrix square root."""
        return A
