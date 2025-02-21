# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Extra functions that need additional dependencies to work.

To also install all additional dependencies, use::

    pip install eminus[all]

Alternatively, you can only install selected extras using the respective name:

* :mod:`~eminus.extras.dispersion`
* :mod:`~eminus.extras.fods`
* :mod:`~eminus.extras.hdf5`
* :mod:`~eminus.extras.jax`
* :mod:`~eminus.extras.libxc`
* :mod:`~eminus.extras.symmetry`
* :mod:`~eminus.extras.torch`
* :mod:`~eminus.extras.viewer`

Note that the :mod:`~eminus.extras.libxc` extra will install PySCF by default. pylibxc is also
supported and will be preferred if it is found in the environment. See :mod:`~eminus.extras.libxc`
for more information.

Additionally, :mod:`~eminus.extras.torch` has different installation flavors. See
:mod:`~eminus.extras.torch` for more information.
"""

from . import jax, torch
from .dispersion import get_Edisp
from .fods import get_fods, remove_core_fods, split_fods
from .hdf5 import read_hdf5, write_hdf5
from .libxc import libxc_functional
from .symmetry import symmetrize
from .viewer import (
    executed_in_notebook,
    plot_bandstructure,
    plot_dos,
    view,
    view_atoms,
    view_contour,
    view_file,
    view_kpts,
)

__all__ = [
    "executed_in_notebook",
    "get_Edisp",
    "get_fods",
    "jax",
    "libxc_functional",
    "plot_bandstructure",
    "plot_dos",
    "read_hdf5",
    "remove_core_fods",
    "split_fods",
    "symmetrize",
    "torch",
    "view",
    "view_atoms",
    "view_contour",
    "view_file",
    "view_kpts",
    "write_hdf5",
]
