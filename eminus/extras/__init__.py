# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Extra functions that need additional dependencies to work.

To also install all additional dependencies, use::

    pip install eminus[all]

Alternatively, you can only install selected extras using the respective name:

* :mod:`~eminus.extras.d3`
* :mod:`~eminus.extras.fods`
* :mod:`~eminus.extras.gui`
* :mod:`~eminus.extras.hdf5`
* :mod:`~eminus.extras.libxc`

Note that the :mod:`~eminus.extras.libxc` extra will install PySCF by default. pylibxc is also
supported and will be preferred if it is found in the environment. See :mod:`~eminus.extras.libxc`
for more information.
"""

from .d3 import get_Edisp
from .fods import get_fods, remove_core_fods, split_fods
from .gui import (
    executed_in_notebook,
    plot_bandstructure,
    plot_dos,
    view,
    view_atoms,
    view_contour,
    view_file,
    view_kpts,
)
from .hdf5 import read_hdf5, write_hdf5
from .libxc import libxc_functional

__all__ = [
    "executed_in_notebook",
    "get_Edisp",
    "get_fods",
    "libxc_functional",
    "plot_bandstructure",
    "plot_dos",
    "read_hdf5",
    "remove_core_fods",
    "split_fods",
    "view",
    "view_atoms",
    "view_contour",
    "view_file",
    "view_kpts",
    "write_hdf5",
]
