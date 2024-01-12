#!/usr/bin/env python3
"""Extra functions that need additional dependencies to work.

To also install all additional dependencies, use::

    pip install eminus[all]

Alternativle, you can only install selected extras using the respective name:

* :mod:`~eminus.extras.dispersion`
* :mod:`~eminus.extras.fods`
* :mod:`~eminus.extras.libxc`
* :mod:`~eminus.extras.symmetry`
* :mod:`~eminus.extras.torch`
* :mod:`~eminus.extras.viewer`
"""
from .dispersion import get_Edisp
from .fods import get_fods, remove_core_fods, split_fods
from .libxc import libxc_functional
from .symmetry import symmetrize
from .viewer import (
    executed_in_notebook,
    plot_bandstructure,
    view,
    view_atoms,
    view_contour,
    view_file,
    view_kpts,
)

__all__ = ['executed_in_notebook', 'get_Edisp', 'get_fods', 'libxc_functional',
           'plot_bandstructure', 'remove_core_fods', 'split_fods', 'symmetrize', 'view',
           'view_atoms', 'view_contour', 'view_file', 'view_kpts']
