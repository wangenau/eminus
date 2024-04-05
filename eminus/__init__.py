#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""eminus - A pythonic plane wave density functional theory (DFT) code.

Minimal usage example to do a DFT calculation for helium::

   from eminus import Atoms, SCF
   atoms = Atoms('He', (0, 0, 0))
   SCF(atoms).run()
"""

from . import config
from .atoms import Atoms
from .cell import Cell
from .dft import get_epsilon, get_psi
from .io import (
    read,
    read_cube,
    read_json,
    read_traj,
    read_xyz,
    write,
    write_cube,
    write_json,
    write_pdb,
    write_traj,
    write_xyz,
)
from .logger import log
from .scf import RSCF, SCF, USCF
from .version import __version__, info

__all__ = [
    'RSCF',
    'SCF',
    'USCF',
    'Atoms',
    'Cell',
    '__version__',
    'config',
    'get_epsilon',
    'get_psi',
    'info',
    'log',
    'read',
    'read_cube',
    'read_json',
    'read_traj',
    'read_xyz',
    'write',
    'write_cube',
    'write_json',
    'write_pdb',
    'write_traj',
    'write_xyz',
]


def demo():
    """Fast demo calculation for helium."""
    atoms = Atoms('He', (0, 0, 0), ecut=5)
    SCF(atoms).run()
