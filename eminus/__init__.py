#!/usr/bin/env python3
'''
eminus - A plane wave density funtional theory code.

Minimal usage example to do a DFT calculation for Hydrogen::

    from eminus import *
    atoms = Atoms('H', [0, 0, 0])
    SCF(atoms)
'''
from .atoms import Atoms
from .atoms_io import load_atoms, read_cube, read_xyz, save_atoms, write_cube, write_xyz
from .scf import get_epsilon, get_psi, SCF
from .version import info, __version__

__all__ = ['Atoms', 'get_epsilon', 'get_psi', 'info', 'load_atoms', 'read_cube', 'read_xyz',
           'save_atoms', 'SCF', 'write_cube', 'write_xyz', '__version__']
