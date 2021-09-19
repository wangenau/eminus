#!/usr/bin/env python3
'''
PlaineDFT - A simple plane wave DFT code.

Minimal usage example to do a DFT calculation for Hydrogen:

| from plainedft import *
| atoms = Atoms('H', [0, 0, 0])
| SCF(atoms)
'''
from .atoms import Atoms, load_atoms, read_xyz, save_atoms, write_cube, write_xyz
from .scf import SCF
from .version import __version__

__all__ = ['Atoms', 'load_atoms', 'read_xyz', 'save_atoms', 'SCF', 'write_cube', 'write_xyz',
           '__version__']
