#!/usr/bin/env python3
'''eminus - A plane wave density functional theory code.

Minimal usage example to do a DFT calculation for helium::

   from eminus import Atoms, SCF
   atoms = Atoms('He', [0, 0, 0])
   SCF(atoms)
'''
from .atoms import Atoms
from .dft import get_epsilon, get_psi
from .filehandler import load, read_cube, read_xyz, save, write_cube, write_xyz
from .logger import log
from .scf import RSCF, SCF, USCF
from .version import __version__, info

__all__ = ['Atoms', 'get_epsilon', 'get_psi', 'info', 'load', 'log', 'read_cube', 'read_xyz',
           'RSCF', 'save', 'SCF', 'USCF', 'write_cube', 'write_xyz', '__version__']
