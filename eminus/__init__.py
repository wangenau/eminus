#!/usr/bin/env python3
'''eminus - A plane wave density functional theory code.

Minimal usage example to do a DFT calculation for helium::

   from eminus import Atoms, SCF
   atoms = Atoms('He', [0, 0, 0])
   SCF(atoms).run()
'''
from .atoms import Atoms
from .dft import get_epsilon, get_psi
from .io import (read, read_cube, read_json, read_xyz, write, write_cube, write_json, write_pdb,
                 write_xyz)
from .logger import log
from .scf import RSCF, SCF, USCF
from .version import __version__, info

__all__ = ['Atoms', 'get_epsilon', 'get_psi', 'info', 'log', 'read', 'read_cube', 'read_json',
           'read_xyz', 'RSCF', 'SCF', 'USCF', 'write', 'write_cube', 'write_json', 'write_pdb',
           'write_xyz', '__version__']
