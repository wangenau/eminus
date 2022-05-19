#!/usr/bin/env python3
'''Fermi-orbital descriptor generation.'''
import os

import numpy as np
from numpy.linalg import norm
try:
    from pyflosic2.atoms.atoms import Atoms
    from pyflosic2.guess.pycom import pycom
    from pyflosic2.parameters.flosic_parameters import parameters
    from pyscf.gto import M  # PySCF is a dependency of PyFLOSIC2
    from pyscf.scf import RKS
except ImportError:
    print('ERROR: Necessary addon dependencies not found. To use this module,\n'
          '       install the package with addons, e.g., with "pip install eminus[addons]"')

from ..data import symbol2number
from ..filehandler import read_xyz
from ..units import bohr2ang


def get_fods(object, basis='pc-0', loc='FB', clean=True):
    '''Generate FOD positions using the PyCOM method.

    Reference: J. Comput. Chem. 40, 2843.

    Args:
        object: Atoms or SCF object.

    Keyword Args:
        basis (str): Basis set for the DFT calculation.
        loc (str): Localization method (case insensitive).
        clean (bool): Remove log files.

    Returns:
        ndarray: FOD positions.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object
    loc = loc.upper()

    # Convert to Angstrom for PySCF
    X = bohr2ang(atoms.X)
    # Build the PySCF input format
    atom_pyscf = list(zip(atoms.atom, X))

    # Do the PySCF DFT calculation
    spin = np.sum(atoms.Z) % 2
    mol = M(atom=atom_pyscf, basis=basis, spin=spin)
    mf = RKS(mol=mol)
    mf.verbose = 0
    mf.kernel()

    # Add some FODs to the positions, otherwise the method will not work
    extra = np.zeros((atoms.Ns, 3))
    atom_pyflosic = atoms.atom + ['X'] * len(extra)
    X_pyflosic = np.vstack((X, extra))

    # Do the pycom FOD generation
    atoms = Atoms(atom_pyflosic, X_pyflosic, elec_symbols=['X', None], spin=spin)
    p = parameters(mode='restricted')
    p.init_atoms(atoms)
    p.basis = basis
    p.pycom_loc = loc
    pc = pycom(mf=mf, p=p)
    pc.get_guess()

    # Get the actual FOD positions from the xyz file
    atom, X = read_xyz(f'{loc}_GUESS_COM.xyz')
    _, _, fods = split_fods(atom, X)

    if clean:
        os.remove(p.log_name)
        os.remove(f'{loc}_GUESS_COM.xyz')
    return fods


def split_fods(atom, X):
    '''Split atom and FOD coordinates.

    Args:
        atom (list): Atom symbols.
        X (ndarray): Atom positions.

    Returns:
        tuple[list, ndarray, ndarray]: Atom types, the respective coordinates, and FOD positions.
    '''
    X_fod = []
    # Iterate in reverted order, because we may delete elements
    for ia in range(len(X) - 1, -1, -1):
        if atom[ia] == 'X':
            X_fod.append(X[ia])
            X = np.delete(X, ia, axis=0)
            del atom[ia]
    X_fod = np.asarray(X_fod)
    return atom, X, X_fod


def remove_core_fods(object, fods):
    '''Remove core FODs from a set of FOD coordinates.

    Args:
        object: Atoms or SCF object.
        fods (ndarray): FOD positions.

    Returns:
        ndarray: Valence FOD positions.
    '''
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object

    for ia in range(atoms.Natoms):
        n_core = symbol2number[atoms.atom[ia]] - atoms.Z[ia]
        # In the spin-paired case two electrons are one state
        # Since only core states are removed in pseudopotentials this value is divisible by 2
        # +1 to account for uneven amount of core fods (e.g., for hydrogen)
        n_core = (n_core + 1) // 2
        dist = norm(fods - atoms.X[ia], axis=1)
        idx = np.argsort(dist)
        # Remove core FODs with the smallest distance to the core
        fods = np.delete(fods, idx[:n_core], axis=0)
    return fods
