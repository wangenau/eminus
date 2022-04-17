#!/usr/bin/env python3
'''
Fermi-orbital descriptor generation.
'''
from os import remove

import numpy as np
from numpy.linalg import norm
try:
    from pyscf.gto import M  # PySCF is a dependency of PyFLOSIC2
    from pyscf.scf import RKS
    from pyflosic2.atoms.atoms import Atoms
    from pyflosic2.guess.pycom import pycom
    from pyflosic2.parameters.flosic_parameters import parameters
except ImportError:
    print('ERROR: Necessary addon dependencies not found. '
          'To use this module, install the package with addons, e.g., with '
          '"pip install eminus[addons]"')

from eminus.atoms_io import read_xyz
from eminus.data import covalent_radii, symbol2number
from eminus.units import bohr2ang


def get_fods(atoms, basis='pc-0', loc='FB', clean=True):
    '''Generate FOD positions using the PyCOM method.

    Args:
        atoms: Atoms object.

    Keyword Args:
        basis (str): Basis set for the DFT calculation.

        loc (str): Localization method  (case insensitive).

        clean (bool): Remove pycom log files.

    Returns:
        array: FOD positions.
    '''
    loc = loc.upper()

    # Convert to Angstrom for PySCF
    X = bohr2ang(atoms.X)
    # Build the PySCF input format
    atom_pyscf = [i for i in zip(atoms.atom, X)]

    # Do the PySCF DFT calculation
    spin = np.sum(atoms.Z) % 2
    mol = M(atom=atom_pyscf, basis=basis, spin=spin)
    mf = RKS(mol=mol)
    mf.verbose = 0
    mf.kernel()

    # Add some FODs to the positions, otherwise the method wont work
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
    _, _, fods = split_atom_and_fod(atom, X)

    if clean:
        remove(p.log_name)
        remove(f'{loc}_GUESS_COM.xyz')
    return fods


def split_atom_and_fod(atom, X):
    '''Split atom and FOD coordinates.

    Args:
        atom (list): Atom symbols.

        X (array): Atom positions.

    Returns:
        tuple(list, array, array): Atom types, the respective coordinates, and FOD positions..
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


def remove_core_fods(atoms, fods, method='auto', radius=0.1):
    '''Remove core FODs from a set of FOD coordinates.

    Args:
        atoms: Atoms object.

        fods (array): FOD positions.

    Keyword Args:
        method (str): Method to remove FODs (case insensitive).

            'Automatic' will remove the nearest FODs of atoms depending on the number of core
            states. 'Radius' removes FODs within a fixed radius around atoms, 'covalent' does the
            same, but scaled by the atom's covalent radius.

        radius (float): Radii scaling.

            Radius size for 'radius' method, covalent radius scaling factor for 'covalent'. No
            effect for the 'automatic' method.

    Returns:
        array: Valence FODs.
    '''
    method = method.lower()
    if method in ('auto', 'automatic'):
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
    else:
        for ia in range(atoms.Natoms):
            dist = norm(fods - atoms.X[ia], axis=1)
            if method in ('rad', 'radius'):
                # Remove FODs within a sphere of given radius around atoms
                fods = fods[dist > radius]
            elif method in ('cov', 'covalent'):
                # Remove FODs within a sphere scaled by the covalent radius around atoms
                fods = fods[dist > radius * covalent_radii[atoms.atom[ia]]]
    return fods
