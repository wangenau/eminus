#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Cell wrapper function."""

import numpy as np

from .atoms import Atoms
from .data import LATTICE_VECTORS
from .utils import molecule2list

#: Crystal structures with their respective lattice and basis.
STRUCTURES = {
    'sc': {
        'lattice': 'sc',
        'basis': [[0, 0, 0]],
    },
    'fcc': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0]],
    },
    'bcc': {
        'lattice': 'bcc',
        'basis': [[0, 0, 0]],
    },
    'tetragonal': {
        'lattice': 'sc',
        'basis': [[0, 0, 0]],
    },
    'orthorhombic': {
        'lattice': 'sc',
        'basis': [[0, 0, 0]],
    },
    'hexagonal': {
        'lattice': 'hexagonal',
        'basis': [[0, 0, 0]],
    },
    'diamond': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]],
    },
    'zincblende': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]],
    },
    'rocksalt': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 2, 0, 0]],
    },
    'cesiumchloride': {
        'lattice': 'sc',
        'basis': [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
    },
    'fluorite': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4]],
    },
}


def Cell(
    atom, lattice, ecut, a, basis=None, bands=None, kmesh=1, smearing=0, verbose=None, **kwargs
):
    """Wrapper to create Atoms classes for crystal systems.

    Args:
        atom (str | list | tuple): Atom symbols.
        lattice (str | list | tuple | ndarray): Lattice system.
        ecut (float):  Cut-off energy.
        a (float | list | tuple | ndarray | None): Cell size.

    Keyword Args:
        basis (list | tuple | ndarray | None): Lattice basis.
        bands (int | None): Number of bands (has to be larger or equal than the occupied states).
        kmesh (int | list | tuple | ndarray): k-point mesh.
        smearing (float): Smearing width in Hartree.
        verbose (int | str | None): Level of output.
        **kwargs: Keyword arguments to pass to the Atoms object.

    Returns:
        Atoms object.
    """
    # Get the lattice vectors from a string or use them directly
    if isinstance(lattice, str):
        lattice = lattice.lower()
        if basis is None:
            basis = STRUCTURES[lattice]['basis']
        lattice = STRUCTURES[lattice]['lattice']
        lattice_vectors = np.asarray(LATTICE_VECTORS[lattice])
    else:
        if basis is None:
            basis = [[0, 0, 0]]
        lattice_vectors = np.asarray(lattice)

    # Only scale the lattice vectors with if a is given
    if a is not None:
        lattice_vectors = a * lattice_vectors
        basis = a * np.asarray(basis)

    # Account for different atom and basis sizes
    if isinstance(atom, str):
        atom_list = molecule2list(atom)
    else:
        atom_list = atom
    if len(atom_list) != len(basis):
        atom = [atom] * len(basis)

    # Build the atoms object
    atoms = Atoms(atom, basis, ecut=ecut, a=lattice_vectors, verbose=verbose, **kwargs)
    # Handle k-points and states
    atoms.kpts.kmesh = kmesh
    if isinstance(lattice, str):
        atoms.kpts.lattice = lattice
    atoms.occ.smearing = smearing
    if bands is not None:
        atoms.occ.bands = bands
    return atoms
