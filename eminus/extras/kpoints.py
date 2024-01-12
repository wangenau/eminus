#!/usr/bin/env python3
"""Symmetrize k-points."""

from ..logger import log


def symmetrize(atoms, space_group_symmetry=True, time_reversal_symmetry=False):
    """Symmetrize k-points of an Atoms object.

    Args:
        atoms: Atoms object.

    Keyword Args:
        space_group_symmetry (bool): Whether to consider space group symmetry.
        time_reversal_symmetry (bool): Whether to consider time reversal symmetry.
    """
    try:
        from pyscf.pbc.gto import Cell
        from pyscf.pbc.lib.kpts import KPoints
    except ImportError:
        log.exception('Necessary dependencies not found. To use this module, '
                      'install them with "pip install eminus[kpoints]".\n\n')
        raise
    if not atoms.kpts.is_built:
        atoms.kpts.build()

    # Build Cell and KPoints objects
    cell = Cell()
    cell.unit = 'bohr'
    cell.atom = [[atoms.atom[ia], atoms.pos[ia]] for ia in range(atoms.Natoms)]
    cell.a = atoms.a
    cell.build()
    kpts = KPoints(cell, atoms.kpts.k)
    kpts.build(space_group_symmetry=space_group_symmetry,
               time_reversal_symmetry=time_reversal_symmetry)

    # Set the k-points
    atoms.kpts.k = kpts.kpts_ibz
    atoms.kpts.wk = kpts.weights_ibz
    atoms.kpts._k_scaled = None  # Remove the scaled k-points
    atoms.kpts.is_built = True  # Indicate the object as built
