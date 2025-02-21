# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Symmetrize k-points.

All necessary dependencies to use this extra can be installed with::

    pip install eminus[symmetry]
"""

from ..logger import log


def symmetrize(atoms, space_group=False, time_reversal=True):
    """Symmetrize k-points of an Atoms object.

    Reference: WIREs Comput. Mol. Sci. 8, e1340.

    Args:
        atoms: Atoms object.

    Keyword Args:
        space_group: Whether to consider space group symmetry.
        time_reversal: Whether to consider time reversal symmetry.
    """
    try:
        from pyscf.pbc.gto import Cell
        from pyscf.pbc.lib.kpts import KPoints
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[symmetry]".\n\n'
        )
        raise
    if not atoms.kpts.is_built:
        atoms.kpts.build()

    # Build Cell and KPoints objects
    cell = Cell()
    cell.unit = "bohr"
    cell.atom = [[atoms.atom[ia], atoms.pos[ia]] for ia in range(atoms.Natoms)]
    cell.a = atoms.a
    cell.build()
    kpts = KPoints(cell, atoms.kpts.k)
    kpts.build(space_group_symmetry=space_group, time_reversal_symmetry=time_reversal)

    # Set the k-points
    atoms.kpts.k = kpts.kpts_ibz
    atoms.kpts.wk = kpts.weights_ibz
    atoms.kpts._k_scaled = None  # Remove the scaled k-points
    atoms.kpts.is_built = True  # Indicate the object as built
    atoms.is_built = False  # wk has to be passed to the occ object and G-vectors need to be updated
