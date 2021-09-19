#!/usr/bin/env python3
'''
Utilities to localize and analyze orbitals.
'''
import numpy as np

from .scf import orth


def eval_psi(atoms, psi, r):
    '''Evaluate orbitals at given coordinate points.

    Args:
        atoms :
            Atoms object.

        psi : array
            Set of orbitals in reciprocal space.

        r : array
            Real-space positions.

    Returns:
        Values of psi at points r as an array.
    '''
    # Shift the evaluation point to (0,0,0), because we always have a lattice point there
    psi_T = atoms.T(psi, -r)
    psi_Trs = atoms.I(psi_T)
    # The zero entry is always the value at point (0,0,0)
    return psi_Trs[0]


def get_R(atoms, psi, fods):
    '''Calculate transformation matrix to build Fermi orbitals from Kohn-Sham orbitals.

    Args:
        atoms :
            Atoms object.

        psi : array
            Set of orbitals in reciprocal space.

        fods : array
            Fermi-orbital descriptors.

    Returns:
        Transformation matrix R as an array.
    '''
    # We only calculate occupied orbitals, so a zero matrix is enough
    R = np.zeros((atoms.Ns, atoms.Ns), dtype=complex)
    for i in range(len(fods)):
        # Get the value at one FOD position for all psi
        psi_fod = eval_psi(atoms, psi, fods[i])
        sum_psi_fod = np.sqrt(np.sum(psi_fod.conj() * psi_fod))
        for j in range(len(fods)):
            R[i, j] = psi_fod[j].conj() / sum_psi_fod
    return R


def get_FOs(atoms, psi, fods):
    '''Calculate Fermi orbitals from Kohn-Sham orbitals and a set of respective FODs.

    Args:
        atoms :
            Atoms object.

        psi : array
            Set of orbitals in reciprocal space.

        fods : array
            Fermi-orbital descriptors.

    Returns:
        Fermi orbitals as an array.
    '''
    FOs = np.zeros((len(atoms.r), atoms.Ns), dtype=complex)
    # Get the transformation matrix R
    R = get_R(atoms, psi, fods)
    # Transform psi to real space
    psi_rs = atoms.I(psi)
    for i in range(len(R)):
        for j in range(atoms.Ns):
            FOs[:, i] += R[i, j] * psi_rs[:, j]
    return FOs


def get_FLOs(atoms, psi, fods):
    '''Calculate Fermi-Löwdin orbitals by orthogornalizing Fermi orbitals.

    Args:
        atoms :
            Atoms object.

        psi : array
            Set of orbitals in reciprocal space.

        fods : array
            Fermi-orbital descriptors.

    Returns:
        Fermi-Löwdin orbitals as an array.
    '''
    FOs = get_FOs(atoms, psi, fods)
    return orth(atoms, FOs)
