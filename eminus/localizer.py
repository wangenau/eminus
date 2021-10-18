#!/usr/bin/env python3
'''
Utilities to localize and analyze orbitals.
'''
import numpy as np
from numpy.linalg import eig, norm


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
    R = np.empty((len(fods), len(fods)), dtype=complex)
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


def get_S(atoms, psirs):
    '''Calculate overlap matrix between orbitals.

    Args:
        atoms :
            Atoms object.

        psirs : array
            Set of orbitals in real-space.

    Returns:
        Overlap matrix S as an array.
    '''
    # Overlap elements: S_ij = \int psi_i^* psi_j dr
    S = np.empty((atoms.Ns, atoms.Ns), dtype=complex)

    dV = atoms.CellVol / np.prod(atoms.S)
    for i in range(atoms.Ns):
        for j in range(atoms.Ns):
            S[i][j] = dV * np.sum(psirs[:, i].conj() * psirs[:, j])
    return S


def get_FLOs(atoms, psi, fods):
    '''Calculate Fermi-Löwdin orbitals by orthonormalizing Fermi orbitals.

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
    # Calculate the overlap matrix for FOs
    S = get_S(atoms, FOs)
    # Calculate eigenvalues and eigenvectors
    Q, T = eig(S)
    # Löwdins symmetric orthonormalization method
    Q12 = np.diag(1 / np.sqrt(Q))
    return FOs @ (T @ Q12 @ T.T)


def wannier_cost(atoms, psirs):
    '''Calculate the Wannier cost function, namely the spread. Equivalent to Foster-Boys.

    Args:
        atoms :
            Atoms object.

        psirs : array
            Set of orbitals in real-space.

    Returns:
        Spread per orbital as an array.
    '''
    centers = wannier_center(atoms, psirs)
    moments = second_moment(atoms, psirs)
    costs = moments - norm(centers, axis=1)**2
    if atoms.verbose >= 3:
        print(f'Centers:\n{centers}\nMoments:\n{moments}')
    if atoms.verbose >= 2:
        print(f'Costs:\n{costs}')
    return np.sum(costs)


def wannier_center(atoms, psirs):
    '''Calculate Wannier centers that are the expecation values of r.

    Args:
        atoms :
            Atoms object.

        psirs : array
            Set of orbitals in real-space.

    Returns:
        Wannier centers per orbital as an array.
    '''
    dV = atoms.CellVol / np.prod(atoms.S)
    r = atoms.r

    centers = np.empty((psirs.shape[1], 3))
    for i in range(psirs.shape[1]):
        for dim in range(3):
            centers[i][dim] = dV * np.real(np.sum(psirs[:, i].conj() * r[:, dim] * psirs[:, i],
                              axis=0))
    return centers


def second_moment(atoms, psirs):
    '''Calculate the second moments that are the expecation values of r^2.

    Args:
        atoms :
            Atoms object.

        psirs : array
            Set of orbitals in real-space.

    Returns:
        Second moments per orbital as an array.
    '''
    dV = atoms.CellVol / np.prod(atoms.S)
    r2 = norm(atoms.r, axis=1)**2

    moments = np.empty(psirs.shape[1])
    for i in range(psirs.shape[1]):
        moments[i] = dV * np.real(np.sum(psirs[:, i].conj() * r2 * psirs[:, i], axis=0))
    return moments
