#!/usr/bin/env python3
'''Utilities to localize and analyze orbitals.'''
import numpy as np
from numpy.linalg import eig, norm

from .logger import log


def eval_psi(atoms, psi, r):
    '''Evaluate orbitals at given coordinate points.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        r (ndarray): Real-space positions.

    Returns:
        ndarray: Values of psi at points r.
    '''
    # Shift the evaluation point to (0,0,0), because we always have a lattice point there
    psi_T = atoms.T(psi, -r)
    psi_Trs = atoms.I(psi_T)
    # The zero entry is always the value at point (0,0,0)
    return psi_Trs[0]


def get_R(atoms, psi, fods):
    '''Calculate transformation matrix to build Fermi orbitals from Kohn-Sham orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        fods (ndarray): Fermi-orbital descriptors.

    Returns:
        ndarray: Transformation matrix R.
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


def get_FO(atoms, psi, fods):
    '''Calculate Fermi orbitals from Kohn-Sham orbitals and a set of respective FODs.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        fods (ndarray): Fermi-orbital descriptors.

    Returns:
        ndarray: Real-space Fermi orbitals.
    '''
    FO = np.zeros((len(atoms.r), atoms.Ns), dtype=complex)
    # Get the transformation matrix R
    R = get_R(atoms, psi, fods)
    # Transform psi to real-space
    psi_rs = atoms.I(psi)
    for i in range(len(R)):
        for j in range(atoms.Ns):
            FO[:, i] += R[i, j] * psi_rs[:, j]
    return FO


def get_S(atoms, psirs):
    '''Calculate overlap matrix between orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Overlap matrix S.
    '''
    # Overlap elements: S_ij = \int psi_i^* psi_j dr
    S = np.empty((atoms.Ns, atoms.Ns), dtype=complex)

    dV = atoms.Omega / np.prod(atoms.s)
    for i in range(atoms.Ns):
        for j in range(atoms.Ns):
            S[i][j] = dV * np.sum(psirs[:, i].conj() * psirs[:, j])
    return S


def get_FLO(atoms, psi, fods):
    '''Calculate Fermi-Löwdin orbitals by orthonormalizing Fermi orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        fods (ndarray): Fermi-orbital descriptors.

    Returns:
        ndarray: Real-space Fermi-Löwdin orbitals.
    '''
    FO = get_FO(atoms, psi, fods)
    # Calculate the overlap matrix for FOs
    S = get_S(atoms, FO)
    # Calculate eigenvalues and eigenvectors
    Q, T = eig(S)
    # Löwdins symmetric orthonormalization method
    Q12 = np.diag(1 / np.sqrt(Q))
    return FO @ (T @ Q12 @ T.T)


def wannier_cost(atoms, psirs):
    '''Calculate the Wannier cost function, namely the orbital variance. Equivalent to Foster-Boys.

    Reference: J. Chem. Phys. 137, 224114.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Variance per orbital.
    '''
    # Variance = \int psi r^2 psi - (\int psi r psi)^2
    centers = wannier_center(atoms, psirs)
    moments = second_moment(atoms, psirs)
    costs = moments - norm(centers, axis=1)**2
    log.debug(f'Centers:\n{centers}\nMoments:\n{moments}')
    log.info(f'Costs:\n{costs}')
    return costs


def wannier_center(atoms, psirs):
    '''Calculate Wannier centers, i.e., the expectation values of r.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Wannier centers per orbital.
    '''
    dV = atoms.Omega / np.prod(atoms.s)
    r = atoms.r

    centers = np.empty((atoms.Ns, 3))
    for i in range(atoms.Ns):
        for dim in range(3):
            centers[i][dim] = dV * np.real(np.sum(psirs[:, i].conj() * r[:, dim] * psirs[:, i],
                              axis=0))
    return centers


def second_moment(atoms, psirs):
    '''Calculate the second moments, i.e., the expectation values of r^2.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Second moments per orbital.
    '''
    dV = atoms.Omega / np.prod(atoms.s)
    r2 = norm(atoms.r, axis=1)**2

    moments = np.empty(atoms.Ns)
    for i in range(atoms.Ns):
        moments[i] = dV * np.real(np.sum(psirs[:, i].conj() * r2 * psirs[:, i], axis=0))
    return moments
