#!/usr/bin/env python3
'''Utilities to localize and analyze orbitals.'''
import numpy as np
from scipy.linalg import eig, norm

from .logger import log
from .utils import handle_spin_gracefully


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
    fo = np.zeros((atoms.Nspin, len(atoms.r), atoms.Nstate), dtype=complex)

    # Transform psi to real-space
    psi_rs = atoms.I(psi)
    for spin in range(atoms.Nspin):
        # Get the transformation matrix R
        R = get_R(atoms, psi[spin], fods[spin])
        for i in range(len(R)):
            for j in range(atoms.Nstate):
                fo[spin, :, i] += R[i, j] * psi_rs[spin, :, j]
    return fo


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
    S = np.empty((atoms.Nstate, atoms.Nstate), dtype=complex)

    dV = atoms.Omega / np.prod(atoms.s)
    for i in range(atoms.Nstate):
        for j in range(atoms.Nstate):
            S[i, j] = dV * np.sum(psirs[:, i].conj() * psirs[:, j])
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
    fo = get_FO(atoms, psi, fods)
    flo = np.empty((atoms.Nspin, len(atoms.r), atoms.Nstate), dtype=complex)

    for spin in range(atoms.Nspin):
        # Calculate the overlap matrix for FOs
        S = get_S(atoms, fo[spin])
        # Calculate eigenvalues and eigenvectors
        Q, T = eig(S)
        # Löwdins symmetric orthonormalization method
        Q12 = np.diag(1 / np.sqrt(Q))
        flo[spin] = fo[spin] @ (T @ Q12 @ T.T)
    return flo


@handle_spin_gracefully
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

    Reference: J. Chem. Phys. 137, 224114.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Wannier centers per orbital.
    '''
    dV = atoms.Omega / np.prod(atoms.s)

    centers = np.empty((atoms.Nstate, 3))
    for i in range(atoms.Nstate):
        for dim in range(3):
            centers[i, dim] = dV * np.real(np.sum(psirs[:, i].conj() * atoms.r[:, dim] *
                                           psirs[:, i], axis=0))
    return centers


def second_moment(atoms, psirs):
    '''Calculate the second moments, i.e., the expectation values of r^2.

    Reference: J. Chem. Phys. 137, 224114.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Second moments per orbital.
    '''
    dV = atoms.Omega / np.prod(atoms.s)
    r2 = norm(atoms.r, axis=1)**2

    moments = np.empty(atoms.Nstate)
    for i in range(atoms.Nstate):
        moments[i] = dV * np.real(np.sum(psirs[:, i].conj() * r2 * psirs[:, i], axis=0))
    return moments
