#!/usr/bin/env python3
"""Utilities to localize and analyze orbitals."""
import numpy as np
from scipy.linalg import eig, expm, norm
from scipy.stats import unitary_group

from .logger import log
from .utils import handle_spin_gracefully, skip_k


@skip_k
def eval_psi(atoms, psi, r):
    """Evaluate orbitals at given coordinate points.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        r (ndarray): Real-space positions.

    Returns:
        ndarray: Values of psi at points r.
    """
    # Shift the evaluation point to (0,0,0) because we always have a lattice point here
    psi_T = atoms.T(psi, -r)
    psi_Trs = atoms.I(psi_T, 0)
    # The zero entry is always the value at point (0,0,0)
    return psi_Trs[0]


@skip_k
def get_R(atoms, psi, fods):
    """Calculate transformation matrix to build Fermi orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        fods (list): Fermi-orbital descriptors.

    Returns:
        ndarray: Transformation matrix R.
    """
    # We only calculate occupied orbitals
    R = np.empty((len(fods), len(fods)), dtype=complex)

    for i in range(len(fods)):
        # Get the value at one FOD position for all psi
        psi_fod = eval_psi(atoms, psi, fods[i])
        sum_psi_fod = np.sqrt(np.sum(psi_fod.conj() * psi_fod))
        for j in range(len(fods)):
            R[i, j] = psi_fod[j].conj() / sum_psi_fod
    return R


@skip_k
def get_FO(atoms, psi, fods):
    """Calculate Fermi orbitals from Kohn-Sham orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        fods (list): Fermi-orbital descriptors.

    Returns:
        ndarray: Real-space Fermi orbitals.
    """
    fo = np.zeros((atoms.occ.Nspin, atoms.Ns, atoms.occ.Nstate), dtype=complex)

    # Transform psi to real-space
    psi_rs = atoms.I(psi, 0)
    for spin in range(atoms.occ.Nspin):
        # Get the transformation matrix R
        R = get_R(atoms, psi[spin], fods[spin])
        for i in range(len(R)):
            for j in range(atoms.occ.Nstate):
                if atoms.occ.f[0, spin, j] > 0:
                    fo[spin, :, i] += R[i, j] * psi_rs[spin, :, j]
    return fo


@handle_spin_gracefully
def get_S(atoms, psirs):
    """Calculate overlap matrix between orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Overlap matrix S.
    """
    # Overlap elements: S_ij = \int psi_i^* psi_j dr
    S = np.empty((atoms.occ.Nstate, atoms.occ.Nstate), dtype=complex)

    for i in range(atoms.occ.Nstate):
        for j in range(atoms.occ.Nstate):
            S[i, j] = atoms.dV * np.sum(psirs[:, i].conj() * psirs[:, j])
    return S


@skip_k
def get_FLO(atoms, psi, fods):
    """Calculate Fermi-Loewdin orbitals by orthonormalizing Fermi orbitals.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.
        psi (ndarray): Set of orbitals in reciprocal space.
        fods (list): Fermi-orbital descriptors.

    Returns:
        ndarray: Real-space Fermi-Loewdin orbitals.
    """
    fo = get_FO(atoms, psi, fods)
    flo = np.empty((atoms.occ.Nspin, atoms.Ns, atoms.occ.Nstate), dtype=complex)

    for spin in range(atoms.occ.Nspin):
        # Calculate the overlap matrix for FOs
        S = get_S(atoms, fo[spin])
        # Calculate eigenvalues and eigenvectors
        Q, T = eig(S)
        # Loewdins symmetric orthonormalization method
        Q12 = np.diag(1 / np.sqrt(Q))
        flo[spin] = fo[spin] @ (T @ Q12 @ T.T)
    return flo


@skip_k
@handle_spin_gracefully
def wannier_cost(atoms, psirs):
    """Calculate the Wannier cost function, namely the orbital variance. Equivalent to Foster-Boys.

    This function does not account for periodicity, it may be a good idea to center the system.

    Reference: J. Chem. Phys. 137, 224114.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Variance per orbital.
    """
    # Variance = \int psi r^2 psi - (\int psi r psi)^2
    centers = wannier_center(atoms, psirs)
    moments = second_moment(atoms, psirs)
    costs = moments - norm(centers, axis=1)**2
    log.debug(f'Centers:\n{centers}\nMoments:\n{moments}')
    log.info(f'Costs:\n{costs}')
    return costs


@skip_k
@handle_spin_gracefully
def wannier_center(atoms, psirs):
    """Calculate Wannier centers, i.e., the expectation values of r.

    Reference: J. Chem. Phys. 137, 224114.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Wannier centers per orbital.
    """
    centers = np.empty((atoms.occ.Nstate, 3))
    for i in range(atoms.occ.Nstate):
        for dim in range(3):
            centers[i, dim] = atoms.dV * np.real(np.sum(psirs[:, i].conj() * atoms.r[:, dim] *
                                                 psirs[:, i], axis=0))
    return centers


@skip_k
@handle_spin_gracefully
def second_moment(atoms, psirs):
    """Calculate the second moments, i.e., the expectation values of r^2.

    Reference: J. Chem. Phys. 137, 224114.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        ndarray: Second moments per orbital.
    """
    r2 = norm(atoms.r, axis=1)**2

    moments = np.empty(atoms.occ.Nstate)
    for i in range(atoms.occ.Nstate):
        moments[i] = atoms.dV * np.real(np.sum(psirs[:, i].conj() * r2 * psirs[:, i], axis=0))
    return moments


@skip_k
@handle_spin_gracefully
def wannier_supercell_matrices(atoms, psirs):
    """Calculate matrices for the supercell Wannier localization.

    Reference: Phys. Rev. B 59, 9703.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Returns:
        tuple[ndarray, ndarray, ndarray]: Matrices X, Y, and Z.
    """
    # Similar to the expectation value of r, but accounting for periodicity
    X = (psirs.conj().T * np.exp(-1j * 2 * np.pi * atoms.r[:, 0] / atoms.a[0, 0])) @ psirs
    Y = (psirs.conj().T * np.exp(-1j * 2 * np.pi * atoms.r[:, 1] / atoms.a[1, 1])) @ psirs
    Z = (psirs.conj().T * np.exp(-1j * 2 * np.pi * atoms.r[:, 2] / atoms.a[2, 2])) @ psirs
    return X * atoms.dV, Y * atoms.dV, Z * atoms.dV


def wannier_supercell_cost(X, Y, Z):
    """Calculate the supercell Wannier cost.

    This is an equivalent criterion to the spread criterion, but not the same. This cost function
    will be maximized instead of the minimization of the spread.

    Reference: Phys. Rev. B 59, 9703.

    Args:
        X (ndarray): Calculation specific matrix.
        Y (ndarray): Calculation specific matrix.
        Z (ndarray): Calculation specific matrix.

    Returns:
        float: Supercell Wannier cost.
    """
    X2 = np.abs(np.diagonal(X))**2
    Y2 = np.abs(np.diagonal(Y))**2
    Z2 = np.abs(np.diagonal(Z))**2
    return np.sum(X2 + Y2 + Z2)


def wannier_supercell_grad(atoms, X, Y, Z):
    """Calculate the supercell Wannier gradient.

    Reference: Phys. Rev. B 59, 9703.

    Args:
        atoms: Atoms object.
        X (ndarray): Calculation specific matrix.
        Y (ndarray): Calculation specific matrix.
        Z (ndarray): Calculation specific matrix.

    Returns:
        ndarray: Supercell Wannier gradient.
    """
    x = np.zeros((atoms.occ.Nstate, atoms.occ.Nstate), dtype=complex)
    y = np.zeros((atoms.occ.Nstate, atoms.occ.Nstate), dtype=complex)
    z = np.zeros((atoms.occ.Nstate, atoms.occ.Nstate), dtype=complex)
    # Just the indexed gradient from the paper, without fancy optimization
    for n in range(atoms.occ.Nstate):
        for m in range(atoms.occ.Nstate):
            x[m, n] = X[n, m] * (X[n, n].conj() - X[m, m].conj()) \
                - X[m, n].conj() * (X[m, m] - X[n, n])
            y[m, n] = Y[n, m] * (Y[n, n].conj() - Y[m, m].conj()) \
                - Y[m, n].conj() * (Y[m, m] - Y[n, n])
            z[m, n] = Z[n, m] * (Z[n, n].conj() - Z[m, m].conj()) \
                - Z[m, n].conj() * (Z[m, m] - Z[n, n])
    return x + y + z


@skip_k
@handle_spin_gracefully
def get_wannier(atoms, psirs, Nit=10000, conv_tol=1e-7, mu=0.25, random_guess=False, seed=None):
    """Steepest descent supercell Wannier localization.

    This function is rather sensitive to the starting point, thus it is a good idea to start from
    already localized orbitals.

    This optimizes the given orbitals under unitary constraint matrices, see
    IEEE Trans. Signal Process. 56, 1134.

    Reference: Phys. Rev. B 59, 9703.

    Args:
        atoms: Atoms object.
        psirs (ndarray): Set of orbitals in real-space.

    Keyword Args:
        Nit (int): Number of iterations.
        conv_tol (float): Convergence tolerance.
        mu (float): Step size.
        random_guess (bool): Whether to use a random unitary starting guess or the identity.
        seed (int | None): Seed to get a reproducible random guess.

    Returns:
        ndarray: Localized orbitals.
    """
    if not (np.diag(np.diag(atoms.a)) == atoms.a).all():
        log.warning('The Wannier localization needs a cubic unit cell.')
        return psirs

    X, Y, Z = wannier_supercell_matrices(atoms, psirs)  # Calculate matrices only once
    # The initial unitary transformation is the identity or a random unitary matrix
    if random_guess and atoms.occ.Nstate > 1:
        U = unitary_group.rvs(atoms.occ.Nstate, random_state=seed)
    else:
        U = np.eye(atoms.occ.Nstate)
    costs = [0]  # Add a zero to the costs to allow the sign evaluation in the first iteration

    atoms.log.debug(f'{"Iteration":<11}{"Cost [a0^2]":<13}{"dCost [a0^2]":<13}')
    for i in range(Nit):
        sign = 1
        costs.append(wannier_supercell_cost(X, Y, Z))
        if abs(costs[-2] - costs[-1]) < conv_tol:
            atoms.log.info(f'Wannier localizer converged after {i} iterations.')
            break
        # If the cost function gets smaller, change the direction
        if costs[-2] - costs[-1] < 0:
            sign = -1

        # Calculate unitary transformation
        dOmega = wannier_supercell_grad(atoms, X, Y, Z)
        A = sign * mu * dOmega
        # dOmega is anti-hermitian, therefore calculate -A instead of A.conj().T
        # expm(A) will be unitary
        expA_pos, expA_neg = expm(A), expm(-A)
        # Update total rotation
        U = U @ expA_pos
        # Update matrices
        X = expA_neg @ X @ expA_pos
        Y = expA_neg @ Y @ expA_pos
        Z = expA_neg @ Z @ expA_pos

        atoms.log.debug(f'{i:>8}   {costs[-1]:<+13,.6f}{costs[-2] - costs[-1]:<+13,.4e}')

    if len(costs) > 1 and abs(costs[-2] - costs[-1]) > conv_tol:
        atoms.log.warning('Wannier localizer not converged!')
    # Return the localized orbitals by rotating them
    return psirs @ U
