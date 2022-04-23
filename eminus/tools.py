#!/usr/bin/env python3
'''Various tools to check physical properties.'''
import numpy as np

from .scf import get_epsilon


def cutoff2gridspacing(E):
    '''Convert plane wave energy cut-off to a real-space grid spacing.

    Reference: Phys. Rev. B 54, 14362.

    Args:
        E (float): Energy in Hartree.

    Returns:
        float: Grid spacing in Bohr.
    '''
    return np.pi / np.sqrt(2 * E)


def gridspacing2cutoff(h):
    '''Convert real-space grid spacing to plane wave energy cut-off.

    Reference: Phys. Rev. B 54, 14362.

    Args:
        h (float): Grid spacing in Bohr.

    Returns:
        float: Cut-off in Hartree.
    '''
    return 0.5 * (np.pi / h)**2


def center_of_mass(coords, masses=None):
    '''Calculate the center of mass for a set of coordinates and masses.

    Args:
        coords (array): Array of real-space coordinates.

    Keyword Args:
        masses (array): Mass or weight for each coordinate.

    Returns:
        array: Center of mass.
    '''
    if masses is None:
        masses = np.ones(len(coords))

    return np.sum(masses * coords.T, axis=1) / np.sum(masses)


def inertia_tensor(coords, masses=None):
    '''Calculate the inertia tensor for a set of coordinates and masses.

    Args:
        coords (array): Array of real-space coordinates.

    Keyword Args:
        masses (array): Mass or weight for each coordinate.

    Returns:
        array: Inertia tensor.
    '''
    if masses is None:
        masses = np.ones(len(coords))

    # The inertia tensor for a set of point masses can be calculated with simple sumation
    # https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    I = np.empty((3, 3))
    I[0][0] = np.sum(masses * (coords[:, 1]**2 + coords[:, 2]**2))
    I[1][1] = np.sum(masses * (coords[:, 0]**2 + coords[:, 2]**2))
    I[2][2] = np.sum(masses * (coords[:, 0]**2 + coords[:, 1]**2))

    I[0][1] = I[1][0] = -np.sum(masses * (coords[:, 0] * coords[:, 1]))
    I[0][2] = I[2][0] = -np.sum(masses * (coords[:, 0] * coords[:, 2]))
    I[1][2] = I[2][1] = -np.sum(masses * (coords[:, 1] * coords[:, 2]))
    return I


def get_dipole(atoms):
    '''Calculate the electric dipole moment.

    Reference: J. Chem. Phys. 155, 224109.

    Args:
        atoms: Atoms object.

    Returns:
        array: Electric dipole moment in e times Bohr.
    '''
    # The dipole may be extremely large. This can be because of periodic boundary conditions,
    # e.g., the density gets "smeared" to the edges if the atom sits at one edge.
    # One fix can be to center the atom/molecule inside the box.
    n = atoms.n
    if n is None:
        print('ERROR: There is no density to calculate a dipole.')
        return 0

    # Diple moment: mu = \sum Z*X - \int n(r)*r dr
    mu = np.array([0, 0, 0], dtype=float)
    for i in range(atoms.Natoms):
        mu += atoms.Z[i] * atoms.X[i]

    dV = atoms.Omega / np.prod(atoms.s)
    for dim in range(3):
        mu[dim] -= dV * np.sum(n * atoms.r[:, dim])
    return mu


def get_IP(atoms):
    '''Calculate the ionization potential by calculating the negative HOMO energy.

    Reference: Physica 1, 104.

    Args:
        atoms: Atoms object.

    Returns:
        float: Ionization potential in Hartree.
    '''
    epsilon = get_epsilon(atoms, atoms.W)
    return -epsilon[-1]


def check_ortho(atoms, func, eps=1e-9):
    '''Check the orthogonality condition for a set of functions.

    Args:
        atoms: Atoms object.
        func (array): Discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Orthogonality status for the set of functions.
    '''
    # It makes no sense to calculate anything for only one function
    if len(func) == 1:
        print('WARNING: Need at least two functions to check their orthogonality.')
        return True

    # We integrate over our unit cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor dV is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    dV = atoms.Omega / np.prod(atoms.s)

    ortho_bool = True
    # Check the condition for every combination
    # Orthogonality condition: \int func1^* func2 dr = 0
    for i in range(func.shape[1]):
        for j in range(i + 1, func.shape[1]):
            res = dV * np.sum(func[:, i].conj() * func[:, j])
            tmp_bool = abs(res) < eps
            ortho_bool *= tmp_bool
            if atoms.verbose >= 3:
                print(f'Function {i} and {j}:\n\tValue: {res:.7f}\n\tOrthogonal: {tmp_bool}')
    print(f'Orthogonal: {ortho_bool}')
    return ortho_bool


def check_norm(atoms, func, eps=1e-9):
    '''Check the normalization condition for a set of functions.

    Args:
        atoms: Atoms object.
        func (array): Discretized set of functions.

    Keyword Args:
        eps (float): Tolerance for the condition.

    Returns:
        bool: Normalization status for the set of functions.
    '''
    # We integrate over our unit cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor dV is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    dV = atoms.Omega / np.prod(atoms.s)

    norm_bool = True
    # Check the condition for every function
    # Normality condition: \int func^* func dr = 1
    for i in range(func.shape[1]):
        res = dV * np.sum(func[:, i].conj() * func[:, i])
        tmp_bool = abs(1 - res) < eps
        norm_bool *= tmp_bool
        if atoms.verbose >= 3:
            print(f'Function {i}:\n\tValue: {res:.7f}\n\tNormalized: {tmp_bool}')
    print(f'Normalized: {norm_bool}')
    return norm_bool


def check_orthonorm(atoms, func):
    '''Check the orthonormality conditions for a set of functions.

    Args:
        atoms: Atoms object.
        func (array): Discretized set of functions.

    Returns:
        bool: Orthonormality status for the set of functions.
    '''
    ortho_bool = check_ortho(atoms, func)
    norm_bool = check_norm(atoms, func)
    print(f'Orthonormal: {ortho_bool * norm_bool}')
    return ortho_bool * norm_bool
