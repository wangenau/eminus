#!/usr/bin/env python3
'''
Various tools to check physical properties.
'''
import numpy as np


# Adapted from GPAW: https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/tools.py
def cutoff2gridspacing(E):
    '''Convert planewave energy cutoff to a real-space gridspacing using a.u..'''
    return np.pi / np.sqrt(2 * E)


# Adapted from GPAW: https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/tools.py
def gridspacing2cutoff(h):
    '''Convert real-space gridspacing to planewave energy cutoff using a.u..'''
    # In Hartree units, E=k^2/2, where k_max is approx. given by pi/h
    # See PRB, Vol 54, 14362 (1996)
    return 0.5 * (np.pi / h)**2


def center_of_mass(coords, masses=None):
    '''Calculate the center of mass for a set of coordinates and masses.'''
    if masses is None:
        masses = np.ones(len(coords))

    return np.sum(masses * coords.T, axis=1) / np.sum(masses)


def inertia_tensor(coords, masses=None):
    '''Calculate the inertia tensor for a set of coordinates and masses.'''
    if masses is None:
        masses = np.ones(len(coords))

    # For the definition see https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    I = np.zeros((3, 3))
    I[0][0] = np.sum(masses * (coords[:, 1]**2 + coords[:, 2]**2))
    I[1][1] = np.sum(masses * (coords[:, 0]**2 + coords[:, 2]**2))
    I[2][2] = np.sum(masses * (coords[:, 0]**2 + coords[:, 1]**2))

    I[0][1] = I[1][0] = -np.sum(masses * (coords[:, 0] * coords[:, 1]))
    I[0][2] = I[2][0] = -np.sum(masses * (coords[:, 0] * coords[:, 2]))
    I[1][2] = I[2][1] = -np.sum(masses * (coords[:, 1] * coords[:, 2]))
    return I


def get_dipole(atoms):
    '''Calculate the electric dipole moment.'''
    # The dipole may be extremly large. This can be because of periodic boundary conditions.
    # E.g., the density gets "smeared" to the edges if the atom sits at one edge.
    # One fix can be to center the atom/molecule inside the box.
    n = atoms.n
    if n is None:
        print('ERROR: There is no density to calculate a dipole.')
        return 0

    # Diple moment: mu = \sum Z*X - \int n(r)*r dr
    mu = np.array([0, 0, 0], dtype=float)
    for i in range(len(atoms.X)):
        mu += atoms.Z[i] * atoms.X[i]

    prefactor = atoms.CellVol / np.prod(atoms.S)
    for dim in range(3):
        mu[dim] -= prefactor * np.sum(n * atoms.r[:, dim])
    return mu


def check_ortho(atoms, func):
    '''Check the orthogonality condition for a set of functions.'''
    # Orthogonality condition: \int func1^* func2 dr = 0
    # Tolerance for the condition
    eps = 1e-9
    # It makes no sense to calculate anything for only one function
    if len(func) == 1:
        print('WARNING: Need at least two functions to check their orthogonality.')
        return True

    # We integrate over our unit cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    prefactor = atoms.CellVol / np.prod(atoms.S)

    ortho_bool = True

    # Check the condition for every combination
    for i in range(len(func)):
        for j in range(i + 1, len(func)):
            res = prefactor * np.sum(func[i].conj() * func[j])
            tmp_bool = np.abs(res) < eps
            ortho_bool *= tmp_bool
            if atoms.verbose >= 3:
                print(f'Function {i} and {j}:\n\tValue: {res:.7f}\n\tOrthogonal: {tmp_bool}')
    print(f'Orthogonal: {ortho_bool}')
    return ortho_bool


def check_norm(atoms, func):
    '''Check the normalization condition for a set of functions.'''
    # Orthogonality condition: \int func dr = 1
    # Tolerance for the condition
    eps = 1e-9
    # We integrate over our unit cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    prefactor = atoms.CellVol / np.prod(atoms.S)

    norm_bool = True

    # Check the condition for every combination
    for i in range(len(func)):
        res = prefactor * np.sum(func[i])
        tmp_bool = np.abs(atoms.f[i] - res) < eps
        norm_bool *= tmp_bool
        if atoms.verbose >= 3:
            print(f'Function {i}:\n\tValue: {res:.7f}\n\tNormalized: {tmp_bool}')
    print(f'Normalized: {norm_bool}')
    return norm_bool


def check_orthonorm(atoms, func):
    '''Check the orthonormality conditions for a set of functions.'''
    ortho_bool = check_ortho(atoms, func)
    norm_bool = check_norm(atoms, func)
    print(f'Orthonormal: {ortho_bool * norm_bool}')
    return ortho_bool * norm_bool
