#!/usr/bin/env python3
'''
Various tools for calculations or unit conversions.
'''
import numpy as np
from numpy.linalg import norm
from math import isclose
from .constants import BOHR, HARTREE, KCALMOL


def cutoff2gridspacing(E):
    '''Convert planewave energy cutoff to a real-space gridspacing using a.u..'''
    # Taken from GPAW: https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/tools.py
    return np.pi / np.sqrt(2 * E)


def gridspacing2cutoff(h):
    '''Convert real-space gridspacing to planewave energy cutoff using a.u..'''
    # Taken from GPAW: https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/tools.py
    # In Hartree units, E=k^2/2, where k_max is approx. given by pi/h
    # See PRB, Vol 54, 14362 (1996)
    return 0.5 * (np.pi / h)**2


def hartree2ev(E):
    '''Convert Hartree to electronvolt.'''
    return E * HARTREE


def ev2hartree(E):
    '''Convert electronvolt to Hartree.'''
    return E / HARTREE


def hartree2kcalmol(E):
    '''Convert Hartree to kcal/mol.'''
    return E * KCALMOL


def kcalmol2hartree(E):
    '''Convert kcal/mol to Hartree.'''
    return E / KCALMOL


def angstrom2bohr(r):
    '''Convert Angstrom to Bohr.'''
    return r / BOHR


def bohr2angstrom(r):
    '''Convert Bohr to Angstrom.'''
    return r * BOHR


def check_ortho(atoms, func):
    '''Check the orthogonality condition for a set of functions.'''
    # Orthogonality condition: \int func1^* func2 dr = 0
    # Tolerance for the condition
    eps = 1e-9
    # It makes no sense to calculate anything for only one function
    if len(func) == 1:
        print('Need at least two functions to check their orthogonality.')
        return

    # We integrate over our unit cell, the integration borders then become a=0 and b=cell length
    # The integration prefactor is (b-a)/n, with n as the sampling
    # For a 3d integral we have to multiply for every direction
    prefactor = atoms.a**3 / np.prod(atoms.S)

    ortho_bool = True

    # Check the condition for every combination
    for i in range(len(func)):
        for j in range(i + 1, len(func)):
            res = prefactor * np.sum(func[i].conj() * func[j])
            tmp_bool = np.abs(res) < eps
            ortho_bool *= tmp_bool
            if atoms.verbose > 2:
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
    prefactor = atoms.a**3 / np.prod(atoms.S)

    norm_bool = True

    # Check the condition for every combination
    for i in range(len(func)):
        res = prefactor * np.sum(func[i])
        tmp_bool = np.abs(atoms.f[i] - res) < eps
        norm_bool *= tmp_bool
        if atoms.verbose > 2:
            print(f'Function {i}:\n\tValue: {res:.7f}\n\tNormalized: {tmp_bool}')
    print(f'Normalized: {norm_bool}')
    return norm_bool


def check_orthonorm(atoms, func):
    '''Check the orthonormality conditions for a set of functions.'''
    ortho_bool = check_ortho(atoms, func)
    norm_bool = check_norm(atoms, func)
    print(f'Orthonormal: {ortho_bool * norm_bool}')
    return ortho_bool * norm_bool


def get_dipole(atoms, n):
    '''Calculate the electric dipole moment.'''
    # Diple moment: mu = \sum Z*X - \int n(r)*r dr
    mu = np.array([0, 0, 0], dtype=float)
    for i in range(len(atoms.X)):
        mu += atoms.Z[i] * atoms.X[i]

    # TODO: Why does thos prefactor even work here?
    prefactor = atoms.a**3 / np.prod(atoms.S)
    for dim in range(3):
        mu[dim] -= prefactor * np.sum(n * atoms.r[:, dim])
    return mu
