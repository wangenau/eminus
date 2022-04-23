#!/usr/bin/env python3
'''Collection of miscellaneous potentials.'''
import numpy as np
from numpy.linalg import norm


def Harmonic(atoms):
    '''Harmonical potential. Can be used for quantum dot calculations.

    Args:
        atoms: Atoms object.

    Returns:
        array: Harmonical potential in reciprocal space.
    '''
    dr = norm(atoms.r - np.sum(atoms.R, axis=1) / 2, axis=1)
    Vharm = 2 * dr**2
    return atoms.Jdag(atoms.O(atoms.J(Vharm)))


def Coulomb(atoms):
    '''All-electron Coulomb potential.

    Args:
        atoms: Atoms object.

    Returns:
        array: Coulomb potential in reciprocal space.
    '''
    Z = atoms.Z[0]  # Potential should only be used for same species
    # Ignore the division by zero for the first elements
    # One could do some proper indexing with [1:] but indexing is slow
    with np.errstate(divide='ignore', invalid='ignore'):
        Vcoul = -4 * np.pi * Z / atoms.G2
    Vcoul[0] = 0

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vcoul * Sf)


def Ge(atoms):
    '''Starkloff-Joannopoulos local pseudopotential for germanium.

    Fourier-transformed by Tomas Arias.
    Reference: Phys. Rev. B 16, 5212.

    Args:
        atoms: Atoms object.

    Returns:
        array: Germanium pseudopotential in reciprocal space.
    '''
    Z = 4  # Potential should only be used for germanium
    lamda = 18.5
    rc = 1.052
    Gm = np.sqrt(atoms.G2)
    Vps = np.empty_like(atoms.G2)

    with np.errstate(divide='ignore', invalid='ignore'):
        Vps = -2 * np.pi * np.exp(-np.pi * Gm / lamda) * np.cos(rc * Gm) * (Gm / lamda) / \
                  (1 - np.exp(-2 * np.pi * Gm / lamda))
        for n in range(5):
            Vps = Vps + (-1)**n * np.exp(-lamda * rc * n) / (1 + (n * lamda / Gm)**2)
        Vps = Vps * 4 * np.pi * Z / Gm**2 * (1 + np.exp(-lamda * rc)) - 4 * np.pi * Z / Gm**2

    # Special case for G=(0,0,0)
    n = np.arange(1, 5)
    Vps[0] = 4 * np.pi * Z * (1 + np.exp(-lamda * rc)) * (rc**2 / 2 + 1 / lamda**2 * (np.pi**2 / 6 +
             np.sum((-1)**n * np.exp(-lamda * rc * n) / n**2)))

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vps * Sf)


def init_pot(atoms):
    '''Handle and initialize potentials.

    Args:
        atoms: Atoms object.

    Returns:
        array: Potential in reciprocal space.
    '''
    implemented = {'harmonic': Harmonic, 'coulomb': Coulomb, 'ge': Ge}
    try:
        pot = implemented[atoms.pot]
    except KeyError:
        print(f'ERROR: No potential found for "{atoms.pot}"')
    return pot(atoms)
