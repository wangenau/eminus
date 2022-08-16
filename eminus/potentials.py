#!/usr/bin/env python3
'''Collection of miscellaneous potentials.'''
import numpy as np
from scipy.linalg import norm

from .logger import log


def harmonic(atoms):
    '''Harmonic potential. Can be used for quantum dot calculations.

    Args:
        atoms: Atoms object.

    Returns:
        ndarray: Harmonic potential in reciprocal space.
    '''
    freq = 2
    dr = norm(atoms.r - np.sum(atoms.R, axis=1) / 2, axis=1)
    Vharm = 0.5 * freq**2 * dr**2
    return atoms.Jdag(atoms.O(atoms.J(Vharm)))


def coulomb(atoms):
    '''All-electron Coulomb potential.

    Args:
        atoms: Atoms object.

    Returns:
        ndarray: Coulomb potential in reciprocal space.
    '''
    Z = atoms.Z[0]  # Potential should only be used for same species

    # Ignore the division by zero for the first elements
    # One could do some proper indexing with [1:] but indexing is slow
    with np.errstate(divide='ignore', invalid='ignore'):
        Vcoul = -4 * np.pi * Z / atoms.G2
    Vcoul[0] = 0

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vcoul * Sf)


def ge(atoms):
    '''Starkloff-Joannopoulos local pseudopotential for germanium.

    Fourier-transformed by Tomas Arias.
    Reference: Phys. Rev. B 16, 5212.

    Args:
        atoms: Atoms object.

    Returns:
        ndarray: Germanium pseudopotential in reciprocal space.
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
    Vps[0] = 4 * np.pi * Z * (1 + np.exp(-lamda * rc)) * \
        (rc**2 / 2 + 1 / lamda**2 * (np.pi**2 / 6 +
         np.sum((-1)**n * np.exp(-lamda * rc * n) / n**2)))

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vps * Sf)


def init_pot(scf):
    '''Handle and initialize potentials.

    Args:
        scf: SCF object.

    Returns:
        ndarray: Potential in reciprocal space.
    '''
    try:
        pot = eval(scf.pot)(scf.atoms)
    except NameError:
        log.exception(f'No potential found for "{scf.pot}"')
        raise
    return pot
