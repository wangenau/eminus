#!/usr/bin/env python3
'''
Collection of miscellaneous potentials.
'''
import numpy as np
from numpy.linalg import norm


def Harmonic(atoms):
    '''Harmonical potential, e.g., for quantum dot calculations.'''
    dr = norm(atoms.r - np.sum(atoms.R, axis=1) / 2, axis=1)
    V = 2 * dr**2
    return atoms.Jdag(atoms.O(atoms.J(V)))


def Coulomb(atoms):
    '''Coulomb potential, e.g., for Hydrogen calculations.'''
    Z = atoms.Z[0]  # Potential should only be used for same species
    Vps = -4 * np.pi * Z / atoms.G2[1:]
    Vps = np.concatenate(([0], Vps))

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vps * Sf)


def Ge(atoms):
    '''Starkloff-Joannopoulos pseudopotential for Germanium. Fourier transformed by Arias.'''
    Z = atoms.Z[0]  # Potential should only be used for same species/Germanium
    lamda = 18.5
    rc = 1.052
    Gm = np.sqrt(atoms.G2[1:])

    Vps = -2 * np.pi * np.exp(-np.pi * Gm / lamda) * np.cos(rc * Gm) * (Gm / lamda) / \
          (1 - np.exp(-2 * np.pi * Gm / lamda))
    for n in range(5):
        Vps += (-1)**n * np.exp(-lamda * rc * n) / (1 + (n * lamda / Gm)**2)
    Vps = Vps * 4 * np.pi * Z / Gm**2 * (1 + np.exp(-lamda * rc)) - 4 * np.pi * Z / Gm**2

    n = np.arange(1, 5)
    eps = 4 * np.pi * Z * (1 + np.exp(-lamda * rc)) * (rc**2 / 2 + 1 / lamda**2 * (np.pi**2 / 6 +
          np.sum((-1)**n * np.exp(-lamda * rc * n) / n**2)))
    Vps = np.concatenate(([eps], Vps))

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vps * Sf)


def init_pot(atoms):
    '''Use potentials from the Arias lectures.'''
    implemented = {'harmonic': Harmonic, 'coulomb': Coulomb, 'ge': Ge}
    try:
        pot = implemented[atoms.pot]
    except KeyError:
        print(f'ERROR: No potential found for "{atoms.pot}"')
    return pot(atoms)
