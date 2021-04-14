#!/usr/bin/env python3
'''
Potentials given by the assignments by Tomas Arias.
'''
import numpy as np
from numpy.linalg import norm


def Harm_pot(atoms):
    '''Harmonical potential, used for quantum dots.'''
    dr = norm(atoms.r - np.sum(atoms.R, axis=1) / 2, axis=1)
    V = 2 * dr**2
    Vdual = atoms.Jdag(atoms.O(atoms.J(V)))
    return Vdual


def Coulomb(atoms):
    '''Coulomb potential, used for Hydrogen.'''
    Z = atoms.Z[0]  # Potential should only be used for same species
    Vps = -4 * np.pi * Z / atoms.G2[1:]
    return np.concatenate(([0], Vps))


def Ge(atoms):
    '''Starkloff-Joannopoulos pseudopotential for Germanium, Fourier transformed by Arias.'''
    Z = atoms.Z[0]  # Potential should only be used for same species
    lamda = 18.5
    rc = 1.052
    Gm = np.sqrt(atoms.G2[1:])
    Vps = -2 * np.pi * np.exp(-np.pi * Gm / lamda) * np.cos(rc * Gm) * (Gm / lamda) / \
          (1 - np.exp(-2 * np.pi * Gm / lamda))
    for n in range(5):
        Vps = Vps + (-1)**n * np.exp(-lamda * rc * n) / (1 + (n * lamda / Gm)**2)
    Vps = Vps * 4 * np.pi * Z / Gm**2 * (1 + np.exp(-lamda * rc)) - 4 * np.pi * Z / Gm**2
    n = np.arange(1, 5)
    Vps0 = 4 * np.pi * Z * (1 + np.exp(-lamda * rc)) * (rc**2 / 2 + 1 / lamda**2 * (np.pi**2 / 6 +
           np.sum((-1)**n * np.exp(-lamda * rc * n) / n**2)))
    return np.concatenate(([Vps0], Vps))


def init_pot(atoms):
    '''Use the potentials from Arias lecture.'''
    implemented = {'HARMONIC': Harm_pot, 'COULOMB': Coulomb, 'GE': Ge}
    pot = implemented[atoms.pot]
    Vps = pot(atoms)
    if atoms.pot == 'HARMONIC':
        Vdual = Vps
    else:
        Vdual = atoms.J(Vps * atoms.Sf)
    return Vdual
