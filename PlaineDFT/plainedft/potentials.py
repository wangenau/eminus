#!/usr/bin/env python3
'''
Potentials given by the assignments by Tomas Arias.
'''
import numpy as np
from numpy.linalg import norm


def Harm_pot(a):
    '''Harmonical potential, used for quantum dots.'''
    dr = norm(a.r - np.sum(a.R, axis=1) / 2, axis=1)
    V = 2 * dr**2
    Vdual = a.Jdag(a.O(a.J(V)))
    return Vdual


def Coulomb(a):
    '''Coulomb potential, used for Hydrogen.'''
    Vps = -4 * np.pi * a.Z[0] / a.G2[1:]
    return np.concatenate(([0], Vps))


def Ge(a):
    '''Starkloff-Joannopoulos pseudopotential for Germanium, Fourier transformed by Arias.'''
    lamda = 18.5
    rc = 1.052
    Gm = np.sqrt(a.G2[1:])
    Vps = -2 * np.pi * np.exp(-np.pi * Gm / lamda) * np.cos(rc * Gm) * (Gm / lamda) / (1 - np.exp(-2 * np.pi * Gm / lamda))
    for n in range(5):
        Vps = Vps + (-1)**n * np.exp(-lamda * rc * n) / (1 + (n * lamda / Gm)**2)
    Vps = Vps * 4 * np.pi * a.Z[0] / Gm**2 * (1 + np.exp(-lamda * rc)) - 4 * np.pi * a.Z[0] / Gm**2
    n = np.arange(1, 5)
    Vps0 = 4 * np.pi * a.Z[0] * (1 + np.exp(-lamda * rc)) * (rc**2 / 2 + 1 / lamda**2 * (np.pi**2 / 6 + np.sum((-1)**n * np.exp(-lamda * rc * n) / n**2)))
    return np.concatenate(([Vps0], Vps))


def init_pot(a):
    '''Use the potentials from Arias lecture.'''
    implemented = {'HARMONIC': Harm_pot, 'COULOMB': Coulomb, 'GE': Ge}
    pot = implemented[a.pot]
    Vps = pot(a)
    if a.pot == 'HARMONIC':
        Vdual = Vps
    else:
        Vdual = a.J(Vps * a.Sf)
    return Vdual
