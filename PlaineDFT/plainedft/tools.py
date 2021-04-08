#!/usr/bin/env python3
'''
Various tools for calculations or unit conversions.
'''
import numpy as np
from .constants import BOHR, HARTREE, KCALMOL


def cutoff2gridspacing(E):
    '''Convert planewave energy cutoff to a real-space gridspacing using a.u..'''
    # Taken from GPAW: https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/tools.py
    return np.pi / np.sqrt(2 * E)


def gridspacing2cutoff(h):
    '''Convert real-space gridspacing to planewave energy cutoff using a.u..'''
    # Taken from GPAW: https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/tools.py
    # In Hartree units, E = k^2 / 2, where k_max is approx. given by pi / h
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
