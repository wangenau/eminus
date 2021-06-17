#!/usr/bin/env python3
'''
Collection of constants that may be needed throughout the calculation.
'''

# Ha in eV (https://en.wikipedia.org/wiki/Hartree)
HARTREE = 27.211386245988
# a0 in Angstrom (https://en.wikipedia.org/wiki/Bohr_radius)
KCALMOL = 627.5094740631
# Ha in kcal/mol (https://en.wikipedia.org/wiki/Hartree)
BOHR = 0.529177210903
# e x a0 in D (https://en.wikipedia.org/wiki/Hartree_atomic_units)
DEBYE = 2.541746473


def ha2ev(E):
    '''Convert Hartree to electronvolt.'''
    return E * HARTREE


def ev2ha(E):
    '''Convert electronvolt to Hartree.'''
    return E / HARTREE


def ha2kcalmol(E):
    '''Convert Hartree to kcal/mol.'''
    return E * KCALMOL


def kcalmol2ha(E):
    '''Convert kcal/mol to Hartree.'''
    return E / KCALMOL


def ev2kcalmol(E):
    '''Convert electronvolt to kcal/mol.'''
    return ha2kcalmol(ev2ha(E))


def kcalmol2ev(E):
    '''Convert kcal/mol to electronvolt.'''
    return ha2ev(kcalmol2ha(E))


def ha2ry(E):
    '''Convert Hartree to Rydberg.'''
    return 2 * E


def ry2ha(E):
    '''Convert Rydberg to Hartree.'''
    return E / 2


def ang2bohr(r):
    '''Convert Angstrom to Bohr.'''
    return r / BOHR


def bohr2ang(r):
    '''Convert Bohr to Angstrom.'''
    return r * BOHR


def ebohr2d(p):
    '''Convert e x Bohr to Debye.'''
    return p * DEBYE


def d2ebohr(p):
    '''Convert Debye to e x Bohr.'''
    return p / DEBYE
