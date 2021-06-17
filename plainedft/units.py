#!/usr/bin/env python3
'''
Collection of constants and conversions that may be needed throughout the calculation.
'''

# Ha in eV (https://en.wikipedia.org/wiki/Hartree)
electronvolt = eV = 27.211386245988
# Ha in kcal/mol (https://en.wikipedia.org/wiki/Hartree)
kcalmol = 627.5094740631
# a0 in Å (https://en.wikipedia.org/wiki/Bohr_radius)
angstrom = A = 0.529177210903
# e x a0 in D (https://en.wikipedia.org/wiki/Hartree_atomic_units)
Debye = D = 2.541746473


def ha2ev(E):
    '''Convert Hartree to electronvolt.'''
    return E * electronvolt


def ev2ha(E):
    '''Convert electronvolt to Hartree.'''
    return E / electronvolt


def ha2kcalmol(E):
    '''Convert Hartree to kcal/mol.'''
    return E * kcalmol


def kcalmol2ha(E):
    '''Convert kcal/mol to Hartree.'''
    return E / kcalmol


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
    return r / angstrom


def bohr2ang(r):
    '''Convert Bohr to Angstrom.'''
    return r * angstrom


def ebohr2d(p):
    '''Convert e x Bohr to Debye.'''
    return p * Debye


def d2ebohr(p):
    '''Convert Debye to e x Bohr.'''
    return p / Debye
