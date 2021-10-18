#!/usr/bin/env python3
'''
Collection of constants and unit conversion tools.
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
    '''Convert Hartree to electronvolt.

    Args:
        E : float
            Energy in Hartree.

    Returns:
        Energy in electronvolt as a float.
    '''
    return E * electronvolt


def ev2ha(E):
    '''Convert electronvolt to Hartree.

    Args:
        E : float
            Energy in electronvolt.

    Returns:
        Energy in Hartree as a float.
    '''
    return E / electronvolt


def ha2kcalmol(E):
    '''Convert Hartree to kcal/mol.

    Args:
        E : float
            Energy in Hartree.

    Returns:
        Energy in kcal/mol as a float.
    '''
    return E * kcalmol


def kcalmol2ha(E):
    '''Convert kcal/mol to Hartree.

    Args:
        E : float
            Energy in kcal/mol.

    Returns:
        Energy in Hartree as a float.
    '''
    return E / kcalmol


def ev2kcalmol(E):
    '''Convert electronvolt to kcal/mol.

    Args:
        E : float
            Energy in electronvolt.

    Returns:
        Energy in kcal/mol as a float.
    '''
    return ha2kcalmol(ev2ha(E))


def kcalmol2ev(E):
    '''Convert kcal/mol to electronvolt.

    Args:
        E : float
            Energy in kcal/mol.

    Returns:
        Energy in electronvolt as a float.
    '''
    return ha2ev(kcalmol2ha(E))


def ha2ry(E):
    '''Convert Hartree to Rydberg.

    Args:
        E : float
            Energy in Hartree.

    Returns:
        Energy in Rydberg as a float.
    '''
    return 2 * E


def ry2ha(E):
    '''Convert Rydberg to Hartree.

    Args:
        E : float
            Energy in Rydberg.

    Returns:
        Energy in Hartree as a float.
    '''
    return E / 2


def ang2bohr(r):
    '''Convert Angstrom to Bohr.

    Args:
        r : float
            Length in Angstrom.

    Returns:
        Length in Bohr as a float.
    '''
    return r / angstrom


def bohr2ang(r):
    '''Convert Bohr to Angstrom.

    Args:
        r : float
            Length in Bohr.

    Returns:
        Length in Angstrom as a float.
    '''
    return r * angstrom


def ebohr2d(p):
    '''Convert e x Bohr to Debye.

    Args:
        p : float
            Electric dipole moment in e times Bohr.

    Returns:
        Electric dipole moment in Debye as a float.
    '''
    return p * Debye


def d2ebohr(p):
    '''Convert Debye to e x Bohr.

    Args:
        p : float
            Electric dipole moment in Debye.

    Returns:
        Electric dipole moment in e times Bohr as a float.
    '''
    return p / Debye
