#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Collection of constants and unit conversion functions.

For more about atomic units, see: https://en.wikipedia.org/wiki/Hartree_atomic_units
All values are directly calculated from the CODATA 2018 constants as found in SciPy.
"""

import numpy as np

#: Hartree in electronvolt.
# scipy.constants.value('Hartree energy in eV')
electronvolt = eV = 27.211386245988
#: Hartree in kcal per mol.
# scipy.constants.value('Hartree energy') * scipy.constants.value('Avogadro constant') / 4.184 / 1e3
kcalmol = 627.5094740630557
#: Bohr radius in Angstrom.
# scipy.constants.value('Bohr radius') * 1e10
Angstrom = A = 0.529177210903
#: Elementary charge times bohr radius in Debye.
# scipy.constants.value('atomic unit of electric dipole mom.') / \
# scipy.constants.value('hertz-inverse meter relationship') * 1e21
Debye = D = 2.5417464731818566
#: Hartree in kb times Kelvin.
# scipy.constants.value('kelvin-hartree relationship')
Kelvin = K = 3.1668115634556e-06


def ha2ev(E):
    """Convert Hartree to electronvolt.

    Args:
        E (float | ndarray): Energy in Hartree.

    Returns:
        float | ndarray: Energy in electronvolt.
    """
    return E * electronvolt


def ev2ha(E):
    """Convert electronvolt to Hartree.

    Args:
        E (float | ndarray): Energy in electronvolt.

    Returns:
        float | ndarray: Energy in Hartree.
    """
    return E / electronvolt


def ha2kcalmol(E):
    """Convert Hartree to kcal/mol.

    Args:
        E (float | ndarray): Energy in Hartree.

    Returns:
        float | ndarray: Energy in kcal/mol.
    """
    return E * kcalmol


def kcalmol2ha(E):
    """Convert kcal/mol to Hartree.

    Args:
        E (float | ndarray): Energy in kcal/mol.

    Returns:
        float | ndarray: Energy in Hartree.
    """
    return E / kcalmol


def ha2ry(E):
    """Convert Hartree to Rydberg.

    Args:
        E (float | ndarray): Energy in Hartree.

    Returns:
        float | ndarray: Energy in Rydberg.
    """
    return 2 * E


def ry2ha(E):
    """Convert Rydberg to Hartree.

    Args:
        E (float | ndarray): Energy in Rydberg.

    Returns:
        float | ndarray: Energy in Hartree.
    """
    return E / 2


def ha2kelvin(E):
    """Convert Hartree to Kelvin.

    Args:
        E (float | ndarray): Energy in Hartree.

    Returns:
        float | ndarray: Temperature in Kelvin.
    """
    return E / Kelvin


def kelvin2ha(T):
    """Convert Kelvin to Hartree.

    Args:
        T (float | ndarray): Temperature in Kelvin.

    Returns:
        float | ndarray: Energy in Hartree.
    """
    return T * Kelvin


def ang2bohr(r):
    """Convert Angstrom to Bohr.

    Args:
        r (float | ndarray): Length in Angstrom.

    Returns:
        float | ndarray: Length in Bohr.
    """
    return r / Angstrom


def bohr2ang(r):
    """Convert Bohr to Angstrom.

    Args:
        r (float | ndarray): Length in Bohr.

    Returns:
        float | ndarray: Length in Angstrom.
    """
    return r * Angstrom


def ebohr2d(p):
    """Convert e * Bohr to Debye.

    Args:
        p (float | ndarray): Electric dipole moment in e * Bohr.

    Returns:
        float | ndarray: Electric dipole moment in Debye.
    """
    return p * Debye


def d2ebohr(p):
    """Convert Debye to e * Bohr.

    Args:
        p (float | ndarray): Electric dipole moment in Debye.

    Returns:
        float | ndarray: Electric dipole moment in e * Bohr.
    """
    return p / Debye


def rad2deg(a):
    """Convert Radians to Degree.

    Args:
        a (float | ndarray): Angle in Radians.

    Returns:
        float | ndarray: Angle in Degree.
    """
    return a * 180 / np.pi


def deg2rad(a):
    """Convert Degree to Radians.

    Args:
        a (float | ndarray): Angle in Degree.

    Returns:
        float | ndarray: Angle in Radians.
    """
    return a * np.pi / 180
