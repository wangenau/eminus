#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test energies for the implemented toy potentials."""

from numpy.testing import assert_allclose

from eminus import Atoms, RSCF
from eminus.dft import get_epsilon
from eminus.units import ang2bohr


def test_harmonic():
    """Compare total energies for the harmonic potential."""
    atom = 'H'
    pos = (0, 0, 0)
    a = 6
    ecut = 10
    s = (20, 25, 30)
    f = (2, 2, 2, 2)
    pot = 'harmonic'

    atoms = Atoms(atom, pos, a=a, ecut=ecut)
    atoms.s = s
    atoms.f = f
    E = RSCF(atoms, pot=pot, guess='random', etol=1e-6).run()
    # We have to get close to the Etot reference value of 43.337 Eh (for different parameters)
    assert_allclose(E, 43.10344)


def test_coulomb():
    """Compare total energies for the ionic potential."""
    atom = 'H'
    pos = (0, 0, 0)
    a = (9, 10, 11)
    ecut = 10
    s = 20
    pot = 'coulomb'

    atoms = Atoms(atom, pos, a=a, ecut=ecut)
    atoms.s = s
    E = RSCF(atoms, pot=pot, guess='random', etol=1e-6).run()
    # In the limit we should come close to the NIST Etot value of -0.445671 Eh
    # See https://nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electronic-7-0
    assert_allclose(E, -0.4393708)


def test_ge():
    """Compare eigenstate energies for the germanium potential."""
    atom = 'Ge'
    pos = (0, 0, 0)
    a = ang2bohr(5.66)
    ecut = 10
    s = 10
    f = (2, 2 / 3, 2 / 3, 2 / 3)
    pot = 'ge'

    atoms = Atoms(atom, pos, a=a, ecut=ecut)
    atoms.s = s
    atoms.f = f
    scf = RSCF(atoms, pot=pot, guess='random', etol=1e-6)
    scf.run()
    eps = get_epsilon(scf, scf.W)[0]
    # In the limit we should come close to the NIST 4s-4p value of -0.276641 Eh
    # See https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electronic-7-31
    assert_allclose(eps[0] - eps[1:], -0.272106, atol=1e-5)


if __name__ == '__main__':
    import inspect
    import pathlib

    import pytest

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
