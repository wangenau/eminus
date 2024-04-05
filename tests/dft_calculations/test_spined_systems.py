#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test total energies of helium atoms with different spin states."""

from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, USCF

# Total energies from a spin-polarized calculation with JDFTx with (almost) same parameters as below
E_ref = {
    1: -2.27454774,
    2: -1.95070509,
}

a = 10
ecut = 10
s = 32
xc = 'svwn'
guess = 'random'
etol = 1e-6
opt = {'auto': 22}


@pytest.mark.parametrize('spin', [1, 2])
def test_polarized(spin):
    """Compare total energies for a test system with a reference value (spin-polarized)."""
    atoms = Atoms('He', (0, 0, 0), a=a, ecut=ecut, spin=spin)
    atoms.s = s
    E = USCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[spin], rtol=etol)  # Use rtol for a looser comparison with JDFTx


if __name__ == '__main__':
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
