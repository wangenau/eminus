# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test total energies of charged helium atoms."""

from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, RSCF, USCF

# Total energies from a spin-polarized calculation with JDFTx with (almost) same parameters as below
E_ref = {
    2: -0.5674595,
    1: -1.86872082,
    -1: -2.63097979,
    -2: -2.74473214,
}

a = 10
ecut = 10
s = 32
xc = 'svwn'
guess = 'random'
etol = 1e-6
opt = {'auto': 19}


@pytest.mark.parametrize('charge', [1, -1])
def test_polarized(charge):
    """Compare total energies for a test system with a reference value (spin-polarized)."""
    atoms = Atoms('He', (0, 0, 0), a=a, ecut=ecut, charge=charge)
    atoms.s = s
    E = USCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[charge], rtol=etol)  # Use rtol for a looser comparison with JDFTx


@pytest.mark.parametrize('charge', [2, -2])
def test_unpolarized(charge):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    atoms = Atoms('He', (0, 0, 0), a=a, ecut=ecut, charge=charge)
    atoms.s = s
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[charge], rtol=etol)  # Use rtol for a looser comparison with JDFTx


if __name__ == '__main__':
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
