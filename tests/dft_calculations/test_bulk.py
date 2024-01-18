#!/usr/bin/env python3
"""Test total energies for bulk silicon for different functionals."""
import inspect
import pathlib

from numpy.testing import assert_allclose
import pytest

from eminus import Cell, RSCF, USCF

# Total energies from a spin-paired calculation with PWDFT.jl with the same parameters as below
# PWDFT.jl does not support spin-polarized calculations with SCAN
E_ref = {
    'SVWN': -7.785143,
    'PBE': -7.726629,
    ':MGGA_X_SCAN,:MGGA_C_SCAN': -7.729585
}

file_path = pathlib.Path(inspect.stack()[0][1]).parent
a = 10.2631
ecut = 5
s = 15
kmesh = 2
guess = 'random'
etol = 1e-6
opt = {'sd': 4, 'pccg': 27}
betat = 1e-3


@pytest.mark.slow()
@pytest.mark.parametrize('xc', E_ref.keys())
def test_polarized(xc):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    cell = Cell('Si', 'Diamond', ecut=ecut, a=a, kmesh=kmesh)
    cell.s = s
    E = USCF(cell, xc=xc, guess=guess, etol=etol, opt=opt).run(betat=betat)
    assert_allclose(E, E_ref[xc], rtol=etol)  # Use rtol over atol so SCAN can pass the test


@pytest.mark.parametrize('xc', E_ref.keys())
def test_unpolarized(xc):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    cell = Cell('Si', 'Diamond', ecut=ecut, a=a, kmesh=kmesh)
    cell.s = s
    E = RSCF(cell, xc=xc, guess=guess, etol=etol, opt=opt).run(betat=betat)
    assert_allclose(E, E_ref[xc], rtol=etol)  # Use rtol over atol so SCAN can pass the test


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
