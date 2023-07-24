#!/usr/bin/env python3
"""Test total energies using SCAN for a small set of spin-paired systems."""
import inspect
import pathlib

from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, read_xyz, RSCF, USCF

# Total energies from a spin-paired calculation with PWDFT.jl with same parameters as below
# PWDFT.jl does not support spin-polarized calculations with SCAN
# Only test methane for faster tests, also SCAN can easily run into convergence issues
E_ref = {
    'CH4': -7.75275
}

file_path = pathlib.Path(inspect.stack()[0][1]).parent
a = 10
ecut = 10
s = 30
xc = ':MGGA_X_SCAN,:MGGA_C_SCAN'
guess = 'random'
etol = 1e-6
opt = {'sd': 3, 'pccg': 16}


@pytest.mark.parametrize('system', E_ref.keys())
def test_polarized(system):
    """Compare total energies for a test system with a reference value (spin-polarized)."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    atom, X = read_xyz(str(file_path.joinpath(f'{system}.xyz')))
    atoms = Atoms(atom, X, a=a, ecut=ecut, Z='pbe', s=s)
    E = USCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[system], atol=etol)


@pytest.mark.parametrize('system', E_ref.keys())
def test_unpolarized(system):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    atom, X = read_xyz(str(file_path.joinpath(f'{system}.xyz')))
    atoms = Atoms(atom, X, a=a, ecut=ecut, Z='pbe', s=s)
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[system], atol=etol)


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
