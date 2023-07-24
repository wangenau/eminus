#!/usr/bin/env python3
"""Test total energies using VWN for a small set of spin-paired systems."""
import inspect
import pathlib

from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, read_xyz, RSCF

# Total energies from a spin-polarized calculation with PWDFT.jl with same parameters as below
# Closed-shell systems have the same energy for spin-paired and -polarized calculations
E_ref = {
    'H2': -1.103621,
    'He': -2.542731,
    'LiH': -0.793497,
    'CH4': -7.699509,
    'Ne': -29.876365
}


@pytest.mark.parametrize('system', E_ref.keys())
def test_unpolarized(system):
    """Compare total energies for a test system with a reference value (spin-paired)."""
    file_path = pathlib.Path(inspect.stack()[0][1]).parent
    a = 10
    ecut = 10
    s = 30
    xc = 'svwn'
    guess = 'random'
    etol = 1e-6
    opt = {'auto': 20}

    atom, X = read_xyz(str(file_path.joinpath(f'{system}.xyz')))
    atoms = Atoms(atom, X, a=a, ecut=ecut, Z='pade', s=s)
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, opt=opt).run()
    assert_allclose(E, E_ref[system], atol=etol)


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
