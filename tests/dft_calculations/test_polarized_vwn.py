#!/usr/bin/env python3
'''Test total energies using VWN for a small set of spin-polarized and -unpolarized systems.'''
import inspect
import pathlib

from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, read_xyz, USCF

# Total energies from a spin-polarized calculation with PWDFT.jl with same parameters as below
E_ref = {
    'H': -0.469823,
    'H2': -1.103621,
    'He': -2.542731,
    'Li': -0.243013,
    'LiH': -0.793497,
    'CH4': -7.699509,
    'Ne': -29.876365
}


@pytest.mark.parametrize('system', E_ref.keys())
def test_polarized(system):
    '''Compare total energies for a test system with a reference value (spin-polarized).'''
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    a = 10
    ecut = 10
    s = 30
    xc = 'svwn'
    guess = 'random'
    etol = 1e-6
    min = {'sd': 15, 'pccg': 23}

    atom, X = read_xyz(str(file_path.joinpath(f'{system}.xyz')))
    atoms = Atoms(atom, X, a=a, ecut=ecut, Z='pade', s=s)
    E = USCF(atoms, xc=xc, guess=guess, etol=etol, min=min).run()
    assert_allclose(E, E_ref[system], atol=etol)


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
