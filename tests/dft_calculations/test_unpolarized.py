#!/usr/bin/env python3
'''Test total energies for a small set of spin-paired systems.'''
import inspect
import pathlib

from numpy.testing import assert_allclose

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


def calc_unpolarized(system):
    '''Compare total energies for a test system with a reference value (spin-paired).'''
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    a = 10
    ecut = 10
    s = 30
    xc = 'lda,vwn'
    guess = 'random'
    etol = 1e-6
    min = {'sd': 15, 'pccg': 25}

    atom, X = read_xyz(str(file_path.joinpath(f'{system}.xyz')))
    atoms = Atoms(atom, X, a=a, ecut=ecut, s=s, verbose='warning')
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, min=min).run()

    try:
        assert_allclose(E, E_ref[system], atol=etol)
    except AssertionError as err:
        print(f'Test for {system} failed.')
        raise SystemExit(err) from None
    else:
        print(f'Test for {system} passed.')
    return


def test_H2():
    calc_unpolarized('H2')


def test_He():
    calc_unpolarized('He')


def test_LiH():
    calc_unpolarized('LiH')


def test_CH4():
    calc_unpolarized('CH4')


def test_Ne():
    calc_unpolarized('Ne')


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    test_H2()
    test_He()
    test_LiH()
    test_CH4()
    test_Ne()
    end = time.perf_counter()
    print(f'Test for unpolarized calculations passed in {end - start:.3f} s.')
