#!/usr/bin/env python3
'''Test total energies for a small set of spin-polarized and -unpolarized systems.'''
import inspect
import os

from numpy.testing import assert_allclose

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


def calc_polarized(system):
    '''Compare total energies for a test system with a reference value (spin-polarized).'''
    file_path = inspect.getfile(inspect.currentframe())
    a = 10
    ecut = 10
    s = 30
    xc = 'LDA,VWN'
    guess = 'random'
    etol = 1e-6

    atom, X = read_xyz(f'{os.path.dirname(file_path)}/{system}.xyz')
    atoms = Atoms(atom, X, a=a, ecut=ecut, s=s, verbose='warning')
    E = USCF(atoms, xc=xc, guess=guess, etol=etol).run()

    try:
        assert_allclose(E, E_ref[system], atol=etol)
    except AssertionError as err:
        print(f'Test for {system} failed.')
        raise SystemExit(err) from None
    else:
        print(f'Test for {system} passed.')
    return


def test_H():
    calc_polarized('H')


def test_H2():
    calc_polarized('H2')


def test_He():
    calc_polarized('He')


def test_Li():
    calc_polarized('Li')


def test_LiH():
    calc_polarized('LiH')


def test_CH4():
    calc_polarized('CH4')


def test_Ne():
    calc_polarized('Ne')


if __name__ == '__main__':
    import timeit
    start = timeit.default_timer()
    test_H()
    test_H2()
    test_He()
    test_Li()
    test_LiH()
    test_CH4()
    test_Ne()
    end = timeit.default_timer()
    print(f'Test for polarized calculations passed in {end - start:.3f} s.')
