#!/usr/bin/env python3
'''Test total energies for a small set of systems.'''
import eminus
from eminus import Atoms, read_xyz, SCF
from numpy.testing import assert_allclose

# Total energies calculated with PWDFT.jl for He, H2, LiH, CH4, and Ne with same parameters as below
Etot_ref = [-2.54356557, -1.10228799, -0.76598438, -7.70803736, -29.88936935]


def calc_spin_paired(system, E_ref):
    '''Compare total energies for a test system with a reference value.'''
    path = f'{eminus.__path__[0]}/../tests/spin_paired'
    a = 16
    ecut = 10
    s = 48
    verbose = 0

    atom, X = read_xyz(f'{path}/{system}.xyz')
    atoms = Atoms(atom=atom, X=X, a=a, ecut=ecut, s=s, verbose=verbose)
    E = SCF(atoms).run()

    try:
        assert_allclose(E, E_ref)
    except AssertionError as err:
        print(f'Test for {system} failed.')
        raise SystemExit(err) from None
    else:
        print(f'Test for {system} passed.')
    return


def test_He():
    calc_spin_paired('He', Etot_ref[0])


def test_H2():
    calc_spin_paired('H2', Etot_ref[1])


def test_LiH():
    calc_spin_paired('LiH', Etot_ref[2])


def test_CH4():
    calc_spin_paired('CH4', Etot_ref[3])


def test_Ne():
    calc_spin_paired('Ne', Etot_ref[4])


if __name__ == '__main__':
    test_He()
    test_H2()
    test_LiH()
    test_CH4()
    test_Ne()
