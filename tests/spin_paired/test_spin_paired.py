from numpy.testing import assert_allclose

from plainedft import Atoms, SCF, read_xyz, __path__

# Total energies calculated with PWDFT.jl for He, H2, LiH, CH4, and Ne with same parameters as below
# These values can be generated with the file ref_spin_paired.jl
Etot_ref = [-2.54356557, -1.10228799, -0.76526088, -7.71058154, -29.88936934]


def calc_spin_paired(system, E_ref):
    '''Compare total energies for a test system with a reference value.'''
    path = f'{__path__[0]}/../tests/spin_paired'
    a = 16
    ecut = 10
    S = 48
    verbose = 0

    atom, X = read_xyz(f'{path}/{system}.xyz')
    atoms = Atoms(atom=atom, X=X, a=a, ecut=ecut, S=S, verbose=verbose)
    E = SCF(atoms)

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
