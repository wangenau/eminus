#!/usr/bin/env python3
'''Test operator identities.'''
from eminus import Atoms
import numpy as np
from numpy.random import randn
from numpy.testing import assert_allclose

# Create an Atoms object to build mock wave functions
atoms = Atoms('Ne', [0, 0, 0], ecut=1)
W_tests = {
    'full': randn(len(atoms.G2), atoms.Nstate),
    'active': randn(len(atoms.G2c), atoms.Nstate),
    'full_single': randn(len(atoms.G2)),
    'active_single': randn(len(atoms.G2c)),
    'full_spin': randn(atoms.Nspin, len(atoms.G2), atoms.Nstate),
    'active_spin': randn(atoms.Nspin, len(atoms.G2c), atoms.Nstate)
}


def run_operator(test):
    '''Run a given operator test.'''
    try:
        test()
    except Exception as err:
        print(f'Test for {test.__name__} failed.')
        raise SystemExit(err) from None
    else:
        print(f'Test for {test.__name__} passed.')
    return


def test_LinvL():
    for i in ['full', 'full_spin']:
        out = atoms.Linv(atoms.L(W_tests[i]))
        test = np.copy(W_tests[i])
        if test.ndim == 3:
            test[:, 0, :] = 0
        else:
            test[0, :] = 0
        assert_allclose(out, test)


def test_LLinv():
    for i in ['full', 'full_spin']:
        out = atoms.L(atoms.Linv(W_tests[i]))
        test = np.copy(W_tests[i])
        if test.ndim == 3:
            test[:, 0, :] = 0
        else:
            test[0, :] = 0
        assert_allclose(out, test)


def test_IJ():
    for i in ['full', 'full_single', 'full_spin']:
        out = atoms.I(atoms.J(W_tests[i]))
        test = W_tests[i]
        assert_allclose(out, test)


def test_JI():
    for i in ['full', 'active', 'full_single', 'active_single', 'full_spin', 'active_spin']:
        if 'active' in i:
            out = atoms.J(atoms.I(W_tests[i]), False)
        else:
            out = atoms.J(atoms.I(W_tests[i]))
        test = W_tests[i]
        assert_allclose(out, test)


def test_IdagJdag():
    for i in ['active', 'active_single', 'active_spin']:
        out = atoms.Idag(atoms.Jdag(W_tests[i]))
        test = W_tests[i]
        assert_allclose(out, test)


def test_JdagIdag():
    for i in ['full', 'full_single', 'full_spin']:
        out = atoms.Jdag(atoms.Idag(W_tests[i], True))
        test = W_tests[i]
        assert_allclose(out, test)


def test_TT():
    dr = randn(3)
    for i in ['active', 'active_single', 'active_spin']:
        out = atoms.T(atoms.T(W_tests[i], dr), -dr)
        test = W_tests[i]
        assert_allclose(out, test)


if __name__ == '__main__':
    import timeit
    start = timeit.default_timer()
    run_operator(test_LLinv)
    run_operator(test_LinvL)
    run_operator(test_IJ)
    run_operator(test_JI)
    run_operator(test_IdagJdag)
    run_operator(test_JdagIdag)
    run_operator(test_TT)
    end = timeit.default_timer()
    print(f'Test for operator identities passed in {end - start:.3f} s.')
