#!/usr/bin/env python3
'''Test operator identities.'''
from eminus import Atoms
import numpy as np
from numpy.random import randn
from numpy.testing import assert_allclose

# Create an Atoms object to build mock wave functions
atoms = Atoms('Ne', [0, 0, 0])
W_tests = {
    'full': randn(len(atoms.G2), atoms.Ns),
    'active': randn(len(atoms.G2c), atoms.Ns),
    'full_single': randn(len(atoms.G2)),
    'active_single': randn(len(atoms.G2c))
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
    for i in ['full']:
        out = atoms.Linv(atoms.L(W_tests[i]))
        test = np.copy(W_tests[i])
        test[0, :] = 0
        assert_allclose(out, test)


def test_LLinv():
    for i in ['full']:
        out = atoms.L(atoms.Linv(W_tests[i]))
        test = np.copy(W_tests[i])
        test[0, :] = 0
        assert_allclose(out, test)


def test_IJ():
    for i in ['full', 'full_single']:
        out = atoms.I(atoms.J(W_tests[i]))
        test = W_tests[i].reshape(out.shape)
        assert_allclose(out, test)


def test_JI():
    for i in ['full', 'active', 'full_single', 'active_single']:
        if 'active' in i:
            out = atoms.J(atoms.I(W_tests[i]), False)
        else:
            out = atoms.J(atoms.I(W_tests[i]))
        test = W_tests[i].reshape(out.shape)
        assert_allclose(out, test)


def test_TT():
    dr = randn(3)
    for i in ['active']:
        out = atoms.T(atoms.T(W_tests[i], dr), -dr)
        test = W_tests[i]
        assert_allclose(out, test)


if __name__ == '__main__':
    run_operator(test_LLinv)
    run_operator(test_LinvL)
    run_operator(test_IJ)
    run_operator(test_JI)
    run_operator(test_TT)
