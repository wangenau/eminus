#!/usr/bin/env python3
'''Test operator identities.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms

# Create an Atoms object to build mock wave functions
atoms = Atoms('Ne', [0, 0, 0], ecut=1).build()
rng = default_rng()
W_tests = {
    'full': rng.standard_normal((len(atoms.G2), atoms.Nstate)),
    'active': rng.standard_normal((len(atoms.G2c), atoms.Nstate)),
    'full_single': rng.standard_normal((len(atoms.G2))),
    'active_single': rng.standard_normal((len(atoms.G2c))),
    'full_spin': rng.standard_normal((atoms.Nspin, len(atoms.G2), atoms.Nstate)),
    'active_spin': rng.standard_normal((atoms.Nspin, len(atoms.G2c), atoms.Nstate))
}
dr = rng.standard_normal(3)


@pytest.mark.parametrize('type', ['full', 'full_spin'])
def test_LinvL(type):
    out = atoms.Linv(atoms.L(W_tests[type]))
    test = np.copy(W_tests[type])
    if test.ndim == 3:
        test[:, 0, :] = 0
    else:
        test[0, :] = 0
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'full_spin'])
def test_LLinv(type):
    out = atoms.L(atoms.Linv(W_tests[type]))
    test = np.copy(W_tests[type])
    if test.ndim == 3:
        test[:, 0, :] = 0
    else:
        test[0, :] = 0
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'full_single', 'full_spin'])
def test_IJ(type):
    out = atoms.I(atoms.J(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'active', 'full_single', 'active_single', 'full_spin',
                                  'active_spin'])
def test_JI(type):
    if 'active' in type:
        out = atoms.J(atoms.I(W_tests[type]), False)
    else:
        out = atoms.J(atoms.I(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['active', 'active_single', 'active_spin'])
def test_IdagJdag(type):
    out = atoms.Idag(atoms.Jdag(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'full_single', 'full_spin'])
def test_JdagIdag(type):
    out = atoms.Jdag(atoms.Idag(W_tests[type], True))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['active', 'active_single', 'active_spin'])
def test_TT(type):
    out = atoms.T(atoms.T(W_tests[type], dr), -dr)
    test = W_tests[type]
    assert_allclose(out, test)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
