#!/usr/bin/env python3
'''Test torch extra.'''
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, config
config.use_torch = True

# Create an Atoms object to build mock wave functions
atoms = Atoms('Ne', (0, 0, 0), ecut=1).build()
rng = default_rng()
W_tests = {
    'full': rng.standard_normal((len(atoms.G2), atoms.Nstate)),
    'active': rng.standard_normal((len(atoms.G2c), atoms.Nstate)),
    'full_single': rng.standard_normal(len(atoms.G2)),
    'active_single': rng.standard_normal(len(atoms.G2c)),
    'full_spin': rng.standard_normal((atoms.Nspin, len(atoms.G2), atoms.Nstate)),
    'active_spin': rng.standard_normal((atoms.Nspin, len(atoms.G2c), atoms.Nstate))
}


@pytest.mark.parametrize('type', ['full', 'full_single', 'full_spin'])
def test_IJ(type):
    '''Test forward and backward operator identity.'''
    pytest.importorskip('torch', reason='torch not installed, skip tests')
    out = atoms.I(atoms.J(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'active', 'full_single', 'active_single', 'full_spin',
                                  'active_spin'])
def test_JI(type):
    '''Test forward and backward operator identity.'''
    pytest.importorskip('torch', reason='torch not installed, skip tests')
    if 'active' in type:
        out = atoms.J(atoms.I(W_tests[type]), False)
    else:
        out = atoms.J(atoms.I(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['active', 'active_single', 'active_spin'])
def test_IdagJdag(type):
    '''Test daggered forward and backward operator identity.'''
    pytest.importorskip('torch', reason='torch not installed, skip tests')
    out = atoms.Idag(atoms.Jdag(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'full_single', 'full_spin'])
def test_JdagIdag(type):
    '''Test daggered forward and backward operator identity.'''
    pytest.importorskip('torch', reason='torch not installed, skip tests')
    out = atoms.Jdag(atoms.Idag(W_tests[type], True))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full_single'])
def test_hermitian_I(type):
    '''Test that I and Idag operators are hermitian.'''
    pytest.importorskip('torch', reason='torch not installed, skip tests')
    a = W_tests[type]
    b = W_tests[type] + rng.standard_normal(1)
    out = (a.conj().T @ atoms.I(b)).conj()
    test = b.conj().T @ atoms.Idag(a, True)
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full_single'])
def test_hermitian_J(type):
    '''Test that J and Jdag operators are hermitian.'''
    pytest.importorskip('torch', reason='torch not installed, skip tests')
    a = W_tests[type]
    b = W_tests[type] + rng.standard_normal(1)
    out = (a.conj().T @ atoms.J(b)).conj()
    test = b.conj().T @ atoms.Jdag(a)
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'full_single', 'full_spin'])
def test_IJ_gpu(type):
    '''Test forward and backward GPU operator identity.'''
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip('GPU not available, skip tests')
    out = atoms.I(atoms.J(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'active', 'full_single', 'active_single', 'full_spin',
                                  'active_spin'])
def test_JI_gpu(type):
    '''Test forward and backward GPU operator identity.'''
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip('GPU not available, skip tests')
    if 'active' in type:
        out = atoms.J(atoms.I(W_tests[type]), False)
    else:
        out = atoms.J(atoms.I(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['active', 'active_single', 'active_spin'])
def test_IdagJdag_gpu(type):
    '''Test daggered forward and backward GPU operator identity.'''
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip('GPU not available, skip tests')
    out = atoms.Idag(atoms.Jdag(W_tests[type]))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full', 'full_single', 'full_spin'])
def test_JdagIdag_gpu(type):
    '''Test daggered forward and backward GPU operator identity.'''
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip('GPU not available, skip tests')
    out = atoms.Jdag(atoms.Idag(W_tests[type], True))
    test = W_tests[type]
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full_single'])
def test_hermitian_I_gpu(type):
    '''Test that I and Idag GPU operators are hermitian.'''
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip('GPU not available, skip tests')
    a = W_tests[type]
    b = W_tests[type] + rng.standard_normal(1)
    out = (a.conj().T @ atoms.I(b)).conj()
    test = b.conj().T @ atoms.Idag(a, True)
    assert_allclose(out, test)


@pytest.mark.parametrize('type', ['full_single'])
def test_hermitian_J_gpu(type):
    '''Test that J and Jdag GPU operators are hermitian.'''
    try:
        config.use_gpu = True
        assert config.use_gpu
    except AssertionError:
        pytest.skip('GPU not available, skip tests')
    a = W_tests[type]
    b = W_tests[type] + rng.standard_normal(1)
    out = (a.conj().T @ atoms.J(b)).conj()
    test = b.conj().T @ atoms.Jdag(a)
    assert_allclose(out, test)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
