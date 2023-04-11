#!/usr/bin/env python3
'''Test exchange-correlation functionals.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus.xc import get_exc, get_vxc, get_xc, parse_functionals, XC_MAP

# Create random mock densities
# Use absolute values since eminus' functionals have no safety checks for simplicity and performance
rng = default_rng()
n_tests = {
    1: np.abs(rng.standard_normal((1, 10000))),
    2: np.abs(rng.standard_normal((2, 10000)))
}
functionals = [xc for xc in XC_MAP if xc.isdigit()]


@pytest.mark.extras
@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_exc(xc, Nspin):
    '''Compare internal functional energy densities to Libxc.'''
    from eminus.extras import libxc_functional
    n_spin = n_tests[Nspin]
    e_out = get_exc(xc, n_spin, Nspin)
    e_test, _ = libxc_functional(xc, n_spin, Nspin)
    assert_allclose(e_out, e_test)


@pytest.mark.extras
@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_vxc(xc, Nspin):
    '''Compare internal functional potentials to Libxc.'''
    from eminus.extras import libxc_functional
    n_spin = n_tests[Nspin]
    v_out = get_vxc(xc, n_spin, Nspin)
    _, v_test = libxc_functional(xc, n_spin, Nspin)
    assert_allclose(v_out, v_test)


@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_exc_only(xc, Nspin):
    '''Test the function to only get the exc part.'''
    n_spin = n_tests[Nspin]
    e_out, _ = get_xc(xc, n_spin, Nspin)
    e_test, v_test = get_xc(xc, n_spin, Nspin, exc_only=True)
    assert_allclose(e_out, e_test)
    assert_allclose(v_test, 0)


@pytest.mark.parametrize('xc,ref', [('svwn', ['lda_x', 'lda_c_vwn']),
                                    ('lda_x', ['lda_x', 'mock_xc']),
                                    ('s,pw', ['lda_x', 'lda_c_pw']),
                                    ('s', ['lda_x', 'mock_xc']),
                                    ('s,', ['lda_x', 'mock_xc']),
                                    ('pw', ['lda_c_pw', 'mock_xc']),
                                    (',pw', ['mock_xc', 'lda_c_pw']),
                                    ('', ['mock_xc', 'mock_xc']),
                                    (',', ['mock_xc', 'mock_xc']),
                                    ('libxc:1,libxc:7', ['libxc:1', 'libxc:7']),
                                    ('libxc:1,', ['libxc:1', 'mock_xc']),
                                    (',libxc:7', ['mock_xc', 'libxc:7']),
                                    ('s,libxc:7', ['lda_x', 'libxc:7'])])
def test_parse_functionals(xc, ref):
    '''Test the xc string parsing.'''
    f_x, f_c = parse_functionals(xc)
    assert f_x == ref[0]
    assert f_c == ref[1]


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
