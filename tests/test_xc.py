#!/usr/bin/env python3
'''Test exchange-correlation functionals.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus.xc import get_exc, get_vxc, get_xc, parse_functionals, parse_psp, XC_MAP

# Create random mock densities
# Use absolute values since eminus' functionals have no safety checks for simplicity and performance
rng = default_rng()
n_tests = {
    1: np.abs(rng.standard_normal((1, 10000))),
    2: np.abs(rng.standard_normal((2, 10000)))
}
functionals = [xc for xc in XC_MAP if xc.isdigit()]


@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_exc(xc, Nspin):
    '''Compare internal functional energy densities to Libxc.'''
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import libxc_functional
    from pyscf.dft.libxc import is_gga
    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = np.stack([n_spin, n_spin, n_spin], axis=2)
    e_out = get_exc(xc, n_spin, Nspin, dn_spin=dn_spin)
    e_test, _, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(e_out, e_test)


@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_vxc(xc, Nspin):
    '''Compare internal functional potentials to Libxc.'''
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import libxc_functional
    from pyscf.dft.libxc import is_gga
    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = np.stack([n_spin, n_spin, n_spin], axis=2)
    v_out, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    _, v_test, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(v_out, v_test)


@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_get_vsigmaxc(xc, Nspin):
    '''Compare internal functional vsigma to Libxc.'''
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from eminus.extras import libxc_functional
    from pyscf.dft.libxc import is_gga
    if not is_gga(xc):
        return
    n_spin = n_tests[Nspin]
    dn_spin = np.stack([n_spin, n_spin, n_spin], axis=2)
    _, vsigma_out = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    _, _, vsigma_test = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(vsigma_out, vsigma_test)


@pytest.mark.parametrize('xc', functionals)
@pytest.mark.parametrize('Nspin', [1, 2])
def test_exc_only(xc, Nspin):
    '''Test the function to only get the exc part.'''
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    from pyscf.dft.libxc import is_gga
    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = np.stack([n_spin, n_spin, n_spin], axis=2)
    e_out, _, _ = get_xc(xc, n_spin, Nspin, dn_spin=dn_spin)
    e_test, v_test, vsigma_test = get_xc(xc, n_spin, Nspin, exc_only=True, dn_spin=dn_spin)
    assert_allclose(e_out, e_test)
    assert v_test is None
    assert vsigma_test is None


@pytest.mark.parametrize('xc,ref', [('svwn', ['lda_x', 'lda_c_vwn']),
                                    ('lda_x', ['lda_x', 'mock_xc']),
                                    ('s,pw', ['lda_x', 'lda_c_pw']),
                                    ('s', ['lda_x', 'mock_xc']),
                                    ('s,', ['lda_x', 'mock_xc']),
                                    ('pw', ['lda_c_pw', 'mock_xc']),
                                    (',pw', ['mock_xc', 'lda_c_pw']),
                                    ('', ['mock_xc', 'mock_xc']),
                                    (',', ['mock_xc', 'mock_xc']),
                                    ('libxc:1,l:7', ['libxc:1', 'l:7']),
                                    ('libxc:1,', ['libxc:1', 'mock_xc']),
                                    (',libxc:7', ['mock_xc', 'libxc:7']),
                                    ('s,l:7', ['lda_x', 'l:7'])])
def test_parse_functionals(xc, ref):
    '''Test the xc string parsing.'''
    f_x, f_c = parse_functionals(xc)
    assert f_x == ref[0]
    assert f_c == ref[1]


@pytest.mark.parametrize('xc,ref', [(['lda_x', 'lda_c_vwn'], 'pade'),
                                    (['gga_x_pbe', 'gga_c_pbe'], 'pbe'),
                                    (['lda_x', 'gga_c_pbe'], 'pbe'),
                                    (['gga_x_pbe', 'lda_c_vwn'], 'pbe'),
                                    (['l:1', 'l:7'], 'pade'),
                                    (['libxc:gga_x_pbe', 'l:gga_c_pbe'], 'pbe'),
                                    (['libxc:1', 'gga_c_pbe'], 'pbe'),
                                    (['libxc:gga_x_pbe', 'lda_c_vwn'], 'pbe'),
                                    (['gga_x_pbe', 'libxc:7'], 'pbe'),])
def test_parse_psp(xc, ref):
    '''Test the pseudopotential parsing.'''
    if ':' in ''.join(xc):
        pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    psp = parse_psp(xc)
    assert psp == ref


def test_libxc_str():
    '''Test that strings that start with libxc get properly parsed.'''
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    n_spin = n_tests[1]
    e_out, v_out, _ = get_xc('1,7', n_spin, 1)
    e_test, v_test, _ = get_xc('l:1,l:7', n_spin, 1)
    assert_allclose(e_out, e_test)
    assert_allclose(v_out, v_test)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
