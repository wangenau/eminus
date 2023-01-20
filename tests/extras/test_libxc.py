#!/usr/bin/env python3
'''Test libxc extra.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus.xc import get_xc, parse_functionals

# Create random mock densities
# Use absolute values since eminus' functionals have no safety checks for simplicity and performance
rng = default_rng()
n_tests = {
    1: np.abs(rng.standard_normal((1, 10000))),
    2: np.abs(rng.standard_normal((2, 10000)))
}


@pytest.mark.extras
@pytest.mark.parametrize('xc', ['1', '7'])
@pytest.mark.parametrize('Nspin', [1, 2])
def test_libxc_functional_exc(xc, Nspin):
    '''Compare LibXC functional energy densities to internal functionals.'''
    from pylibxc import LibXCFunctional  # noqa: F401
    from eminus.extras import libxc_functional
    n_spin = n_tests[Nspin]
    e_out, _ = libxc_functional(xc, n_spin, Nspin)
    e_test, _ = get_xc(parse_functionals(xc), n_spin, Nspin)
    assert_allclose(e_out, e_test)


@pytest.mark.extras
@pytest.mark.parametrize('xc', ['1', '7'])
@pytest.mark.parametrize('Nspin', [1, 2])
def test_libxc_functional_vxc(xc, Nspin):
    '''Compare LibXC functional potentials to internal functionals.'''
    from pylibxc import LibXCFunctional  # noqa: F401
    from eminus.extras import libxc_functional
    n_spin = n_tests[Nspin]
    _, v_out = libxc_functional(xc, n_spin, Nspin)
    _, v_test = get_xc(parse_functionals(xc), n_spin, Nspin)
    assert_allclose(v_out, v_test)


@pytest.mark.extras
@pytest.mark.parametrize('xc', ['1', '7'])
@pytest.mark.parametrize('Nspin', [1, 2])
def test_pyscf_functional_exc(xc, Nspin):
    '''Compare LibXC functional energy densities as implemented in PySCF to internal functionals.'''
    from eminus.extras.libxc import pyscf_functional
    n_spin = n_tests[Nspin]
    e_out, _ = pyscf_functional(xc, n_spin, Nspin)
    e_test, _ = get_xc(parse_functionals(xc), n_spin, Nspin)
    assert_allclose(e_out, e_test)


@pytest.mark.extras
@pytest.mark.parametrize('xc', ['1', '7'])
@pytest.mark.parametrize('Nspin', [1, 2])
def test_pyscf_functional_vxc(xc, Nspin):
    '''Compare LibXC functional potentials as implemented in PySCF to internal functionals.'''
    from eminus.extras.libxc import pyscf_functional
    n_spin = n_tests[Nspin]
    _, v_out = pyscf_functional(xc, n_spin, Nspin)
    _, v_test = get_xc(parse_functionals(xc), n_spin, Nspin)
    assert_allclose(v_out, v_test)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
