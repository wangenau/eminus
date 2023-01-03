#!/usr/bin/env python3
'''Test exchange-correlation functionals.'''
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
import pytest

from eminus.xc import get_xc

# Create random mock densities
# Use absolute values since eminus' functionals have no safety checks for simplicity and performance
rng = default_rng()
n_tests = {
    1: np.abs(rng.standard_normal((1, 10000))),
    2: np.abs(rng.standard_normal((2, 10000)))
}


@pytest.mark.extras
@pytest.mark.parametrize('xc', ['1', '7', '12', '287'])
@pytest.mark.parametrize('Nspin', [1, 2])
def test_libxc_functional_exc(xc, Nspin):
    '''Compare internal functional energy densities to LibXC.'''
    from eminus.extras import libxc_functional
    n_spin = n_tests[Nspin]
    e_out, _ = get_xc(',' + xc, n_spin, Nspin)
    e_test, _ = libxc_functional(xc, n_spin, Nspin)
    assert_allclose(e_out, e_test)


@pytest.mark.extras
@pytest.mark.parametrize('xc', ['1', '7', '12', '287'])
@pytest.mark.parametrize('Nspin', [1, 2])
def test_libxc_functional_vxc(xc, Nspin):
    '''Compare internal functional poetntials to LibXC.'''
    from eminus.extras import libxc_functional
    n_spin = n_tests[Nspin]
    _, v_out = get_xc(',' + xc, n_spin, Nspin)
    _, v_test = libxc_functional(xc, n_spin, Nspin)
    assert_allclose(v_out, v_test)


if __name__ == '__main__':
    import inspect
    import pathlib
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
