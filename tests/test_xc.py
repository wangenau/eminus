# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="no-redef"
"""Test exchange-correlation functionals."""

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from eminus.xc import get_exc, get_vxc, get_xc, get_zeta, IMPLEMENTED, XC_MAP

# Create random mock densities
# Use absolute values since eminus' functionals have no safety checks for simplicity and performance
rng = default_rng()
n_tests = {
    1: np.abs(rng.standard_normal((1, 10000))),
    2: np.abs(rng.standard_normal((2, 10000))),
}
functionals = {xc for xc in XC_MAP if xc.isdigit()}
excludelist = {
    "577",  # GDSMFB has inconsistencies in the Libxc implementation
}
functionals -= excludelist


@pytest.mark.parametrize("xc", functionals)
@pytest.mark.parametrize("Nspin", [1, 2])
def test_get_exc(xc, Nspin):
    """Compare internal functional energy densities to Libxc."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras import libxc_functional

    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = np.stack((n_spin, n_spin, n_spin), axis=2)  # type: np.typing.NDArray[np.floating]
    e_out = get_exc(xc, n_spin, Nspin, dn_spin=dn_spin)
    e_test, _, _, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(e_out, e_test)


@pytest.mark.parametrize("xc", functionals)
@pytest.mark.parametrize("Nspin", [1, 2])
def test_get_vxc(xc, Nspin):
    """Compare internal functional potentials to Libxc."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras import libxc_functional

    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = np.stack((n_spin, n_spin, n_spin), axis=2)  # type: np.typing.NDArray[np.floating]
    v_out, _, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    _, v_test, _, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(v_out, v_test)


@pytest.mark.parametrize("xc", functionals)
@pytest.mark.parametrize("Nspin", [1, 2])
def test_get_vsigmaxc(xc, Nspin):
    """Compare internal functional vsigma to Libxc."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras import libxc_functional

    if not is_gga(xc):
        return
    n_spin = n_tests[Nspin]
    dn_spin = np.stack((n_spin, n_spin, n_spin), axis=2)
    _, vsigma_out, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    _, _, vsigma_test, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(vsigma_out, vsigma_test)


@pytest.mark.parametrize("xc", IMPLEMENTED)
def test_xc_shape(xc):
    """Compare return shapes from functionals compared to the wrapper function call."""
    # Skip the mock functional
    if "mock" in xc:
        return

    if "_spin" in xc:
        Nspin = 2
    else:
        Nspin = 1
    n_spin = n_tests[Nspin]
    dn_spin = np.stack((n_spin, n_spin, n_spin), axis=2)

    # The functionals need n and zeta...
    n = np.sum(n_spin, axis=0)
    zeta = get_zeta(n_spin)
    e_out, v_out, vsigma_out = IMPLEMENTED[xc](n, zeta=zeta, dn_spin=dn_spin)

    # ...while the wrapper function takes n_spin instead
    e_test, v_test, vsigma_test, _ = get_xc(xc.replace("_spin", ""), n_spin, Nspin, dn_spin=dn_spin)

    assert e_out.shape == e_test.shape
    assert v_out.shape == v_test.shape
    if vsigma_out is not None:
        assert vsigma_out.shape == vsigma_test.shape


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
