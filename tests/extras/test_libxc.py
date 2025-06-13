# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="no-redef"
"""Test libxc extra."""

import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from eminus import backend as xp
from eminus.xc import get_exc, get_vxc, get_xc_defaults

# Create random mock densities
# Use absolute values since eminus' functionals have no safety checks for simplicity and performance
rng = default_rng()
n_tests = {
    1: xp.abs(xp.asarray(rng.standard_normal((1, 10000)))),
    2: xp.abs(xp.asarray(rng.standard_normal((2, 10000)))),
}


@pytest.mark.parametrize("xc", ["1", "7", "101", "130"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_libxc_functional_exc(xc, Nspin):
    """Compare Libxc functional energy densities to internal functionals."""
    pytest.importorskip("pylibxc", reason="pylibxc not installed, skip tests")
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras import libxc_functional

    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)  # type: np.typing.NDArray[np.floating]
    e_out, _, _, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    e_test = get_exc(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(e_out, e_test)


@pytest.mark.parametrize("xc", ["1", "7", "101", "130"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_libxc_functional_vxc(xc, Nspin):
    """Compare Libxc functional potentials to internal functionals."""
    pytest.importorskip("pylibxc", reason="pylibxc not installed, skip tests")
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras import libxc_functional

    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)  # type: np.typing.NDArray[np.floating]
    _, v_out, _, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    v_test, _, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(v_out, v_test)


@pytest.mark.parametrize("xc", ["101", "130"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_libxc_functional_vsigmaxc(xc, Nspin):
    """Compare Libxc functional vsigma to internal functionals."""
    pytest.importorskip("pylibxc", reason="pylibxc not installed, skip tests")
    from eminus.extras import libxc_functional

    n_spin = n_tests[Nspin]
    dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)
    _, _, vsigma_out, _ = libxc_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    _, vsigma_test, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(vsigma_out, vsigma_test)


@pytest.mark.parametrize("Nspin", [1, 2])
def test_libxc_ext_params(Nspin):
    """Test the custom xc parameters in libxc."""
    pytest.importorskip("pylibxc", reason="pylibxc not installed, skip tests")
    from eminus.extras import libxc_functional

    assert get_xc_defaults(":577") == {"T": 0}
    get_xc_defaults(":18,:21")  # This should print a warning

    n_spin = n_tests[Nspin]
    dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)
    xc_params = {
        "_beta": 0.046,  # PBEsol parameter
        "_mu": 10 / 81,  # PBEsol parameter
    }

    # Make sure default parameters are viable
    libxc_functional("GGA_X_PBE_SOL", n_spin, Nspin, dn_spin=dn_spin, xc_params=None)
    libxc_functional("GGA_X_PBE_SOL", n_spin, Nspin, dn_spin=dn_spin, xc_params={})

    e_out, v_out, vsigma_out, _ = libxc_functional("GGA_X_PBE_SOL", n_spin, Nspin, dn_spin=dn_spin)
    e_test, v_test, vsigma_test, _ = libxc_functional(
        "GGA_X_PBE", n_spin, Nspin, dn_spin=dn_spin, xc_params=xc_params
    )
    assert_allclose(e_out, e_test)
    assert_allclose(v_out, v_test)
    assert_allclose(vsigma_out, vsigma_test)

    e_out, v_out, vsigma_out, _ = libxc_functional("GGA_C_PBE_SOL", n_spin, Nspin, dn_spin=dn_spin)
    e_test, v_test, vsigma_test, _ = libxc_functional(
        "GGA_C_PBE", n_spin, Nspin, dn_spin=dn_spin, xc_params=xc_params
    )
    assert_allclose(e_out, e_test)
    assert_allclose(v_out, v_test)
    assert_allclose(vsigma_out, vsigma_test)


@pytest.mark.parametrize("xc", ["1", "7", "101", "130"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_pyscf_functional_exc(xc, Nspin):
    """Compare Libxc functional energy densities as implemented in PySCF to internal functionals."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras.libxc import pyscf_functional

    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)  # type: np.typing.NDArray[np.floating]
    e_out, _, _, _ = pyscf_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    e_test = get_exc(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(e_out, e_test)


@pytest.mark.parametrize("xc", ["1", "7", "101", "130"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_pyscf_functional_vxc(xc, Nspin):
    """Compare Libxc functional potentials as implemented in PySCF to internal functionals."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from pyscf.dft.libxc import is_gga

    from eminus.extras.libxc import pyscf_functional

    n_spin = n_tests[Nspin]
    dn_spin = None
    if is_gga(xc):
        dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)  # type: np.typing.NDArray[np.floating]
    _, v_out, _, _ = pyscf_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    v_test, _, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(v_out, v_test)


@pytest.mark.parametrize("xc", ["101", "130"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_pyscf_functional_vsigmaxc(xc, Nspin):
    """Compare Libxc functional vsigma as implemented in PySCF to internal functionals."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from eminus.extras.libxc import pyscf_functional

    n_spin = n_tests[Nspin]
    dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)
    _, _, vsigma_out, _ = pyscf_functional(xc, n_spin, Nspin, dn_spin=dn_spin)
    _, vsigma_test, _ = get_vxc(xc, n_spin, Nspin, dn_spin=dn_spin)
    assert_allclose(vsigma_out, vsigma_test)


@pytest.mark.parametrize("xc", ["202", "231"])
@pytest.mark.parametrize("Nspin", [1, 2])
def test_pyscf_mgga(xc, Nspin):
    """Test the execution of meta-GGAs using PySCF."""
    pytest.importorskip("pylibxc", reason="pylibxc not installed, skip tests")
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    from eminus.extras.libxc import libxc_functional, pyscf_functional

    n_spin = n_tests[Nspin]
    dn_spin = xp.stack([n_spin, n_spin, n_spin], axis=2)
    tau = n_spin
    e_out, v_out, vsigma_out, vtau_out = pyscf_functional(
        xc, n_spin, Nspin, dn_spin=dn_spin, tau=tau
    )
    e_test, v_test, vsigma_test, vtau_test = libxc_functional(
        xc, n_spin, Nspin, dn_spin=dn_spin, tau=tau
    )
    assert_allclose(e_out, e_test)
    assert_allclose(v_out, v_test)
    assert_allclose(vsigma_out, vsigma_test)
    assert_allclose(vtau_out, vtau_test)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
