# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the GDSMFB exchange-correlation functional."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus import Atoms, SCF
from eminus.xc.lda_xc_gdsmfb import (  # type: ignore[attr-defined]
    _get_fxc_zeta,
    _get_phi,
    _get_theta,
    _get_theta0,
    _get_theta1,
    lda_xc_gdsmfb,
    lda_xc_gdsmfb_spin,
    PhiParams,
    Zeta0Coeffs,
    Zeta1Coeffs,
)


@pytest.mark.parametrize(
    ("n_up", "n_dw", "T"),
    [
        (0.9, 0.1, 0.01),
        (0.4, 0.6, 0.1),
        (0.5, 0.5, 1),
        (0.1, 0.1, 100),
    ],
)
def test_lda_xc_gdsmfb_spin_vxc(n_up, n_dw, T):
    """Compare functional potentials to finite difference derivatives."""
    # Input
    n_up = np.asarray(n_up)
    n_dw = np.asarray(n_dw)

    # Calculate analytical energy density and derivative
    n = n_up + n_dw
    zeta = (n_up - n_dw) / n
    fxc, vxc, _ = lda_xc_gdsmfb_spin(n, zeta, T=T)

    def get_fxc_n_up_n_dw(n_up, n_dw):
        """Wrapper function for the finite difference calculation.

        Returns:
            Functional energy density.
        """
        # Calculate properties
        n = n_up + n_dw
        zeta = (n_up - n_dw) / n
        rs = (3 / (4 * np.pi * n)) ** (1 / 3)
        theta = _get_theta(T, n, zeta)
        theta0 = _get_theta0(theta, zeta)
        theta1 = _get_theta1(theta, zeta)
        # Initialize parameters
        phi_params = PhiParams()
        zeta0theta0 = Zeta0Coeffs(theta0)
        zeta1theta1 = Zeta1Coeffs(theta1)
        # Calculate fxc
        fxc0 = _get_fxc_zeta(rs, zeta0theta0)
        fxc1 = _get_fxc_zeta(rs, zeta1theta1)
        phi = _get_phi(rs, theta0, zeta, phi_params)
        return fxc0 + (fxc1 - fxc0) * phi

    def get_vxc_up(n_up, n_dw, eps=1e-6):
        """Finite difference derivative dfxc / dn_up using a central difference gradient.

        Returns:
            Functional potential.
        """
        exc1 = get_fxc_n_up_n_dw(n_up + eps, n_dw)  # type: ignore[no-untyped-call]
        exc2 = get_fxc_n_up_n_dw(n_up - eps, n_dw)  # type: ignore[no-untyped-call]
        return (exc1 - exc2) / (2 * eps)

    def get_vxc_dw(n_up, n_dw, eps=1e-6):
        """Finite difference derivative dfxc / dn_up using a central difference gradient.

        Returns:
            Functional potential.
        """
        exc1 = get_fxc_n_up_n_dw(n_up, n_dw + eps)  # type: ignore[no-untyped-call]
        exc2 = get_fxc_n_up_n_dw(n_up, n_dw - eps)  # type: ignore[no-untyped-call]
        return (exc1 - exc2) / (2 * eps)

    # Calculate finite difference derivatives
    vxc_up = get_vxc_up(n_up, n_dw)  # type: ignore[no-untyped-call]
    vxc_dw = get_vxc_dw(n_up, n_dw)  # type: ignore[no-untyped-call]
    vxc_fd = fxc + np.array([vxc_up, vxc_dw]) * n

    assert_allclose(vxc_fd, vxc, atol=1e-4)


@pytest.mark.parametrize(
    ("rs", "zeta", "T", "ref"),
    [
        (0.4, 0.0, 1e-8, -1.22407997),
        (0.8, 0.4, 1e-6, -0.65378099),
        (1, 0.8, 1e-3, -0.57261495),
        (2, 0.9, 3, -0.12552032),
        (3, 1.0, 10, -0.03624066),
    ],
)
def test_lda_xc_gdsmfb_spin_exc(rs, zeta, T, ref):
    """Compare functional energy densities to reference values from the original implementation.

    Reference values generated with fxc.py from https://github.com/agbonitz/xc_functional.git
    """
    n = 1 / (4 * np.pi / 3 * rs**3)
    e_out, _, _ = lda_xc_gdsmfb_spin(np.array([n]), np.array([zeta]), T=T)
    assert_allclose(e_out, ref)


@pytest.mark.parametrize(("rs", "T", "ref"), [(0.4, 1e-8, -1.22407997)])
def test_lda_xc_gdsmfb_exc(rs, T, ref):
    """Compare functional energy densities to reference values from the original implementation.

    Test case for the spin-unpolarized case from test_lda_xc_gdsmfb_spin_exc.
    """
    n = 1 / (4 * np.pi / 3 * rs**3)
    e_out, _, _ = lda_xc_gdsmfb(np.array([n]), T=T)
    assert_allclose(e_out, ref)


def test_lda_xc_gdsmfb_dft():
    """DFT calculation with tight thresholds to ensure that energies and gradients don't change."""
    atoms = Atoms("He", [0, 0, 0], ecut=1)
    scf = SCF(atoms, guess="pseudo", xc="lda_xc_gdsmfb", opt={"sd": 5})
    etot = scf.run()
    assert_allclose(etot, -1.528978872314443)


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
