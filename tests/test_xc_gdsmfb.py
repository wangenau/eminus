# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test the GDSMFB exchange-correlation functional."""

import inspect
import pathlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from eminus.xc.lda_xc_gdsmfb import (  # type: ignore[attr-defined]
    _get_fxc_zeta,
    _get_phi,
    _get_theta,
    _get_theta0,
    _get_theta1,
    lda_xc_gdsmfb_spin,
    PhiParams,
    Zeta0Coeffs,
    Zeta1Coeffs,
)


def grad(idx, eps=1e-4, shift=False):
    """Central difference gradient to perform finte difference calculations.

    df / dargs[idx]

    idx:   integer, index of the parameter which should be differentiate against
    eps:   finite difference epsilion/h
    shift: shift for Fxc

    Notes:
    - Nested decorator.
    """

    def wrap(f):
        def wrapped_f(*args):
            # central difference stencil
            x0 = args[idx]
            args = list(args)
            r0 = f(*args)
            args[idx] = x0 + eps  # type: ignore[index]
            # assume 1st argument is the energy
            r1 = f(*args)
            args[idx] = x0 - eps  # type: ignore[index]
            # assume 1st argument is the energy
            r2 = f(*args)
            if isinstance(r0, (list, tuple)):
                e0 = r0[0]
                e1 = r1[0]
                e2 = r2[0]
            else:
                e0 = r0
                e1 = r1
                e2 = r2
            # central
            d = (e1 - e2) / (2 * eps)
            # forward
            # d = (e1-e0)/(eps)
            # backward
            # d = (e0-e2)/(eps)
            if shift is False:
                res = d
            if shift is True:
                res = e0 + d
            return res

        return wrapped_f

    return wrap


def test_derivative_lda_xc_gdsmfb_spin():
    """Compare analytical derivatives with finite difference derivatives."""
    # input
    nup = np.array([0.9])
    ndn = np.array([0.1])
    n = nup + ndn
    zeta = (nup - ndn) / n
    # T is limited to 1e-3 b/c of np.coth
    T = 0.1

    # Analytical derivatives
    fxc, vxc, _ = lda_xc_gdsmfb_spin(n, zeta, T=T)

    # Get finite difference derivatives.
    theta = _get_theta(T, n, zeta)
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)

    # parameters
    theta0 = _get_theta0(theta, zeta)
    theta1 = _get_theta1(theta, zeta)
    phi_params = PhiParams()
    zeta0theta0 = Zeta0Coeffs(theta0)
    zeta1theta1 = Zeta1Coeffs(theta1)

    def _get_fxc(rs, theta, zeta, zeta0theta0, zeta1theta1, phi_params):
        theta0 = _get_theta0(theta, zeta)
        fxc0 = _get_fxc_zeta(rs, zeta0theta0)
        fxc1 = _get_fxc_zeta(rs, zeta1theta1)
        phi = _get_phi(rs, theta0, zeta, phi_params)
        return fxc0 + (fxc1 - fxc0) * phi

    # fxc
    fxc = _get_fxc(rs, theta, zeta, zeta0theta0, zeta1theta1, phi_params)  # type: ignore[no-untyped-call]

    def _get_fxc_nupndn(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params):
        """Get fxc utilizing nup, ndn, and T."""
        n = nup + ndn
        zeta = (nup - ndn) / n
        rs = (3 / (4 * np.pi * n)) ** (1 / 3)
        theta = _get_theta(T, n, zeta)
        return _get_fxc(rs, theta, zeta, zeta0theta0, zeta1theta1, phi_params)  # type: ignore[no-untyped-call]

    @grad(idx=0, eps=1e-5, shift=False)  # type: ignore[no-untyped-call]
    def _get_fd1(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params):
        """Finite difference derivative dfxc / dnup."""
        return _get_fxc_nupndn(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params)  # type: ignore[no-untyped-call]

    @grad(idx=1, eps=1e-5, shift=False)  # type: ignore[no-untyped-call]
    def _get_fd2(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params):
        """Finite difference derivative dfxc / dndn."""
        return _get_fxc_nupndn(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params)  # type: ignore[no-untyped-call]

    fd1 = _get_fd1(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params)
    fd2 = _get_fd2(nup, ndn, T, zeta0theta0, zeta1theta1, phi_params)
    vxc_fd = np.array([fd1, fd2]) * n + fxc

    assert_allclose(vxc_fd, vxc, rtol=1e-03)


def test_energy_lda_xc_gdsmfb_spin():
    """Validate the energy expression of GDSMFB.

    Comparison with reference values from the original implementation.
    """
    RS = np.array([0.4, 0.8, 1, 2, 3])
    TEMP = np.array([1e-8, 1e-6, 1e-3, 3, 10])
    ZETA = np.array([0, 0.4, 0.8, 0.9, 1])
    # Reference values generated with fxc.py
    # from https://github.com/agbonitz/xc_functional.git
    REF = np.array([-1.22407997, -0.65378099, -0.57261495, -0.12552032, -0.03624066])
    CAL = np.zeros_like(REF)
    for idx, (rs, zeta, T) in enumerate(zip(RS, ZETA, TEMP)):
        n = 1 / (4 * np.pi / 3 * rs**3)
        fxc, _, _ = lda_xc_gdsmfb_spin(n, zeta, T=T)
        CAL[idx] = fxc

    assert_allclose(CAL, REF, rtol=1e-07)


if __name__ == "__main__":
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
