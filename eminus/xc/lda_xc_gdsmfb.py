# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""GDSMFB LDA exchange-correlation.

Reference: Phys. Rev. Lett. 119, 135001.
"""

import dataclasses
import functools

import numpy as np


def lda_xc_gdsmfb(n, T=0, **kwargs):
    """GDSMFB exchange-correlation functional (spin-paired).

    Similar to the functional with the label LDA_XC_GDSMFB and ID 577 in Libxc, but the
    implementations differ: https://gitlab.com/libxc/libxc/-/issues/525
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 119, 135001.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        T: Temperature in Hartree.
        **kwargs: Throwaway arguments.

    Returns:
        GDSMFB exchange-correlation energy density and potential.
    """
    kwargs["zeta"] = np.zeros_like(n)
    exc, vxc, _ = lda_xc_gdsmfb_spin(n, T=T, **kwargs)
    return exc, np.array([vxc[0]]), None


def lda_xc_gdsmfb_spin(n, zeta, T=0, **kwargs):
    """GDSMFB exchange-correlation functional (spin-polarized).

    Similar to the functional with the label LDA_XC_GDSMFB and ID 577 in Libxc, but the
    implementations differ: https://gitlab.com/libxc/libxc/-/issues/525
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 119, 135001.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        T: Temperature in Hartree.
        **kwargs: Throwaway arguments.

    Returns:
        GDSMFB exchange-correlation energy density and potential.
    """
    # Calculate properties
    n_up = (1 + zeta) * n / 2
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    theta = _get_theta(T, n, zeta)
    theta0 = _get_theta0(theta, zeta)
    theta1 = _get_theta1(theta, zeta)

    # Initialize parameters
    # We need to calculate coefficients for specific theta
    # Create a coefficients object for each theta with its corresponding parameters
    phi_params = PhiParams()
    zeta0theta = Zeta0Coeffs(theta)
    zeta0theta0 = Zeta0Coeffs(theta0)
    zeta1theta = Zeta1Coeffs(theta)
    zeta1theta1 = Zeta1Coeffs(theta1)

    # Calculate fxc
    fxc0 = _get_fxc_zeta(rs, zeta0theta0)
    fxc1 = _get_fxc_zeta(rs, zeta1theta1)
    phi = _get_phi(rs, theta0, zeta, phi_params)
    fxc = fxc0 + (fxc1 - fxc0) * phi

    # Generic derivatives
    drsdn = -(6 ** (1 / 3)) * (1 / n) ** (1 / 3) / (6 * np.pi ** (1 / 3) * n)
    dzetadn_up = -zeta / n**2 + 1 / n
    dzetadn_dw = -zeta / n**2 - 1 / n

    # fxc derivatives
    dfxc0drs = _get_dfxc_zetadrs(rs, zeta0theta)
    dfxc1drs = _get_dfxc_zetadrs(rs, zeta1theta)
    dfxc0dtheta0 = _get_dfxc_zetadtheta(rs, zeta0theta0)
    dfxc1dtheta1 = _get_dfxc_zetadtheta(rs, zeta1theta1)

    # phi derivatives
    dphidrs = _get_dphidrs(rs, theta0, zeta, phi_params)
    dphidtheta = _get_dphidtheta(rs, theta0, zeta, phi_params)
    dphidzeta = _get_dphidzeta(rs, theta0, zeta, phi_params)

    # theta derivatives
    dthetadn_up = _get_dthetadn_up(T, n_up)
    dtheta0dtheta = _get_dtheta0dtheta(zeta)
    dtheta0dzeta = _get_dtheta0dzeta(theta0, zeta)
    dtheta1dtheta0 = _get_dtheta1dtheta0()

    # Calculate vxc_up (using dndn_up=1)
    dfxc0a = dfxc0drs * drsdn  # * dndn_up = 1
    dfxc1a = dfxc1drs * drsdn  # * dndn_up = 1
    dfxc0b_up = dfxc0dtheta0 * (dtheta0dtheta * dthetadn_up + dtheta0dzeta * dzetadn_up)
    dfxc1b_up = (
        dfxc1dtheta1 * dtheta1dtheta0 * (dtheta0dtheta * dthetadn_up + dtheta0dzeta * dzetadn_up)
    )
    dphi_up = dphidtheta * dthetadn_up + dphidzeta * dzetadn_up + dphidrs * drsdn  # * dndn_up = 1
    vxc_up = (
        dfxc0a
        + dfxc0b_up
        - phi * (dfxc0a + dfxc0b_up - dfxc1a - dfxc1b_up)
        - (fxc0 - fxc1) * dphi_up
    )

    # Calculate vxc_dw (using dndn_dw=1 and dthetadn_dw=0)
    # dfxc0a and dfxc1a are identical for vxc_up and vxc_dw
    dfxc0b_dw = dfxc0dtheta0 * dtheta0dzeta * dzetadn_dw
    # dfxc0b += dfxc0dtheta0 * dtheta0dtheta * dthetadn_dw
    dfxc1b_dw = dfxc1dtheta1 * dtheta1dtheta0 * dtheta0dzeta * dzetadn_dw
    # dfxc1b += dfxc1dtheta1 * dtheta1dtheta0 * dtheta0dtheta * dthetadn_dw
    dphi_dw = dphidzeta * dzetadn_dw + dphidrs * drsdn  # * dndn_dw = 1
    # dfxc1b += dphidtheta * dthetadn_dw
    vxc_dw = (
        dfxc0a
        + dfxc0b_dw
        - phi * (dfxc0a + dfxc0b_dw - dfxc1a - dfxc1b_dw)
        - (fxc0 - fxc1) * dphi_dw
    )

    return fxc, fxc + np.array([vxc_up, vxc_dw]) * n, None


# ### Temperature dependent coefficients ###


@dataclasses.dataclass
class Coefficients:
    """Coefficients class to calculate temperature/theta dependent coefficients.

    Reference: Phys. Rev. Lett. 119, 135001.
    """

    theta: float  #: Reduced temperature.
    a0: float = 0.610887
    a1: float = 0.75
    a2: float = 3.04363
    a3: float = -0.09227
    a4: float = 1.7035
    a5: float = 8.31051
    a6: float = 5.1105

    @functools.cached_property
    def b5(self):
        """Calculate b5."""
        return self.b3 * np.sqrt(3 / 2) * self.omega * (4 / (9 * np.pi)) ** (-1 / 3)

    @functools.cached_property
    def a(self):
        """Calculate a."""
        with np.errstate(divide="ignore"):
            u = self.a0 * np.tanh(
                1 / self.theta, out=np.ones_like(self.theta), where=self.theta > 0
            )
        return u * _pade(self.theta, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6)

    @property
    def dadtheta(self):
        """Calculate da / dtheta."""
        with np.errstate(divide="ignore"):
            u = self.a0 * np.tanh(
                1 / self.theta, out=np.ones_like(self.theta), where=self.theta > 0
            )
        du = np.divide(
            u**2 / self.a0 - self.a0,
            self.theta**2,
            out=np.zeros_like(self.theta),
            where=self.theta > 0,
        )
        v, dv = _dpade(self.theta, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6)
        return du * v + u * dv

    @functools.cached_property
    def b(self):
        """Calculate b."""
        with np.errstate(divide="ignore"):
            u = np.tanh(1 / np.sqrt(self.theta), out=np.ones_like(self.theta), where=self.theta > 0)
        return u * _pade(self.theta, self.b1, self.b2, 0, self.b3, self.b4, self.b5)

    @property
    def dbdtheta(self):
        """Calculate db / dtheta."""
        with np.errstate(divide="ignore"):
            u = np.tanh(1 / np.sqrt(self.theta), out=np.ones_like(self.theta), where=self.theta > 0)
        du = np.divide(
            u**2 - 1,
            2 * self.theta * np.sqrt(self.theta),
            out=np.zeros_like(self.theta),
            where=self.theta > 0,
        )
        v, dv = _dpade(self.theta, self.b1, self.b2, 0, self.b3, self.b4, self.b5)
        return du * v + u * dv

    @functools.cached_property
    def c(self):
        """Calculate c."""
        with np.errstate(divide="ignore"):
            exp = np.exp(-1 / self.theta, out=np.zeros_like(self.theta), where=self.theta > 0)
        return (self.c1 + self.c2 * exp) * self.e

    @property
    def dcdtheta(self):
        """Calculate dc / dtheta."""
        with np.errstate(divide="ignore"):
            exp = np.exp(-1 / self.theta, out=np.zeros_like(self.theta), where=self.theta > 0)
        u = self.c1 + self.c2 * exp
        du = np.divide(
            self.c2 * exp, self.theta**2, out=np.zeros_like(self.theta), where=self.theta > 0
        )
        v, dv = self.e, self.dedtheta
        return du * v + u * dv

    @functools.cached_property
    def d(self):
        """Calculate d."""
        with np.errstate(divide="ignore"):
            u = np.tanh(1 / np.sqrt(self.theta), out=np.ones_like(self.theta), where=self.theta > 0)
        return u * _pade(self.theta, self.d1, self.d2, 0, self.d3, self.d4, self.d5)

    @property
    def dddtheta(self):
        """Calculate dd / dtheta."""
        with np.errstate(divide="ignore"):
            u = np.tanh(1 / np.sqrt(self.theta), out=np.ones_like(self.theta), where=self.theta > 0)
        du = np.divide(
            u**2 - 1,
            2 * self.theta * np.sqrt(self.theta),
            out=np.zeros_like(self.theta),
            where=self.theta > 0,
        )
        v, dv = _dpade(self.theta, self.d1, self.d2, 0, self.d3, self.d4, self.d5)
        return du * v + u * dv

    @functools.cached_property
    def e(self):
        """Calculate e."""
        with np.errstate(divide="ignore"):
            u = np.tanh(1 / self.theta, out=np.ones_like(self.theta), where=self.theta > 0)
        return u * _pade(self.theta, self.e1, self.e2, 0, self.e3, self.e4, self.e5)

    @functools.cached_property
    def dedtheta(self):
        """Calculate de / dtheta."""
        with np.errstate(divide="ignore"):
            u = np.tanh(1 / self.theta, out=np.ones_like(self.theta), where=self.theta > 0)
        du = np.divide(u**2 - 1, self.theta**2, out=np.zeros_like(self.theta), where=self.theta > 0)
        v, dv = _dpade(self.theta, self.e1, self.e2, 0, self.e3, self.e4, self.e5)
        return du * v + u * dv


# ### Parameters ###


@dataclasses.dataclass
class Zeta0Coeffs(Coefficients):
    """Coefficient class using the parameters for zeta=0.

    Reference: Phys. Rev. Lett. 119, 135001.
    """

    omega: float = 1
    b1: float = 0.3436902
    b2: float = 7.82159531356
    b3: float = 0.300483986662
    b4: float = 15.8443467125
    c1: float = 0.8759442
    c2: float = -0.230130843551
    d1: float = 0.72700876
    d2: float = 2.38264734144
    d3: float = 0.30221237251
    d4: float = 4.39347718395
    d5: float = 0.729951339845
    e1: float = 0.25388214
    e2: float = 0.815795138599
    e3: float = 0.0646844410481
    e4: float = 15.0984620477
    e5: float = 0.230761357474


@dataclasses.dataclass
class Zeta1Coeffs(Coefficients):
    """Coefficient class using the parameters for zeta=1.

    Reference: Phys. Rev. Lett. 119, 135001.
    """

    omega: float = 2 ** (1 / 3)
    b1: float = 0.84987704
    b2: float = 3.04033012073
    b3: float = 0.0775730131248
    b4: float = 7.57703592489
    c1: float = 0.91126873
    c2: float = -0.0307957123308
    d1: float = 1.48658718
    d2: float = 4.92684905511
    d3: float = 0.0849387225179
    d4: float = 8.3269821188
    d5: float = 0.218864952126
    e1: float = 0.27454097
    e2: float = 0.400994856555
    e3: float = 2.88773194962
    e4: float = 6.33499237092
    e5: float = 24.823008753


@dataclasses.dataclass
class PhiParams:
    """Parameter class holding the spin-interpolation function parameters.

    Reference: Phys. Rev. Lett. 119, 135001.
    """

    # Sign of parameters is different from the supplemental material
    h1: float = 3.18747258
    h2: float = 7.74662802
    lambda1: float = 1.85909536
    lambda2: float = 0


# ### Pade approximation and derivative ###


def _pade(x, n1, n2, n3, n4, d1, d2):
    """Pade approximation.

    Not the general case but as needed for this functional.
    """
    x2, x4 = x**2, x**4
    num = n1 + n2 * x2 + n3 * x**3 + n4 * x4
    denom = 1 + d1 * x2 + d2 * x4
    return num / denom


def _dpade(x, n1, n2, n3, n4, d1, d2):
    """Pade approximation and its derivative."""
    x2, x3, x4 = x**2, x**3, x**4
    num = n1 + n2 * x2 + n3 * x3 + n4 * x4
    denom = 1 + d1 * x2 + d2 * x4
    dnum = 2 * n2 * x + 3 * n3 * x2 + 4 * n4 * x3
    ddenom = 2 * d1 * x + 4 * d2 * x3
    # df = (a'b - ab') / b^2
    return num / denom, (dnum * denom - num * ddenom) / denom**2


# ### theta and derivatives ###


def _get_theta(T, n, zeta):
    """Calculate the reduced temperature theta.

    Reference: Phys. Rev. Lett. 119, 135001.
    Only mentioned in the arXiv version: https://arxiv.org/abs/1703.08074
    """
    n_up = (1 + zeta) * n / 2
    T_fermi = (6 * np.pi**2 * n_up) ** (2 / 3) / 2
    return T / T_fermi


def _get_dthetadn_up(T, n_up):
    """Calculate dtheta / dn_up."""
    return -4 / (3 * 6 ** (2 / 3)) * T / (np.pi ** (4 / 3) * n_up ** (5 / 3))


def _get_theta0(theta, zeta):
    """Calculate theta0.

    Reference: Phys. Rev. Lett. 119, 135001.
    """
    return theta * (1 + zeta) ** (2 / 3)


def _get_dtheta0dtheta(zeta):
    """Calculate dtheta0 / dtheta."""
    return (1 + zeta) ** (2 / 3)


def _get_dtheta0dzeta(theta, zeta):
    """Calculate dtheta0 / dzeta."""
    return 2 * theta / (3 * (1 + zeta) ** (1 / 3))


def _get_theta1(theta, zeta):
    """Calculate theta1.

    Reference: Phys. Rev. Lett. 119, 135001.

    It is not explicitly mentioned but introduced as used in Eq. (5).
    """
    theta0 = _get_theta0(theta, zeta)
    return theta0 * 2 ** (-2 / 3)


def _get_dtheta1dtheta0():
    """Calculate dtheta1 / dtheta0."""
    return 2 ** (-2 / 3)


# ### fxc_zeta and derivatives ###


def _get_fxc_zeta(rs, p):
    """Calculate the Pade formula f_xc^zeta.

    Reference: Phys. Rev. Lett. 119, 135001.
    """
    num = p.a * p.omega + p.b * np.sqrt(rs) + p.c * rs
    denom = 1 + p.d * np.sqrt(rs) + p.e * rs
    return -1 / rs * num / denom


def _get_dfxc_zetadrs(rs, p):
    """Calculate dfxc_zeta / drs."""
    num = p.a * p.omega + p.b * np.sqrt(rs) + p.c * rs
    denom = 1 + p.d * np.sqrt(rs) + p.e * rs
    tmp1 = (p.d / (2 * np.sqrt(rs)) + p.e) * num / denom**2
    tmp2 = (p.b / (2 * np.sqrt(rs)) + p.c) / denom
    return (tmp1 - tmp2) / rs + num / denom / rs**2


def _get_dfxc_zetadtheta(rs, p):
    """Calculate dfxc_zeta / dzeta."""
    num = p.a * p.omega + p.b * np.sqrt(rs) + p.c * rs
    denom = 1 + p.d * np.sqrt(rs) + p.e * rs
    tmp1 = (p.dddtheta * np.sqrt(rs) + p.dedtheta * rs) * num / denom**2
    tmp2 = (p.dadtheta * p.omega + p.dbdtheta * np.sqrt(rs) + p.dcdtheta * rs) / denom
    return (tmp1 - tmp2) / rs


# ### phi and derivatives ###


def _get_phi(rs, theta, zeta, phi_params):
    """Calculate the interpolation function phi.

    Reference: Phys. Rev. Lett. 119, 135001.
    """
    alpha = _get_alpha(rs, theta, phi_params)
    return ((1 + zeta) ** alpha + (1 - zeta) ** alpha - 2) / (2**alpha - 2)


def _get_dphidrs(rs, theta, zeta, phi_params):
    """Calculate dphi / drs."""
    alpha = _get_alpha(rs, theta, phi_params)
    dalphadrs = _get_dalphadrs(rs, theta, phi_params)
    tmp1 = (1 - zeta) ** alpha * np.log(1 - zeta, out=np.zeros_like(zeta), where=1 - zeta > 0)
    tmp2 = (1 + zeta) ** alpha * np.log(1 + zeta)
    duv = (tmp1 + tmp2) * (2**alpha - 2)
    udv = ((1 - zeta) ** alpha + (1 + zeta) ** alpha - 2) * 2**alpha * np.log(2)
    vv = (2**alpha - 2) ** 2
    return (duv - udv) * dalphadrs / vv


def _get_dphidtheta(rs, theta, zeta, phi_params):
    """Calculate dphi / dtheta."""
    alpha = _get_alpha(rs, theta, phi_params)
    dalphadtheta = _get_dalphadtheta(rs, theta, phi_params)
    tmp1 = (1 - zeta) ** alpha * np.log(1 - zeta, out=np.zeros_like(zeta), where=1 - zeta > 0)
    tmp2 = (1 + zeta) ** alpha * np.log(1 + zeta)
    duv = (tmp1 + tmp2) * (2**alpha - 2)
    udv = ((1 - zeta) ** alpha + (1 + zeta) ** alpha - 2) * 2**alpha * np.log(2)
    vv = (2**alpha - 2) ** 2
    return (duv - udv) * dalphadtheta / vv


def _get_dphidzeta(rs, theta, zeta, phi_params):
    """Calculate dphi / dzeta."""
    alpha = _get_alpha(rs, theta, phi_params)
    tmp1 = alpha * (1 + zeta) ** alpha / (1 + zeta)
    tmp2 = np.divide(
        alpha * (1 - zeta) ** alpha, 1 - zeta, out=np.zeros_like(zeta), where=1 - zeta > 0
    )
    return (tmp1 - tmp2) / (2**alpha - 2)


# ### alpha and derivatives ###


def _get_alpha(rs, theta, phi_params):
    """Calculate alpha.

    Reference: Phys. Rev. Lett. 119, 135001.
    """
    h = _get_h(rs, phi_params)
    lamda = _get_lambda(rs, theta, phi_params)
    return 2 - h * np.exp(-theta * lamda)


def _get_dalphadrs(rs, theta, phi_params):
    """Calculate dalpha / drs."""
    h = _get_h(rs, phi_params)
    lamda = _get_lambda(rs, theta, phi_params)
    dhdrs = _get_dhdrs(rs, phi_params)
    dlambdadrs = _get_dlambdadrs(rs, theta, phi_params)
    return -dhdrs * np.exp(-theta * lamda) + dlambdadrs * theta * h * np.exp(-theta * lamda)


def _get_dalphadtheta(rs, theta, phi_params):
    """Calculate dalpha / dtheta."""
    h = _get_h(rs, phi_params)
    lamda = _get_lambda(rs, theta, phi_params)
    dlambdadtheta = _get_dlambdadtheta(rs, phi_params)
    return (dlambdadtheta * theta + lamda) * h * np.exp(-theta * lamda)


# ### h and derivative ###


def _get_h(rs, phi_params):
    """Calculate h.

    Reference: Phys. Rev. Lett. 119, 135001.
    """
    return (2 / 3 + phi_params.h1 * rs) / (1 + phi_params.h2 * rs)


def _get_dhdrs(rs, phi_params):
    """Calculate dh / drs."""
    return (
        phi_params.h1 / (phi_params.h2 * rs + 1)
        - phi_params.h2 * (phi_params.h1 * rs + 2 / 3) / (phi_params.h2 * rs + 1) ** 2
    )


# ### lambda and derivatives ###


def _get_lambda(rs, theta, phi_params):
    """Calculate lambda.

    Reference: Phys. Rev. Lett. 119, 135001.
    """
    return phi_params.lambda1 + phi_params.lambda2 * theta * rs ** (1 / 2)


def _get_dlambdadrs(rs, theta, phi_params):
    """Calculate dlambda / drs."""
    return phi_params.lambda2 * theta / (2 * np.sqrt(rs))


def _get_dlambdadtheta(rs, phi_params):
    """Calculate dlambda / dtheta."""
    return phi_params.lambda2 * np.sqrt(rs)
