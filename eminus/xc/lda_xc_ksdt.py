# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""KSDT LDA exchange-correlation.

Reference: Phys. Rev. Lett. 112, 076403.
"""

import dataclasses
import functools

import numpy as np

# ### Temperature dependent coefficients ###


@dataclasses.dataclass
class Coefficients:
    """Coefficients class to calculate temperature/theta dependent coefficients.

    Reference: Phys. Rev. Lett. 112, 076403.
    """

    theta: float  #: Reduced temperature.
    omega: float
    e1: float
    e2: float
    e3: float
    e4: float
    e5: float
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    c1: float
    c2: float
    c3: float
    b1: float
    b2: float
    b3: float
    b4: float
    a1: float = 0.75
    a2: float = 3.04363
    a3: float = -0.09227
    a4: float = 1.7035
    a5: float = 8.31051
    a6: float = 5.1105

    @functools.cached_property
    def a0(self):
        """Calculate a0."""
        return 1 / (np.pi * (4 / (9 * np.pi)) ** (1 / 3))

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
            exp = np.exp(-self.c3 / self.theta, out=np.zeros_like(self.theta), where=self.theta > 0)
        return (self.c1 + self.c2 * exp) * self.e

    @property
    def dcdtheta(self):
        """Calculate dc / dtheta."""
        with np.errstate(divide="ignore"):
            exp = np.exp(-self.c3 / self.theta, out=np.zeros_like(self.theta), where=self.theta > 0)
        u = self.c1 + self.c2 * exp
        du = np.divide(
            self.c2 * self.c3 * exp,
            self.theta**2,
            out=np.zeros_like(self.theta),
            where=self.theta > 0,
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

    Reference: Phys. Rev. Lett. 112, 076403.
    """

    omega: float = 1
    b1: float = 0.283997
    b2: float = 48.932154
    b3: float = 0.370919
    b4: float = 61.095357
    c1: float = 0.870089
    c2: float = 0.193077
    c3: float = 2.414644
    d1: float = 0.579824
    d2: float = 94.537454
    d3: float = 97.839603
    d4: float = 59.939999
    d5: float = 24.388037
    e1: float = 0.212036
    e2: float = 16.731249
    e3: float = 28.485792
    e4: float = 34.028876
    e5: float = 17.235515


@dataclasses.dataclass
class Zeta1Coeffs(Coefficients):
    """Coefficient class using the parameters for zeta=1.

    Reference: Phys. Rev. Lett. 112, 076403.
    """

    omega: float = 2 ** (1 / 3)
    b1: float = 0.329001
    b2: float = 111.598308
    b3: float = 0.537053
    b4: float = 105.086663
    c1: float = 0.84893
    c2: float = 0.167952
    c3: float = 0.08882
    d1: float = 0.55133
    d2: float = 180.213159
    d3: float = 134.486231
    d4: float = 103.861695
    d5: float = 17.75071
    e1: float = 0.153124
    e2: float = 19.543945
    e3: float = 43.400337
    e4: float = 120.255145
    e5: float = 15.662836


@dataclasses.dataclass
class PhiParams:
    """Parameter class holding the spin-interpolation function parameters.

    Reference: Phys. Rev. Lett. 112, 076403.
    """

    g1: float = 2 / 3
    g2: float = -0.0139261
    g3: float = 0.183208
    lambda1: float = 1.064009
    lambda2: float = 0.572565


# ### Functional implementation ###


def lda_xc_ksdt(
    n, T=0, zeta0_coeffs=Zeta0Coeffs, zeta1_coeffs=Zeta1Coeffs, phi_params=PhiParams, **kwargs
):
    """KSDT exchange-correlation functional (spin-polarized).

    Similar to the functional with the label LDA_XC_KSDT and ID 259 in Libxc, but the
    implementations differ: https://gitlab.com/libxc/libxc/-/issues/525
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 112, 076403.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        T: Temperature in Hartree.
        zeta0_coeffs: Coefficient class using the parameters for zeta=0.
        zeta1_coeffs: Coefficient class using the parameters for zeta=1.
        phi_params: Parameter class holding the spin-interpolation function parameters.
        **kwargs: Throwaway arguments.

    Returns:
        KSDT exchange-correlation energy density and potential.
    """
    kwargs["zeta"] = np.zeros_like(n)
    exc, vxc, _ = lda_xc_ksdt_spin(
        n,
        T=T,
        zeta0_coeffs=zeta0_coeffs,
        zeta1_coeffs=zeta1_coeffs,
        phi_params=phi_params,
        **kwargs,
    )
    return exc, np.array([vxc[0]]), None


def lda_xc_ksdt_spin(
    n, zeta, T=0, zeta0_coeffs=Zeta0Coeffs, zeta1_coeffs=Zeta1Coeffs, phi_params=PhiParams, **kwargs
):
    """KSDT exchange-correlation functional (spin-polarized).

    Similar to the functional with the label LDA_XC_KSDT and ID 259 in Libxc, but the
    implementations differ: https://gitlab.com/libxc/libxc/-/issues/525
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 112, 076403.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        T: Temperature in Hartree.
        zeta0_coeffs: Coefficient class using the parameters for zeta=0.
        zeta1_coeffs: Coefficient class using the parameters for zeta=1.
        phi_params: Parameter class holding the spin-interpolation function parameters.
        **kwargs: Throwaway arguments.

    Returns:
        KSDT exchange-correlation energy density and potential.
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
    phi_params = phi_params()
    zeta0theta = zeta0_coeffs(theta)
    zeta0theta0 = zeta0_coeffs(theta0)
    zeta1theta = zeta1_coeffs(theta)
    zeta1theta1 = zeta1_coeffs(theta1)

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

    Reference: Phys. Rev. Lett. 112, 076403.
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

    Reference: Phys. Rev. Lett. 112, 076403.
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

    Reference: Phys. Rev. Lett. 112, 076403.
    """
    g = _get_g(rs, phi_params)
    lamda = _get_lambda(rs, theta, phi_params)
    return 2 - g * np.exp(-theta * lamda)


def _get_dalphadrs(rs, theta, phi_params):
    """Calculate dalpha / drs."""
    g = _get_g(rs, phi_params)
    lamda = _get_lambda(rs, theta, phi_params)
    dgdrs = _get_dgdrs(rs, phi_params)
    dlambdadrs = _get_dlambdadrs(rs, theta, phi_params)
    return -dgdrs * np.exp(-theta * lamda) + dlambdadrs * theta * g * np.exp(-theta * lamda)


def _get_dalphadtheta(rs, theta, phi_params):
    """Calculate dalpha / dtheta."""
    g = _get_g(rs, phi_params)
    lamda = _get_lambda(rs, theta, phi_params)
    dlambdadtheta = _get_dlambdadtheta(rs, phi_params)
    return (dlambdadtheta * theta + lamda) * g * np.exp(-theta * lamda)


# ### g and derivative ###


def _get_g(rs, phi_params):
    """Calculate g.

    Reference: Phys. Rev. Lett. 112, 076403.
    """
    return (phi_params.g1 + phi_params.g2 * rs) / (1 + phi_params.g3 * rs)


def _get_dgdrs(rs, phi_params):
    """Calculate dg / drs."""
    return (
        phi_params.g2 / (phi_params.g3 * rs + 1)
        - phi_params.g3 * (phi_params.g2 * rs + phi_params.g1) / (phi_params.g3 * rs + 1) ** 2
    )


# ### lambda and derivatives ###


def _get_lambda(rs, theta, phi_params):
    """Calculate lambda.

    Reference: Phys. Rev. Lett. 112, 076403.
    """
    return phi_params.lambda1 + phi_params.lambda2 * theta * rs ** (1 / 2)


def _get_dlambdadrs(rs, theta, phi_params):
    """Calculate dlambda / drs."""
    return phi_params.lambda2 * theta / (2 * np.sqrt(rs))


def _get_dlambdadtheta(rs, phi_params):
    """Calculate dlambda / dtheta."""
    return phi_params.lambda2 * np.sqrt(rs)
