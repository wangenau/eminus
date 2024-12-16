# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""GDSMFB LDA exchange-correlation.

Reference: Phys. Rev. Lett. 119, 135001.
"""

import dataclasses

import numpy as np


def lda_xc_gdsmfb(n, T=0, **kwargs):
    """GDSMFB exchange-correlation functional (spin-paired).

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
    exc, vxc, _ = lda_xc_gdsmfb_spin(n, T=T, **kwargs)
    return exc, np.array([vxc[0]]), None


def lda_xc_gdsmfb_spin(n, zeta, T=0, **kwargs):
    """GDSMFB exchange-correlation functional (spin-polarized).

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
    n_dw = (1 - zeta) * n / 2
    n_up = (1 + zeta) * n / 2
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    theta = get_theta(T, n, zeta)

    p0, p1, p2 = get_gdsmfb_parameters()
    fxc = get_fxc(rs, theta, zeta, p0, p1, p2)
    dfxcdn_up, dfxcdn_dw = get_dfxcdn(n_up, n_dw, T, p0, p1, p2)
    return fxc, fxc + np.array([dfxcdn_up, dfxcdn_dw]) * n, None


@dataclasses.dataclass
class Parameters:
    """Parameters class.

    Holds parameters of exchange-correlation functionals.
    Saving same space through attribute access.
    """

    @property
    def b5(self):
        return self.b3 * np.sqrt(3 / 2) * self.omega * (4 / (9 * np.pi)) ** (-1 / 3)

    def a(self, theta):
        tmp1 = 0.610887 * np.tanh(1 / theta)
        tmp2 = 0.75 + 3.04363 * theta**2 - 0.09227 * theta**3 + 1.7035 * theta**4
        tmp3 = 1 + 8.31051 * theta**2 + 5.1105 * theta**4
        return tmp1 * tmp2 / tmp3

    def b(self, theta):
        tmp1 = np.tanh(1 / np.sqrt(theta)) * (self.b1 + self.b2 * theta**2 + self.b3 * theta**4)
        tmp2 = 1 + self.b4 * theta**2 + self.b5 * theta**4
        return tmp1 / tmp2

    def c(self, theta):
        thres = 1e-6
        return np.where(
            theta > thres,
            (self.c1 + self.c2 * np.exp(-1 / theta)) * self.e(theta),
            self.c1 * self.e(theta),
        )

    def d(self, theta):
        tmp1 = np.tanh(1 / np.sqrt(theta)) * (self.d1 + self.d2 * theta**2 + self.d3 * theta**4)
        tmp2 = 1 + self.d4 * theta**2 + self.d5 * theta**4
        return tmp1 / tmp2

    def e(self, theta):
        return (
            np.tanh(1 / theta)
            * (self.e1 + self.e2 * theta**2 + self.e3 * theta**4)
            / (1 + self.e4 * theta**2 + self.e5 * theta**4)
        )

    def dadtheta(self, theta):  # TODO
        tmp1 = -0.00884515668249876 * (20.442 * theta**3 + 16.62102 * theta)
        tmp2 = 1.7035 * theta**4 - 0.09227 * theta**3 + 3.04363 * theta**2 + 0.75
        tmp3 = (
            np.tanh(1 / theta) / (0.614944209200157 * theta**4 + theta**2 + 0.12032955859508) ** 2
        )
        denom = 5.1105 * theta**4 + 8.31051 * theta**2 + 1
        tmp4 = (
            0.610887
            * (6.814 * theta**3 - 0.27681 * theta**2 + 6.08726 * theta)
            * np.tanh(1 / theta)
            / denom
        )
        with np.errstate(over="ignore"):
            tmp41 = -0.610887 * (1.7035 * theta**4 - 0.09227 * theta**3 + 3.04363 * theta**2 + 0.75)
            tmp42 = denom * theta**2 * np.cosh(1 / theta) ** 2
            tmp43 = tmp41 / tmp42
            tmp5 = np.where(theta < 0.0025, 0, tmp43)
        return tmp1 * tmp2 * tmp3 + tmp4 + tmp5

    def dbdtheta(self, theta):
        tmp1 = (
            (2 * self.b2 * theta + 4 * self.b3 * theta**3)
            * np.tanh(1 / np.sqrt(theta))
            / (self.b4 * theta**2 + self.b5 * theta**4 + 1)
        )
        tmp11 = (
            (2 * self.b4 * theta + 4 * self.b5 * theta**3)
            * (self.b1 + self.b2 * theta**2 + self.b3 * theta**4)
            * np.tanh(1 / np.sqrt(theta))
        )
        tmp12 = (self.b4 * theta**2 + self.b5 * theta**4 + 1) ** 2
        tmp2 = tmp11 / tmp12
        with np.errstate(over="ignore"):
            tmp21 = self.b1 + self.b2 * theta**2 + self.b3 * theta**4
            tmp22 = (
                2
                * (self.b4 * theta**2 + self.b5 * theta**4 + 1)
                * theta ** (3 / 2)
                * np.cosh(1 / np.sqrt(theta)) ** 2
            )
            tmp23 = tmp21 / tmp22
            tmp3 = np.where(theta < 0.001, 0, tmp23)
        return tmp1 - tmp2 - tmp3

    def dcdtheta(self, theta):
        tmp1 = self.c2 * self.e(theta) * np.exp(-1 / theta) / theta**2
        tmp2 = self.c1 + self.c2 * np.exp(-1 / theta)
        tmp3 = (
            (2 * self.e2 * theta + 4 * self.e3 * theta**3)
            * np.tanh(1 / theta)
            / (self.e4 * theta**2 + self.e5 * theta**4 + 1)
        )
        tmp31 = (
            (2 * self.e4 * theta + 4 * self.e5 * theta**3)
            * (self.e1 + self.e2 * theta**2 + self.e3 * theta**4)
            * np.tanh(1 / theta)
        )
        tmp32 = (self.e4 * theta**2 + self.e5 * theta**4 + 1) ** 2
        tmp4 = tmp31 / tmp32
        with np.errstate(over="ignore"):
            tmp41 = self.e1 + self.e2 * theta**2 + self.e3 * theta**4
            tmp42 = (
                (self.e4 * theta**2 + self.e5 * theta**4 + 1) * theta**2 * np.cosh(1 / theta) ** 2
            )
            tmp43 = tmp41 / tmp42
            tmp5 = np.where(theta < 0.0025, 0, tmp43)
        return tmp1 + tmp2 * (tmp3 - tmp4 - tmp5)

    def dddtheta(self, theta):
        tmp1 = (
            (2 * self.d2 * theta + 4 * self.d3 * theta**3)
            * np.tanh(1 / np.sqrt(theta))
            / (self.d4 * theta**2 + self.d5 * theta**4 + 1)
        )
        tmp11 = (
            (2 * self.d4 * theta + 4 * self.d5 * theta**3)
            * (self.d1 + self.d2 * theta**2 + self.d3 * theta**4)
            * np.tanh(1 / np.sqrt(theta))
        )
        tmp12 = (self.d4 * theta**2 + self.d5 * theta**4 + 1) ** 2
        tmp2 = tmp11 / tmp12
        with np.errstate(over="ignore"):
            tmp21 = self.d1 + self.d2 * theta**2 + self.d3 * theta**4
            tmp22 = (
                2
                * (self.d4 * theta**2 + self.d5 * theta**4 + 1)
                * theta ** (3 / 2)
                * np.cosh(1 / np.sqrt(theta)) ** 2
            )
            tmp23 = tmp21 / tmp22
            tmp3 = np.where(theta < 0.001, 0, tmp23)
        return tmp1 - tmp2 - tmp3

    def dedtheta(self, theta):
        tmp1 = (
            (2 * self.e2 * theta + 4 * self.e3 * theta**3)
            * np.tanh(1 / theta)
            / (self.e4 * theta**2 + self.e5 * theta**4 + 1)
        )
        tmp11 = (
            (2 * self.e4 * theta + 4 * self.e5 * theta**3)
            * (self.e1 + self.e2 * theta**2 + self.e3 * theta**4)
            * np.tanh(1 / theta)
        )
        tmp12 = (self.e4 * theta**2 + self.e5 * theta**4 + 1) ** 2
        tmp2 = tmp11 / tmp12
        with np.errstate(over="ignore"):
            tmp21 = self.e1 + self.e2 * theta**2 + self.e3 * theta**4
            tmp22 = (
                (self.e4 * theta**2 + self.e5 * theta**4 + 1) * theta**2 * np.cosh(1 / theta) ** 2
            )
            tmp23 = tmp21 / tmp22
            tmp3 = np.where(theta < 0.0025, 0, tmp23)
        return tmp1 - tmp2 - tmp3


def get_gdsmfb_parameters():
    """Get the GDSMFB parameters."""
    # zeta = 0
    p0 = Parameters()
    p0.omega = 1
    p0.b1 = 0.3436902
    p0.b2 = 7.82159531356
    p0.b3 = 0.300483986662
    p0.b4 = 15.8443467125
    p0.c1 = 0.8759442
    p0.c2 = -0.230130843551
    p0.d1 = 0.72700876
    p0.d2 = 2.38264734144
    p0.d3 = 0.30221237251
    p0.d4 = 4.39347718395
    p0.d5 = 0.729951339845
    p0.e1 = 0.25388214
    p0.e2 = 0.815795138599
    p0.e3 = 0.0646844410481
    p0.e4 = 15.0984620477
    p0.e5 = 0.230761357474

    # zeta = 1
    p1 = Parameters()
    p1.omega = 2 ** (1 / 3)
    p1.b1 = 0.84987704
    p1.b2 = 3.04033012073
    p1.b3 = 0.0775730131248
    p1.b4 = 7.57703592489
    p1.c1 = 0.91126873
    p1.c2 = -0.0307957123308
    p1.d1 = 1.48658718
    p1.d2 = 4.92684905511
    p1.d3 = 0.0849387225179
    p1.d4 = 8.3269821188
    p1.d5 = 0.218864952126
    p1.e1 = 0.27454097
    p1.e2 = 0.400994856555
    p1.e3 = 2.88773194962
    p1.e4 = 6.33499237092
    p1.e5 = 24.823008753

    # spin interpolation
    p2 = Parameters()
    # Sign of parameters is different as in the supp. mat.
    p2.h1 = 3.18747258
    p2.h2 = 7.74662802
    p2.lambda1 = 1.85909536
    p2.lambda2 = 0

    return p0, p1, p2


### fxc


def get_fxc(rs, theta, zeta, p0, p1, p2):
    theta0 = get_theta0(theta, zeta)
    theta1 = get_theta1(theta, zeta)
    fxc0 = get_fxc_zeta(rs, theta0, p0)
    fxc1 = get_fxc_zeta(rs, theta1, p1)
    phi = get_phi(rs, theta0, zeta, p2)
    return fxc0 + (fxc1 - fxc0) * phi


def get_dfxcdn(n_up, n_dw, T, p0, p1, p2):
    n = n_up + n_dw

    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    drsdn = -(6 ** (1 / 3)) * (1 / n) ** (1 / 3) / (6 * np.pi ** (1 / 3) * n)

    zeta = (n_up - n_dw) / n
    dzetadn_up = -(-n_dw + n_up) / (n_dw + n_up) ** 2 + 1 / (n_dw + n_up)
    dzetadn_dw = -(-n_dw + n_up) / (n_dw + n_up) ** 2 - 1 / (n_dw + n_up)

    theta = get_theta(T, n, zeta)
    theta0 = get_theta0(theta, zeta)
    fxc0 = get_fxc_zeta(rs, theta0, p0)
    theta1 = get_theta1(theta, zeta)
    fxc1 = get_fxc_zeta(rs, theta1, p1)
    phi = get_phi(rs, theta0, zeta, p2)

    dfxc0drs = get_dfxc_zetadrs(rs, theta, p0)
    dfxc1drs = get_dfxc_zetadrs(rs, theta, p1)

    dfxc0dtheta0 = get_dfxc_zetadtheta(rs, theta0, p0)
    dfxc1dtheta1 = get_dfxc_zetadtheta(rs, theta1, p1)

    dtheta0dtheta = get_dtheta0dtheta(zeta)
    dthetadn_up = get_dthetadn_up(T, n_up)
    dthetadn_dw = 0
    dtheta0dzeta = get_dtheta0dzeta(theta0, zeta)
    dtheta1dtheta0 = 2 ** (-2 / 3)

    dphidrs = get_dphidrs(rs, theta0, zeta, p2)
    dphidtheta = get_dphidtheta(rs, theta0, zeta, p2)
    dphidzeta = get_dphidzeta(rs, theta0, zeta, p2)

    dndn_up = 1
    dfxc0a = dfxc0drs * dndn_up * drsdn
    dfxc1a = dfxc1drs * dndn_up * drsdn
    dfxc0b = dfxc0dtheta0 * (dtheta0dtheta * dthetadn_up + dtheta0dzeta * dzetadn_up)
    dfxc1b = (
        dfxc1dtheta1 * dtheta1dtheta0 * (dtheta0dtheta * dthetadn_up + dtheta0dzeta * dzetadn_up)
    )
    dphi = dndn_up * dphidrs * drsdn + dphidtheta * dthetadn_up + dphidzeta * dzetadn_up
    dfxcdn_up = dfxc0a + dfxc0b - phi * (dfxc0a + dfxc0b - dfxc1a - dfxc1b) - (fxc0 - fxc1) * dphi

    dndn_dw = 1
    dfxc0a = dfxc0drs * dndn_dw * drsdn
    dfxc1a = dfxc1drs * dndn_dw * drsdn
    dfxc0b = dfxc0dtheta0 * (dtheta0dtheta * dthetadn_dw + dtheta0dzeta * dzetadn_dw)
    dfxc1b = (
        dfxc1dtheta1 * dtheta1dtheta0 * (dtheta0dtheta * dthetadn_dw + dtheta0dzeta * dzetadn_dw)
    )
    dphi = dndn_dw * dphidrs * drsdn + dphidtheta * dthetadn_dw + dphidzeta * dzetadn_dw
    dfxcdn_dw = dfxc0a + dfxc0b - phi * (dfxc0a + dfxc0b - dfxc1a - dfxc1b) - (fxc0 - fxc1) * dphi
    return dfxcdn_up, dfxcdn_dw


### fxc_zeta


def get_fxc_zeta(rs, theta, p):
    return (
        -1
        / rs
        * (p.omega * p.a(theta) + np.sqrt(rs) * p.b(theta) + rs * p.c(theta))
        / (1 + np.sqrt(rs) * p.d(theta) + rs * p.e(theta))
    )


def get_dfxc_zetadrs(rs, theta, p):  # TODO
    tmp1 = (-p.b(theta) / (2 * np.sqrt(rs)) - p.c(theta)) / (
        rs * (p.d(theta) * np.sqrt(rs) + p.e(theta) * rs + 1)
    )
    tmp2 = (-p.d(theta) / (2 * np.sqrt(rs)) - p.e(theta)) * (
        -p.a(theta) * p.omega - p.b(theta) * np.sqrt(rs) - p.c(theta) * rs
    )
    tmp3 = rs * (p.d(theta) * np.sqrt(rs) + p.e(theta) * rs + 1) ** 2
    tmp4 = tmp2 / tmp3
    tmp5 = (-p.a(theta) * p.omega - p.b(theta) * np.sqrt(rs) - p.c(theta) * rs) / (
        rs**2 * (p.d(theta) * np.sqrt(rs) + p.e(theta) * rs + 1)
    )
    return tmp1 + tmp4 - tmp5


def get_dfxc_zetadtheta(rs, theta, p):
    tmp1 = (-np.sqrt(rs) * p.dddtheta(theta) - rs * p.dedtheta(theta)) * (
        -p.omega * p.a(theta) - p.b(theta) * np.sqrt(rs) - p.c(theta) * rs
    )
    tmp2 = (p.d(theta) * np.sqrt(rs) + p.e(theta) * rs + 1) ** 2 * rs
    tmp3 = tmp1 / tmp2
    tmp4 = -p.omega * p.dadtheta(theta) - np.sqrt(rs) * p.dbdtheta(theta) - rs * p.dcdtheta(theta)
    tmp5 = (p.d(theta) * np.sqrt(rs) + p.e(theta) * rs + 1) * rs
    tmp6 = tmp4 / tmp5
    return tmp3 + tmp6


## alpha


def get_alpha(rs, theta, p):
    h = get_h(rs, p)
    lam = get_lambda(rs, theta, p)
    return 2 - h * np.exp(-theta * lam)


def get_dalphadrs(rs, theta, p):
    h = get_h(rs, p)
    lam = get_lambda(rs, theta, p)
    dhdrs = get_dhdrs(rs, p)
    dlamdrs = get_dlambdadrs(rs, theta, p)
    return -dhdrs * np.exp(-theta * lam) + dlamdrs * theta * h * np.exp(-theta * lam)


def get_dalphadtheta(rs, theta, p):
    h = get_h(rs, p)
    lam = get_lambda(rs, theta, p)
    dlamdtheta = get_dlambdadtheta(rs, p)
    return -(-dlamdtheta * theta - lam) * h * np.exp(-theta * lam)


### h


def get_h(rs, p):
    return (2 / 3 + p.h1 * rs) / (1 + p.h2 * rs)


def get_dhdrs(rs, p):
    return p.h1 / (p.h2 * rs + 1) - p.h2 * (p.h1 * rs + 2 / 3) / (p.h2 * rs + 1) ** 2


### lambda


def get_lambda(rs, theta, p):
    return p.lambda1 + p.lambda2 * theta * rs ** (1 / 2)


def get_dlambdadrs(rs, theta, p):
    return p.lambda2 * theta / (2 * np.sqrt(rs))


def get_dlambdadtheta(rs, p):
    return p.lambda2 * np.sqrt(rs)


### phi


def get_phi(rs, theta, zeta, p):
    alpha = get_alpha(rs, theta, p)
    return ((1 + zeta) ** alpha + (1 - zeta) ** alpha - 2) / (2**alpha - 2)


def get_dphidrs(rs, theta, zeta, p):
    thres = 1e-15
    alpha = get_alpha(rs, theta, p)
    tmp1 = (1 - zeta) ** alpha * np.log(1 - zeta + thres)
    tmp2 = (zeta + 1) ** alpha * np.log(zeta + 1)
    duv = (tmp1 + tmp2) * (2**alpha - 2)
    udv = ((1 - zeta) ** alpha + (zeta + 1) ** alpha - 2) * 2**alpha * np.log(2)
    vv = (2**alpha - 2) ** 2
    dalphadrs = get_dalphadrs(rs, theta, p)
    return (duv - udv) * dalphadrs / vv


def get_dphidtheta(rs, theta, zeta, p):
    thres = 1e-15
    alpha = get_alpha(rs, theta, p)
    dalphadtheta = get_dalphadtheta(rs, theta, p)
    tmp1 = (1 - zeta) ** alpha * np.log(1 - zeta + thres)
    tmp2 = (zeta + 1) ** alpha * np.log(zeta + 1)
    duv = (tmp1 + tmp2) * (2**alpha - 2)
    udv = ((1 - zeta) ** alpha + (zeta + 1) ** alpha - 2) * 2**alpha * np.log(2)
    vv = (2**alpha - 2) ** 2
    dalphadtheta = get_dalphadtheta(rs, theta, p)
    return (duv - udv) * dalphadtheta / vv


def get_dphidzeta(rs, theta, zeta, p):
    alpha = get_alpha(rs, theta, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        tmp1 = alpha * (zeta + 1) ** alpha / (zeta + 1)
        tmp2 = alpha * (1 - zeta) ** alpha / (1 - zeta)
        dphidzeta = (tmp1 - tmp2) / (2**alpha - 2)
    return np.nan_to_num(dphidzeta, nan=0, posinf=0, neginf=0)


### theta


def get_theta(T, n, zeta):
    """Calculate the reduced temperature.

    Reference: Phys. Rev. Lett. 112, 076403.

    Args:
        T: Absolute temperature in Hartree.
        n: Real-space electronic temperature.
        zeta: Relative spin polarization.

    Returns:
        Reduced temperature.
    """
    n_up = (1 + zeta) * n / 2
    T_fermi = (6 * np.pi**2 * n_up) ** (2 / 3) / 2
    return T / T_fermi


def get_dthetadn_up(T, n_up):
    return -4 / (3 * 6 ** (2 / 3)) * T / (np.pi ** (4 / 3) * n_up ** (5 / 3))


def get_theta0(theta, zeta):
    return theta * (1 + zeta) ** (2 / 3)


def get_dtheta0dtheta(zeta):
    return (zeta + 1) ** (2 / 3)


def get_dtheta0dzeta(theta, zeta):
    return 2 * theta / (3 * (zeta + 1) ** (1 / 3))


def get_theta1(theta, zeta):
    theta0 = get_theta0(theta, zeta)
    return theta0 * 2 ** (-2 / 3)
