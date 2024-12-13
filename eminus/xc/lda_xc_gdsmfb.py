# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""GDSMFB LDA exchange-correlation.

Reference: Phys. Rev. Lett. 119, 135001.
"""

import numpy as np


class Parameters:
    """Parameters class.

    Holds parameters of exchange-correlation functionals.
    Saving same space through attribute access.
    """

    def __init__(self, params):
        """Intitialize an instance of this class."""
        for key, val in params.items():
            setattr(self, key, val)


def get_b5(omega, b3):
    """Get b5."""
    return b3 * np.sqrt(3 / 2) * omega * (4 / (9 * np.pi)) ** (-1 / 3)


def get_gdsmfb_parameters():
    """Get the GDSMFB parameters."""
    # zeta = 0
    p_zeta0 = {}
    p_zeta0["omega"] = 1
    p_zeta0["b1"] = 0.3436902
    p_zeta0["b2"] = 7.82159531356
    p_zeta0["b3"] = 0.300483986662
    p_zeta0["b4"] = 15.8443467125
    p_zeta0["b5"] = get_b5(p_zeta0["omega"], p_zeta0["b3"])
    p_zeta0["c1"] = 0.8759442
    p_zeta0["c2"] = -0.230130843551
    p_zeta0["d1"] = 0.72700876
    p_zeta0["d2"] = 2.38264734144
    p_zeta0["d3"] = 0.30221237251
    p_zeta0["d4"] = 4.39347718395
    p_zeta0["d5"] = 0.729951339845
    p_zeta0["e1"] = 0.25388214
    p_zeta0["e2"] = 0.815795138599
    p_zeta0["e3"] = 0.0646844410481
    p_zeta0["e4"] = 15.0984620477
    p_zeta0["e5"] = 0.230761357474
    p0 = Parameters(p_zeta0)

    # zeta = 1
    p_zeta1 = {}
    p_zeta1["omega"] = 2 ** (1 / 3)
    p_zeta1["b1"] = 0.84987704
    p_zeta1["b2"] = 3.04033012073
    p_zeta1["b3"] = 0.0775730131248
    p_zeta1["b4"] = 7.57703592489
    p_zeta1["b5"] = get_b5(p_zeta1["omega"], p_zeta1["b3"])
    p_zeta1["c1"] = 0.91126873
    p_zeta1["c2"] = -0.0307957123308
    p_zeta1["d1"] = 1.48658718
    p_zeta1["d2"] = 4.92684905511
    p_zeta1["d3"] = 0.0849387225179
    p_zeta1["d4"] = 8.3269821188
    p_zeta1["d5"] = 0.218864952126
    p_zeta1["e1"] = 0.27454097
    p_zeta1["e2"] = 0.400994856555
    p_zeta1["e3"] = 2.88773194962
    p_zeta1["e4"] = 6.33499237092
    p_zeta1["e5"] = 24.823008753
    p1 = Parameters(p_zeta1)

    # spin interpolation
    p_spin = {}
    # Sign of parameters is different as in the supp. mat.
    p_spin["h1"] = 3.18747258
    p_spin["h2"] = 7.74662802
    p_spin["lambda1"] = 1.85909536
    p_spin["lambda2"] = 0
    p2 = Parameters(p_spin)

    return p0, p1, p2


def get_a(theta):
    """Get a."""
    tmp1 = 0.610887 * np.tanh(1 / theta)
    tmp2 = 0.75 + 3.04363 * theta**2 - 0.09227 * theta**3 + 1.7035 * theta**4
    tmp3 = 1 + 8.31051 * theta**2 + 5.1105 * theta**4
    return tmp1 * tmp2 / tmp3


def get_dadtheta(theta):
    """Get da / dtheta."""
    tmp1 = -0.00884515668249876 * (20.442 * theta**3 + 16.62102 * theta)
    tmp2 = 1.7035 * theta**4 - 0.09227 * theta**3 + 3.04363 * theta**2 + 0.75
    tmp3 = np.tanh(1 / theta) / (0.614944209200157 * theta**4 + theta**2 + 0.12032955859508) ** 2
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


def get_b(theta, b1, b2, b3, b4, b5):
    """Get b."""
    tmp1 = np.tanh(1 / np.sqrt(theta)) * (b1 + b2 * theta**2 + b3 * theta**4)
    tmp2 = 1 + b4 * theta**2 + b5 * theta**4
    return tmp1 / tmp2


def get_dbdtheta(theta, b1, b2, b3, b4, b5):
    """Get db / dtheta."""
    tmp1 = (
        (2 * b2 * theta + 4 * b3 * theta**3)
        * np.tanh(1 / np.sqrt(theta))
        / (b4 * theta**2 + b5 * theta**4 + 1)
    )
    tmp11 = (
        (2 * b4 * theta + 4 * b5 * theta**3)
        * (b1 + b2 * theta**2 + b3 * theta**4)
        * np.tanh(1 / np.sqrt(theta))
    )
    tmp12 = (b4 * theta**2 + b5 * theta**4 + 1) ** 2
    tmp2 = tmp11 / tmp12
    with np.errstate(over="ignore"):
        tmp21 = b1 + b2 * theta**2 + b3 * theta**4
        tmp22 = (
            2
            * (b4 * theta**2 + b5 * theta**4 + 1)
            * theta ** (3 / 2)
            * np.cosh(1 / np.sqrt(theta)) ** 2
        )
        tmp23 = tmp21 / tmp22
        tmp3 = np.where(theta < 0.001, 0, tmp23)
    return tmp1 - tmp2 - tmp3


def get_e(theta, e1, e2, e3, e4, e5):
    """Get e."""
    return (
        np.tanh(1 / theta)
        * (e1 + e2 * theta**2 + e3 * theta**4)
        / (1 + e4 * theta**2 + e5 * theta**4)
    )


def get_dedtheta(theta, e1, e2, e3, e4, e5):
    """Get de / dtheta."""
    tmp1 = (
        (2 * e2 * theta + 4 * e3 * theta**3)
        * np.tanh(1 / theta)
        / (e4 * theta**2 + e5 * theta**4 + 1)
    )
    tmp11 = (
        (2 * e4 * theta + 4 * e5 * theta**3)
        * (e1 + e2 * theta**2 + e3 * theta**4)
        * np.tanh(1 / theta)
    )
    tmp12 = (e4 * theta**2 + e5 * theta**4 + 1) ** 2
    tmp2 = tmp11 / tmp12
    with np.errstate(over="ignore"):
        tmp21 = e1 + e2 * theta**2 + e3 * theta**4
        tmp22 = (e4 * theta**2 + e5 * theta**4 + 1) * theta**2 * np.cosh(1 / theta) ** 2
        tmp23 = tmp21 / tmp22
        tmp3 = np.where(theta < 0.0025, 0, tmp23)
    return tmp1 - tmp2 - tmp3


def get_c(theta, c1, c2, e1, e2, e3, e4, e5):
    """Get c."""
    thres = 1e-6
    e = get_e(theta, e1, e2, e3, e4, e5)
    return np.where(theta > thres, (c1 + c2 * np.exp(-1 / theta)) * e, c1 * e)


def get_dcdtheta(theta, c1, c2, e1, e2, e3, e4, e5):
    """Get dc / dtheta."""
    e = get_e(theta, e1, e2, e3, e4, e5)
    tmp1 = c2 * e * np.exp(-1 / theta) / theta**2
    tmp2 = c1 + c2 * np.exp(-1 / theta)
    tmp3 = (
        (2 * e2 * theta + 4 * e3 * theta**3)
        * np.tanh(1 / theta)
        / (e4 * theta**2 + e5 * theta**4 + 1)
    )
    tmp31 = (
        (2 * e4 * theta + 4 * e5 * theta**3)
        * (e1 + e2 * theta**2 + e3 * theta**4)
        * np.tanh(1 / theta)
    )
    tmp32 = (e4 * theta**2 + e5 * theta**4 + 1) ** 2
    tmp4 = tmp31 / tmp32
    with np.errstate(over="ignore"):
        tmp41 = e1 + e2 * theta**2 + e3 * theta**4
        tmp42 = (e4 * theta**2 + e5 * theta**4 + 1) * theta**2 * np.cosh(1 / theta) ** 2
        tmp43 = tmp41 / tmp42
        tmp5 = np.where(theta < 0.0025, 0, tmp43)
    return tmp1 + tmp2 * (tmp3 - tmp4 - tmp5)


def get_d(theta, d1, d2, d3, d4, d5):
    """Get d."""
    tmp1 = np.tanh(1 / np.sqrt(theta)) * (d1 + d2 * theta**2 + d3 * theta**4)
    tmp2 = 1 + d4 * theta**2 + d5 * theta**4
    return tmp1 / tmp2


def get_dddtheta(theta, d1, d2, d3, d4, d5):
    """Get dd / dtheta."""
    tmp1 = (
        (2 * d2 * theta + 4 * d3 * theta**3)
        * np.tanh(1 / np.sqrt(theta))
        / (d4 * theta**2 + d5 * theta**4 + 1)
    )
    tmp11 = (
        (2 * d4 * theta + 4 * d5 * theta**3)
        * (d1 + d2 * theta**2 + d3 * theta**4)
        * np.tanh(1 / np.sqrt(theta))
    )
    tmp12 = (d4 * theta**2 + d5 * theta**4 + 1) ** 2
    tmp2 = tmp11 / tmp12
    with np.errstate(over="ignore"):
        tmp21 = d1 + d2 * theta**2 + d3 * theta**4
        tmp22 = (
            2
            * (d4 * theta**2 + d5 * theta**4 + 1)
            * theta ** (3 / 2)
            * np.cosh(1 / np.sqrt(theta)) ** 2
        )
        tmp23 = tmp21 / tmp22
        tmp3 = np.where(theta < 0.001, 0, tmp23)
    return tmp1 - tmp2 - tmp3


def get_fxc_zeta_params(rs, omega, a, b, c, d, e):
    """Get fxc_zeta with explict parameters."""
    return -1 / rs * (omega * a + np.sqrt(rs) * b + rs * c) / (1 + np.sqrt(rs) * d + rs * e)


def get_dfxc_zeta_paramsdtheta(
    rs, omega, a, b, c, d, e, dadtheta, dbdtheta, dcdtheta, dddtheta, dedtheta
):
    """Get dfxc_zeta / dtheta using explict parameters."""
    tmp1 = (-np.sqrt(rs) * dddtheta - rs * dedtheta) * (-omega * a - b * np.sqrt(rs) - c * rs)
    tmp2 = (d * np.sqrt(rs) + e * rs + 1) ** 2 * rs
    tmp3 = tmp1 / tmp2
    tmp4 = -omega * dadtheta - np.sqrt(rs) * dbdtheta - rs * dcdtheta
    tmp5 = (d * np.sqrt(rs) + e * rs + 1) * rs
    tmp6 = tmp4 / tmp5
    return tmp3 + tmp6


def get_fxc_zeta(rs, theta, p):
    """Get fxc_zeta using a parameters object."""
    a = get_a(theta)
    b = get_b(theta, p.b1, p.b2, p.b3, p.b4, p.b5)
    e = get_e(theta, p.e1, p.e2, p.e3, p.e4, p.e5)
    c = get_c(theta, p.c1, p.c2, p.e1, p.e2, p.e3, p.e4, p.e5)
    d = get_d(theta, p.d1, p.d2, p.d3, p.d4, p.d5)
    return get_fxc_zeta_params(rs, p.omega, a, b, c, d, e)


def get_dfxc_zetadtheta(rs, theta, p):
    """Get dfxc / dtheta."""
    a = get_a(theta)
    b = get_b(theta, p.b1, p.b2, p.b3, p.b4, p.b5)
    e = get_e(theta, p.e1, p.e2, p.e3, p.e4, p.e5)
    c = get_c(theta, p.c1, p.c2, p.e1, p.e2, p.e3, p.e4, p.e5)
    d = get_d(theta, p.d1, p.d2, p.d3, p.d4, p.d5)
    dadt = get_dadtheta(theta)
    dbdt = get_dbdtheta(theta, p.b1, p.b2, p.b3, p.b4, p.b5)
    dcdt = get_dcdtheta(theta, p.c1, p.c2, p.e1, p.e2, p.e3, p.e4, p.e5)
    dddt = get_dddtheta(theta, p.d1, p.d2, p.d3, p.d4, p.d5)
    dedt = get_dedtheta(theta, p.e1, p.e2, p.e3, p.e4, p.e5)
    return get_dfxc_zeta_paramsdtheta(rs, p.omega, a, b, c, d, e, dadt, dbdt, dcdt, dddt, dedt)


def get_theta0(theta, zeta):
    """Get theta0."""
    return theta * (1 + zeta) ** (2 / 3)


def get_theta1(theta, zeta):
    """Get theta1."""
    theta0 = get_theta0(theta, zeta)
    return theta0 * 2 ** (-2 / 3)


def get_lambda(rs, theta, p):
    """Get lambda."""
    return p.lambda1 + p.lambda2 * theta * rs ** (1 / 2)


def get_h(rs, p):
    """Get h."""
    return (2 / 3 + p.h1 * rs) / (1 + p.h2 * rs)


def get_alpha(rs, theta, p):
    """Get alpha."""
    h = get_h(rs, p)
    lam = get_lambda(rs, theta, p)
    return 2 - h * np.exp(-theta * lam)


def get_phi(rs, theta, zeta, p):
    """Get phi from rs, theta, and zeta."""
    alpha = get_alpha(rs, theta, p)
    return ((1 + zeta) ** alpha + (1 - zeta) ** alpha - 2) / (2**alpha - 2)


def get_phi_T(rs, T, zeta, p):
    """Get phi from rs, T, and zeta."""
    n = get_n(rs)
    theta = get_theta(T, n, zeta)
    alpha = get_alpha(rs, theta, p)
    return ((1 + zeta) ** alpha + (1 - zeta) ** alpha - 2) / (2**alpha - 2)


def get_fxc0(rs, theta, zeta):
    """Get fxc0."""
    p0, _, _ = get_gdsmfb_parameters()
    theta0 = get_theta0(theta, zeta)
    return get_fxc_zeta(rs, theta0, p0)


def get_fxc1(rs, theta, zeta):
    """Get fxc1."""
    _, p1, _ = get_gdsmfb_parameters()
    theta1 = get_theta1(theta, zeta)
    return get_fxc_zeta(rs, theta1, p1)


def get_fxc(rs, theta, zeta, p0, p1, p2):
    """Get fxc utilizing rs,zeta, and theta."""
    theta0 = get_theta0(theta, zeta)
    theta1 = get_theta1(theta, zeta)
    fxc0 = get_fxc_zeta(rs, theta0, p0)
    fxc1 = get_fxc_zeta(rs, theta1, p1)
    phi = get_phi(rs, theta0, zeta, p2)
    return fxc0 + (fxc1 - fxc0) * phi


def get_fxc_nupndn(nup, ndn, T, p0, p1, p2):
    """Get fxc utilizing nup, ndn, and T."""
    n = nup + ndn
    zeta = (nup - ndn) / n
    rs = get_rs_from_n(n)
    theta = get_theta(T, n, zeta)
    return get_fxc(rs, theta, zeta, p0, p1, p2)


def get_zeta(nup, ndn):
    """Get zeta from nup and ndn."""
    return (nup - ndn) / (nup + ndn)


def get_dzetadnup(nup, ndn):
    """Get dzeta / dnup."""
    return -(-ndn + nup) / (ndn + nup) ** 2 + 1 / (ndn + nup)


def get_dzetadndn(nup, ndn):
    """Get dzeta / dndn."""
    return -(-ndn + nup) / (ndn + nup) ** 2 - 1 / (ndn + nup)


def get_drsdn(n):
    """Get drs / dn."""
    return -(6 ** (1 / 3)) * (1 / n) ** (1 / 3) / (6 * np.pi ** (1 / 3) * n)


def get_dtheta0dtheta(zeta):
    """Get dtheta0 / dtheta."""
    return (zeta + 1) ** (2 / 3)


def get_dtheta0dzeta(theta, zeta):
    """Get dtheta0 / dzeta."""
    return 2 * theta / (3 * (zeta + 1) ** (1 / 3))


def get_theta_nup(T, nup):
    """Get theta from T and nup."""
    k_fermi_sq = (6 * np.pi**2 * nup) ** (2 / 3)
    T_fermi = 1 / 2 * k_fermi_sq
    return T / T_fermi


def get_dthetadnup(T, nup):
    """Get dtheta / dnup."""
    return -0.40380457618492 * T / (np.pi ** (4 / 3) * nup ** (5 / 3))


def get_dfxc_zeta_paramsdrs(rs, omega, a, b, c, d, e):
    """Get dfxc / drs with explicit parameters."""
    tmp1 = (-b / (2 * np.sqrt(rs)) - c) / (rs * (d * np.sqrt(rs) + e * rs + 1))
    tmp2 = (-d / (2 * np.sqrt(rs)) - e) * (-a * omega - b * np.sqrt(rs) - c * rs)
    tmp3 = rs * (d * np.sqrt(rs) + e * rs + 1) ** 2
    tmp4 = tmp2 / tmp3
    tmp5 = (-a * omega - b * np.sqrt(rs) - c * rs) / (rs**2 * (d * np.sqrt(rs) + e * rs + 1))
    return tmp1 + tmp4 - tmp5


def get_dfxc_zetadrs(rs, theta, p):
    """Get dfxc / drs utilizing a parameters object."""
    a = get_a(theta)
    b = get_b(theta, p.b1, p.b2, p.b3, p.b4, p.b5)
    e = get_e(theta, p.e1, p.e2, p.e3, p.e4, p.e5)
    c = get_c(theta, p.c1, p.c2, p.e1, p.e2, p.e3, p.e4, p.e5)
    d = get_d(theta, p.d1, p.d2, p.d3, p.d4, p.d5)
    return get_dfxc_zeta_paramsdrs(rs, p.omega, a, b, c, d, e)


def get_dfxcdnup_params(nup, ndn, T, p0, p1, p2):
    """Get dfxc / dnup."""
    n = nup + ndn
    zeta = (nup - ndn) / n
    rs = get_rs_from_n(n)
    theta = get_theta(T, n, zeta)
    theta0 = get_theta0(theta, zeta)
    fxc0 = get_fxc_zeta(rs, theta0, p0)
    theta1 = get_theta1(theta, zeta)
    fxc1 = get_fxc_zeta(rs, theta1, p1)
    phi = get_phi(rs, theta0, zeta, p2)

    dndnup = 1
    dzetadnup = get_dzetadnup(nup, ndn)
    drsdn = get_drsdn(n)
    dfxc0drs = get_dfxc_zetadrs(rs, theta, p0)
    dfxc1drs = get_dfxc_zetadrs(rs, theta, p1)

    dfxc0dtheta0 = get_dfxc_zetadtheta(rs, theta0, p0)
    dfxc1dtheta1 = get_dfxc_zetadtheta(rs, theta1, p1)

    dtheta0dtheta = get_dtheta0dtheta(zeta)
    dthetadnup = get_dthetadnup(T, nup)
    dtheta0dzeta = get_dtheta0dzeta(theta0, zeta)
    dtheta1dtheta0 = 2 ** (-2 / 3)

    dphidrs = get_dphidrs(rs, theta0, zeta, p2)
    dphidtheta = get_dphidtheta(rs, theta0, zeta, p2)
    dphidzeta = get_dphidzeta(rs, theta0, zeta, p2)

    dfxc0a = dfxc0drs * dndnup * drsdn
    dfxc1a = dfxc1drs * dndnup * drsdn
    dfxc0b = dfxc0dtheta0 * (dtheta0dtheta * dthetadnup + dtheta0dzeta * dzetadnup)
    dfxc1b = dfxc1dtheta1 * dtheta1dtheta0 * (dtheta0dtheta * dthetadnup + dtheta0dzeta * dzetadnup)
    dphi = dndnup * dphidrs * drsdn + dphidtheta * dthetadnup + dphidzeta * dzetadnup
    return dfxc0a + dfxc0b - phi * (dfxc0a + dfxc0b - dfxc1a - dfxc1b) - (fxc0 - fxc1) * dphi


def get_dfxcdndn_params(nup, ndn, T, p0, p1, p2):
    """Get dfxc / dndn."""
    n = nup + ndn
    zeta = (nup - ndn) / n
    rs = get_rs_from_n(n)
    theta = get_theta(T, n, zeta)
    theta0 = get_theta0(theta, zeta)
    fxc0 = get_fxc_zeta(rs, theta0, p0)
    theta1 = get_theta1(theta, zeta)
    fxc1 = get_fxc_zeta(rs, theta1, p1)
    phi = get_phi(rs, theta0, zeta, p2)

    dndndn = 1
    dzetadndn = get_dzetadndn(nup, ndn)
    drsdn = get_drsdn(n)
    dfxc0drs = get_dfxc_zetadrs(rs, theta, p0)
    dfxc1drs = get_dfxc_zetadrs(rs, theta, p1)

    dfxc0dtheta0 = get_dfxc_zetadtheta(rs, theta0, p0)
    dfxc1dtheta1 = get_dfxc_zetadtheta(rs, theta1, p1)

    dtheta0dtheta = get_dtheta0dtheta(zeta)
    dthetadndn = 0
    dtheta0dzeta = get_dtheta0dzeta(theta0, zeta)
    dtheta1dtheta0 = 2 ** (-2 / 3)

    dphidrs = get_dphidrs(rs, theta0, zeta, p2)
    dphidtheta = get_dphidtheta(rs, theta0, zeta, p2)
    dphidzeta = get_dphidzeta(rs, theta0, zeta, p2)

    dfxc0a = dfxc0drs * dndndn * drsdn
    dfxc1a = dfxc1drs * dndndn * drsdn
    dfxc0b = dfxc0dtheta0 * (dtheta0dtheta * dthetadndn + dtheta0dzeta * dzetadndn)
    dfxc1b = dfxc1dtheta1 * dtheta1dtheta0 * (dtheta0dtheta * dthetadndn + dtheta0dzeta * dzetadndn)
    dphi = dndndn * dphidrs * drsdn + dphidtheta * dthetadndn + dphidzeta * dzetadndn
    return dfxc0a + dfxc0b - phi * (dfxc0a + dfxc0b - dfxc1a - dfxc1b) - (fxc0 - fxc1) * dphi


def get_rs_from_n(n):
    """Get rs from n."""
    return (3 / (4 * np.pi * n)) ** (1 / 3)


def get_n(rs):
    """Get n from rs."""
    return 1 / (4 * np.pi / 3 * rs**3)


def get_dhdrs(rs, p):
    """Get dh / drs."""
    return p.h1 / (p.h2 * rs + 1) - p.h2 * (p.h1 * rs + 2 / 3) / (p.h2 * rs + 1) ** 2


def get_dlamdrs(rs, theta, p):
    """Get dlam / drs."""
    return p.lambda2 * theta / (2 * np.sqrt(rs))


def get_dalphadrs(rs, theta, p):
    """Get dalpha / drs."""
    h = get_h(rs, p)
    lam = get_lambda(rs, theta, p)
    dhdrs = get_dhdrs(rs, p)
    dlamdrs = get_dlamdrs(rs, theta, p)
    return -dhdrs * np.exp(-theta * lam) + dlamdrs * theta * h * np.exp(-theta * lam)


def get_dlamdtheta(rs, p):
    """Get dlam / dtheta."""
    return p.lambda2 * np.sqrt(rs)


def get_dalphadtheta(rs, theta, p):
    """Get dalpha / dtheta."""
    h = get_h(rs, p)
    lam = get_lambda(rs, theta, p)
    dlamdtheta = get_dlamdtheta(rs, p)
    return -(-dlamdtheta * theta - lam) * h * np.exp(-theta * lam)


def get_zeta_rs(rs, nup, ndn):
    """Get zeta from rs, nup, and ndn."""
    return (nup - ndn) / (get_n(rs))


def get_dzetadrs(rs, nup, ndn):
    """Get dzeta / drs."""
    return 4 * np.pi * (-ndn + nup) * rs**2


def get_dphidrs(rs, theta, zeta, p):
    """Get dphi / drs."""
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
    """Get dphi / dtheta."""
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
    """Get dphi / dzeta."""
    alpha = get_alpha(rs, theta, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        tmp1 = alpha * (zeta + 1) ** alpha / (zeta + 1)
        tmp2 = alpha * (1 - zeta) ** alpha / (1 - zeta)
        dphidzeta = (tmp1 - tmp2) / (2**alpha - 2)
    return np.nan_to_num(dphidzeta, nan=0, posinf=0, neginf=0)


def get_theta(T, n, zeta):
    """Reduced temperature.

    Calculates the reduced temperature

    theta = T / T_Fermi

    Reference: Phys. Rev. Lett. 119, 135001.

    Args:
        T: Absolute temperature in Hartree.
        n: Real-space electronic temperature.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Reduced temperature.
    """
    n_up = 0.5 * n * (1 + zeta)
    k_fermi_sq = (6 * np.pi**2 * n_up) ** (2 / 3)
    T_fermi = 1 / 2 * k_fermi_sq
    return T / T_fermi


def get_T(theta, n, zeta):
    """Absolute temperature.

    Calculates the absolute temperature

    T = theta * T_Fermi

    Reference: Phys. Rev. Lett. 119, 135001.

    Args:
        theta: reduced temperature.
        n: Real-space electronic temperature.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Reduced temperature.
    """
    n_up = 0.5 * n * (1 + zeta)
    k_fermi_sq = (6 * np.pi**2 * n_up) ** (2 / 3)
    T_fermi = 1 / 2 * k_fermi_sq
    return theta * T_fermi


def lda_xc_gdsmfb(n, T=0, **kwargs):
    """GDSMFB exchange-correlation functional (spin-paired).

    Exchange and correlation connot be separated.

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

    Exchange and correlation connot be separated.

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
    ndn = n * (1 - zeta) / 2
    nup = n * (1 + zeta) / 2
    rs = get_rs_from_n(n)
    theta = get_theta(T, n, zeta)

    p0, p1, p2 = get_gdsmfb_parameters()
    fxc = get_fxc(rs, theta, zeta, p0, p1, p2)
    dfxcdnup = get_dfxcdnup_params(nup, ndn, T, p0, p1, p2)
    dfxcdndn = get_dfxcdndn_params(nup, ndn, T, p0, p1, p2)
    return fxc, np.array([dfxcdnup, dfxcdndn]) * n + fxc, None
