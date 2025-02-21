# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Perdew-Burke-Ernzerhof GGA correlation.

Reference: Phys. Rev. Lett. 77, 3865.
"""

import numpy as np
from scipy.linalg import norm

from .lda_c_pw_mod import lda_c_pw_mod, lda_c_pw_mod_spin


def gga_c_pbe(n, beta=0.06672455060314922, dn_spin=None, **kwargs):
    """Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        beta: Functional parameter.
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE correlation energy density, potential, and vsigma.
    """
    gamma = (1 - np.log(2)) / np.pi**2

    pi34 = (3 / (4 * np.pi)) ** (1 / 3)
    rs = pi34 * n ** (-1 / 3)
    norm_dn = norm(dn_spin[0], axis=1)
    ec, vc, _ = lda_c_pw_mod(n, **kwargs)
    vc = vc[0]  # Remove spin dimension for the correct shape

    kf = (9 / 4 * np.pi) ** (1 / 3) / rs
    ks = np.sqrt(4 * kf / np.pi)
    divt = 2 * ks * n
    t = norm_dn / divt
    expec = np.exp(-ec / gamma)
    A = beta / (gamma * (expec - 1))
    t2 = t**2
    At2 = A * t2
    A2t4 = At2**2
    divsum = 1 + At2 + A2t4
    div = (1 + At2) / divsum
    nolog = 1 + beta / gamma * t2 * div
    gec = gamma * np.log(nolog)

    factor = A2t4 * (2 + At2) / divsum**2
    dgec = beta * t2 / nolog * (-7 / 3 * div - factor * (A * expec * (vc - ec) / beta - 7 / 3))
    gvc = gec + dgec

    vsigmac = beta / (divt * ks) * (div - factor) / nolog
    return ec + gec, np.array([vc + gvc]), np.array([0.5 * vsigmac])


def gga_c_pbe_spin(n, zeta, beta=0.06672455060314922, dn_spin=None, **kwargs):
    """Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        beta: Functional parameter.
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE correlation energy density, potential, and vsigma.
    """
    gamma = (1 - np.log(2)) / np.pi**2

    pi34 = (3 / (4 * np.pi)) ** (1 / 3)
    rs = pi34 * n ** (-1 / 3)
    norm_dn = norm(dn_spin[0] + dn_spin[1], axis=1)
    ec, vc, _ = lda_c_pw_mod_spin(n, zeta, **kwargs)
    vc_up, vc_dw = vc

    kf = (9 / 4 * np.pi) ** (1 / 3) / rs
    ks = np.sqrt(4 * kf / np.pi)
    phi = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2
    phi2 = phi**2
    phi3 = phi2 * phi
    t = norm_dn / (2 * phi * ks * n)
    expec = np.exp(-ec / (gamma * phi3))
    A = beta / (gamma * (expec - 1))
    t2 = t**2
    At2 = A * t2
    A2t4 = At2**2
    divsum = 1 + At2 + A2t4
    div = (1 + At2) / divsum
    nolog = 1 + beta / gamma * t2 * div
    gec = gamma * phi3 * np.log(nolog)

    # Handle divisions by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        dfz = ((1 + zeta) ** (-1 / 3) - (1 - zeta) ** (-1 / 3)) / 3
    dfz = np.nan_to_num(dfz, nan=0, posinf=0, neginf=0)
    factor = A2t4 * (2 + At2) / divsum**2
    bfpre = expec / phi3
    bf_up = bfpre * (vc_up - ec)
    bf_dw = bfpre * (vc_dw - ec)
    dgecpre = beta * t2 * phi3 / nolog
    dgec_up = dgecpre * (-7 / 3 * div - factor * (A * bf_up / beta - 7 / 3))
    dgec_dw = dgecpre * (-7 / 3 * div - factor * (A * bf_dw / beta - 7 / 3))
    dgeczpre = (
        3 * gec / phi
        - beta * t2 * phi2 / nolog * (2 * div - factor * (3 * A * expec * ec / phi3 / beta + 2))
    ) * dfz
    dgecz_up = dgeczpre * (1 - zeta)
    dgecz_dw = -dgeczpre * (1 + zeta)
    gvc_up = gec + dgec_up + dgecz_up
    gvc_dw = gec + dgec_dw + dgecz_dw

    vsigma = beta * phi / (2 * ks * ks * n) * (div - factor) / nolog
    vsigmac = np.array([0.5 * vsigma, vsigma, 0.5 * vsigma])
    return ec + gec, np.array([vc_up + gvc_up, vc_dw + gvc_dw]), vsigmac
