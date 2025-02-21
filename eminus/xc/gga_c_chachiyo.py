# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Chachiyo GGA correlation.

Reference: Comput. Theor. Chem. 1172, 112669.
"""

import numpy as np
from scipy.linalg import norm

from .lda_c_chachiyo_mod import chachiyo_scaling_mod as weight_function


def gga_c_chachiyo(n, dn_spin=None, **kwargs):
    """Chachiyo parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label GGA_C_CHACHIYO and ID 309 in Libxc.

    Reference: Comput. Theor. Chem. 1172, 112669.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        Chachiyo correlation energy density, potential and vsigma.
    """
    h = 0.06672632  # 0.5 * 0.00847 * 16 * (3 / np.pi)**(1 / 3)

    # ### Start lda_c_chachiyo_mod ### #
    a = -0.01554535  # (np.log(2) - 1) / (2 * np.pi**2)
    b = 20.4562557

    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    rs2 = rs**2
    ecinner = 1 + b / rs + b / rs2

    ec = a * np.log(ecinner)
    # ### End lda_c_chachiyo_mod ### #

    norm_dn = norm(dn_spin[0], axis=1)
    t = (np.pi / 3) ** (1 / 6) / 4 * norm_dn / n ** (7 / 6)
    t2 = t**2
    gec = 1 + t2
    expgec = gec ** (h / ec)

    # ### Start lda_c_chachiyo_mod ### #
    vc = ec + a * b * (2 + rs) / (3 * (b + b * rs + rs2))
    # ### End lda_c_chachiyo_mod ### #

    term1 = h * (1 - 1 / gec)
    term2 = h * np.log(gec) * (1 - vc / ec)
    gvc = (vc - 7 / 3 * term1 + term2) * expgec

    vsigmac = n * expgec * term1 / norm_dn**2
    return ec * expgec, np.array([gvc]), np.array([vsigmac])


def gga_c_chachiyo_spin(n, zeta, dn_spin=None, **kwargs):
    """Chachiyo parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label GGA_C_CHACHIYO and ID 309 in Libxc.

    Reference: Comput. Theor. Chem. 1172, 112669.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        Chachiyo correlation energy density, potential and vsigma.
    """
    h = 0.06672632  # 0.5 * 0.00847 * 16 * (3 / np.pi)**(1 / 3)

    # ### Start lda_c_chachiyo_spin_mod ### #
    a0 = -0.01554535  # (np.log(2) - 1) / (2 * np.pi**2)
    a1 = -0.007772675  # (np.log(2) - 1) / (4 * np.pi**2)
    b0 = 20.4562557
    b1 = 27.4203609

    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    rs2 = rs**2

    fzeta, dfdzeta = weight_function(zeta)

    ec0inner = 1 + b0 / rs + b0 / rs2
    ec1inner = 1 + b1 / rs + b1 / rs2
    ec0 = a0 * np.log(ec0inner)
    ec1 = a1 * np.log(ec1inner)

    ec = ec0 + (ec1 - ec0) * fzeta
    # ### End lda_c_chachiyo_spin_mod ### #

    norm_dn = norm(dn_spin[0] + dn_spin[1], axis=1)
    t = (np.pi / 3) ** (1 / 6) / 4 * norm_dn / n ** (7 / 6)
    t2 = t**2
    gec = 1 + t2
    expgec = gec ** (h / ec)

    # ### Start lda_c_chachiyo_spin_mod ### #
    factor = -1 / rs2 - 2 / rs**3
    dec0drs = a0 / ec0inner * b0 * factor
    dec1drs = a1 / ec1inner * b1 * factor
    decdrs = dec0drs + (dec1drs - dec0drs) * fzeta
    # prefactor = ec - rs / 3 * decdrs
    decdf = (ec1 - ec0) * dfdzeta

    # vc_up = prefactor + decdf * (1 - zeta)
    # vc_dw = prefactor - decdf * (1 + zeta)
    # ### End lda_c_chachiyo_spin_mod ### #

    dn2 = (
        norm(dn_spin[0], axis=1) ** 2
        + 2 * np.sum(dn_spin[0] * dn_spin[1], axis=1)
        + norm(dn_spin[1], axis=1) ** 2
    )
    ht2divgecdn2 = (1 - 1 / gec) * h / norm_dn**2
    term1 = -7 / 3 * ht2divgecdn2 * dn2
    term2 = 1 - h * np.log(gec) / ec
    prefactor = -decdrs * rs / 3
    gvc = ec + term1 + term2 * prefactor
    gvc_up = gvc + term2 * decdf * (1 - zeta)
    gvc_dw = gvc - term2 * decdf * (1 + zeta)

    vsigma = n * expgec * 2 * ht2divgecdn2
    vsigmac = np.array([0.5 * vsigma, vsigma, 0.5 * vsigma])
    return ec * expgec, np.array([gvc_up, gvc_dw]) * expgec, vsigmac
