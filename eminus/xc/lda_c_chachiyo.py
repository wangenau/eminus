# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Chachiyo LDA correlation.

Reference: J. Chem. Phys. 145, 021101.
"""

import numpy as np


def lda_c_chachiyo(n, **kwargs):
    """Chachiyo parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_CHACHIYO and ID 287 in Libxc.

    Reference: J. Chem. Phys. 145, 021101.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Chachiyo correlation energy density and potential.
    """
    a = -0.01554535  # (np.log(2) - 1) / (2 * np.pi**2)
    b = 20.4562557

    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    rs2 = rs**2
    ecinner = 1 + b / rs + b / rs2

    ec = a * np.log(ecinner)

    vc = ec + a * b * (2 + rs) / (3 * (b + b * rs + rs2))
    return ec, np.array([vc]), None


def chachiyo_scaling(zeta):
    """Weighting factor between the paramagnetic and the ferromagnetic case.

    Reference: J. Chem. Phys. 145, 021101.

    Args:
        zeta: Relative spin polarization.

    Returns:
        Weighting factor and its derivative.
    """
    fzeta = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 * (2 ** (1 / 3) - 1))

    dfdzeta = (2 * (1 - zeta) ** (1 / 3) - 2 * (1 + zeta) ** (1 / 3)) / (3 - 3 * 2 ** (1 / 3))
    return fzeta, dfdzeta


def lda_c_chachiyo_spin(n, zeta, weight_function=chachiyo_scaling, **kwargs):
    """Chachiyo parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_CHACHIYO and ID 287 in Libxc.

    Reference: J. Chem. Phys. 145, 021101.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        weight_function: Functional function.
        **kwargs: Throwaway arguments.

    Returns:
        Chachiyo correlation energy density and potential.
    """
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

    factor = -1 / rs2 - 2 / rs**3
    dec0drs = a0 / ec0inner * b0 * factor
    dec1drs = a1 / ec1inner * b1 * factor
    decdrs = dec0drs + (dec1drs - dec0drs) * fzeta
    prefactor = ec - rs / 3 * decdrs
    decdf = (ec1 - ec0) * dfdzeta

    vc_up = prefactor + decdf * (1 - zeta)
    vc_dw = prefactor - decdf * (1 + zeta)
    return ec, np.array([vc_up, vc_dw]), None
