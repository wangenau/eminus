# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Perdew-Wang LDA correlation.

Reference: Phys. Rev. B 45, 13244.
"""

import numpy as np


def lda_c_pw(n, A=0.031091, a1=0.2137, b1=7.5957, b2=3.5876, b3=1.6382, b4=0.49294, **kwargs):
    """Perdew-Wang parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_PW and ID 12 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        A: Functional parameter.
        a1: Functional parameter.
        b1: Functional parameter.
        b2: Functional parameter.
        b3: Functional parameter.
        b4: Functional parameter.
        **kwargs: Throwaway arguments.

    Returns:
        PW correlation energy density and potential.
    """
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    rs12 = np.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    om = 2 * A * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    olog = np.log(1 + 1 / om)
    ec = -2 * A * (1 + a1 * rs) * olog

    dom = 2 * A * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    vc = -2 * A * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * A * (1 + a1 * rs) * dom / (om * (om + 1))
    return ec, np.array([vc]), None


def lda_c_pw_spin(n, zeta, A=(0.031091, 0.015545, 0.016887), fzeta0=1.709921, **kwargs):
    """Perdew-Wang parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW and ID 12 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        A: Functional parameters.
        fzeta0: Functional parameter.
        **kwargs: Throwaway arguments.

    Returns:
        PW correlation energy density and potential.
    """
    a1 = (0.2137, 0.20548, 0.11125)
    b1 = (7.5957, 14.1189, 10.357)
    b2 = (3.5876, 6.1977, 3.6231)
    b3 = (1.6382, 3.3662, 0.88026)
    b4 = (0.49294, 0.62517, 0.49671)

    ec0, vc0, _ = lda_c_pw(n, A[0], a1[0], b1[0], b2[0], b3[0], b4[0])  # Unpolarized
    ec1, vc1, _ = lda_c_pw(n, A[1], a1[1], b1[1], b2[1], b3[1], b4[1])  # Polarized
    ac, dac, _ = lda_c_pw(n, A[2], a1[2], b1[2], b2[2], b3[2], b4[2])  # Spin stiffness
    ac = -ac  # The PW spin interpolation is parametrized with -ac instead of ac
    vc0, vc1, dac = vc0[0], vc1[0], dac[0]  # Remove spin dimension for the correct shape

    fzeta = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    zeta3 = zeta**3
    zeta4 = zeta3 * zeta

    ec = ec0 + ac * fzeta * (1 - zeta4) / fzeta0 + (ec1 - ec0) * fzeta * zeta4

    dac = -dac  # Also change the sign for the derivative
    dfzeta = ((1 + zeta) ** (1 / 3) - (1 - zeta) ** (1 / 3)) * 4 / (3 * (2 ** (4 / 3) - 2))
    factor1 = vc0 + dac * fzeta * (1 - zeta4) / fzeta0 + (vc1 - vc0) * fzeta * zeta4
    factor2 = ac / fzeta0 * (dfzeta * (1 - zeta4) - 4 * fzeta * zeta3) + (ec1 - ec0) * (
        dfzeta * zeta4 + 4 * fzeta * zeta3
    )

    vc_up = factor1 + factor2 * (1 - zeta)
    vc_dw = factor1 - factor2 * (1 + zeta)
    return ec, np.array([vc_up, vc_dw]), None
