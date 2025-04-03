# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Corrected KSDT LDA exchange-correlation.

The corrected KSDT functional has not been parameterized for spin-polarized calculations.

Reference: Phys. Rev. Lett. 120, 076401.
"""

import dataclasses

from .lda_xc_ksdt import Coefficients, lda_xc_ksdt


@dataclasses.dataclass
class Zeta0Coeffs(Coefficients):
    """Coefficient class using the parameters for zeta=0.

    Reference: Phys. Rev. Lett. 120, 076401.
    """

    omega: float = 1
    b1: float = 0.342554
    b2: float = 9.141315
    b3: float = 0.448483
    b4: float = 18.553096
    c1: float = 0.87513
    c2: float = -0.25632
    c3: float = 0.953988
    d1: float = 0.725917
    d2: float = 2.237347
    d3: float = 0.280748
    d4: float = 4.185911
    d5: float = 0.692183
    e1: float = 0.255415
    e2: float = 0.931933
    e3: float = 0.115398
    e4: float = 17.234117
    e5: float = 0.451437


def lda_xc_corr_ksdt(n, T=0, **kwargs):
    """Corrected KSDT exchange-correlation functional (spin-paired).

    Similar to the functional with the label LDA_XC_CORRKSDT and ID 318 in Libxc, but the
    implementations differ: https://gitlab.com/libxc/libxc/-/issues/525
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 120, 076401.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        T: Temperature in Hartree.
        **kwargs: Throwaway arguments.

    Returns:
        Corrected KSDT exchange-correlation energy density and potential.
    """
    return lda_xc_ksdt(n, T=T, zeta0_coeffs=Zeta0Coeffs, **kwargs)
