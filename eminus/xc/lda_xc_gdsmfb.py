# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""GDSMFB LDA exchange-correlation.

Reference: Phys. Rev. Lett. 119, 135001.
"""

import dataclasses

from .lda_xc_ksdt import Coefficients, lda_xc_ksdt, lda_xc_ksdt_spin

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
    c3: float = 1
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
    c3: float = 1
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
    g1: float = 2 / 3
    g2: float = 3.18747258
    g3: float = 7.74662802
    lambda1: float = 1.85909536
    lambda2: float = 0


# ### Functional implementation ###


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
    return lda_xc_ksdt(
        n, T=T, zeta0_coeffs=Zeta0Coeffs, zeta1_coeffs=Zeta1Coeffs, phi_params=PhiParams, **kwargs
    )


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
    return lda_xc_ksdt_spin(
        n,
        zeta,
        T=T,
        zeta0_coeffs=Zeta0Coeffs,
        zeta1_coeffs=Zeta1Coeffs,
        phi_params=PhiParams,
        **kwargs,
    )
