# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""KSDT LDA exchange-correlation.

Reference: Phys. Rev. Lett. 112, 076403.
"""

import dataclasses

from .lda_xc_gdsmfb import Coefficients, lda_xc_gdsmfb, lda_xc_gdsmfb_spin

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


def lda_xc_ksdt(n, T=0, **kwargs):
    """KSDT exchange-correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_XC_KSDT and ID 259 in Libxc.
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 112, 076403.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        T: Temperature in Hartree.
        **kwargs: Throwaway arguments.

    Returns:
        KSDT exchange-correlation energy density and potential.
    """
    return lda_xc_gdsmfb(
        n, T=T, zeta0_coeffs=Zeta0Coeffs, zeta1_coeffs=Zeta1Coeffs, phi_params=PhiParams, **kwargs
    )


def lda_xc_ksdt_spin(n, zeta, T=0, **kwargs):
    """KSDT exchange-correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_XC_KSDT and ID 259 in Libxc.
    Exchange and correlation cannot be separated.

    Reference: Phys. Rev. Lett. 112, 076403.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        T: Temperature in Hartree.
        **kwargs: Throwaway arguments.

    Returns:
        KSDT exchange-correlation energy density and potential.
    """
    return lda_xc_gdsmfb_spin(
        n,
        zeta,
        T=T,
        zeta0_coeffs=Zeta0Coeffs,
        zeta1_coeffs=Zeta1Coeffs,
        phi_params=PhiParams,
        **kwargs,
    )
