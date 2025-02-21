# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Perdew-Burke-Ernzerhof GGA exchange.

Reference: Phys. Rev. Lett. 77, 3865.
"""

import numpy as np
from scipy.linalg import norm

from .lda_x import lda_x, lda_x_spin


def gga_x_pbe(n, mu=0.2195149727645171, dn_spin=None, **kwargs):
    """Perdew-Burke-Ernzerhof parametrization of the exchange functional (spin-paired).

    Corresponds to the functional with the label GGA_X_PBE and ID 101 in Libxc.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        mu: Functional parameter.
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE exchange energy density, potential, and vsigma.
    """
    ex, vx, _ = lda_x(n, **kwargs)
    gex, gvx, vsigmax = pbe_x_base(n, mu, dn_spin[0], **kwargs)
    vx, gvx = vx[0], gvx[0]  # Remove spin dimension for the correct shape
    return ex + gex / n, np.array([vx + gvx]), np.array([0.5 * vsigmax])


def gga_x_pbe_spin(n, zeta, mu=0.2195149727645171, dn_spin=None, **kwargs):
    """Perdew-Burke-Ernzerhof parametrization of the exchange functional (spin-polarized).

    Corresponds to the functional with the label GGA_X_PBE and ID 101 in Libxc.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        mu: Functional parameter.
        dn_spin: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE exchange energy density, potential, and vsigma.
    """
    # Use the spin-scaling relationship Exc(n_up, n_down)=(Exc(2 n_up)+Exc(2 n_down))/2
    zeta = zeta[0]  # Getting the non-zero values from zeta adds an extra dimension, remove it here
    n_up = zeta * n + n  # 2 * n_up
    n_dw = -zeta * n + n  # 2 * n_down
    ex_up, vx_up, vsigma_up = pbe_x_base(n_up, mu, 2 * dn_spin[0], **kwargs)
    ex_dw, vx_dw, vsigma_dw = pbe_x_base(n_dw, mu, 2 * dn_spin[1], **kwargs)
    vx_up, vx_dw = vx_up[0], vx_dw[0]  # Remove spin dimension for the correct shape

    ex, vx, _ = lda_x_spin(n, zeta, **kwargs)

    vsigmax = np.array([vsigma_up, np.zeros_like(ex), vsigma_dw])
    return ex + 0.5 * (ex_up + ex_dw) / n, np.array([vx[0] + vx_up, vx[1] + vx_dw]), vsigmax


def pbe_x_base(n, mu=0.2195149727645171, dn=None, **kwargs):
    """Base PBE exchange functional to be used in the spin-paired and -polarized case.

    Reference: Phys. Rev. Lett. 77, 3865.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        mu: Functional parameter.
        dn: Real-space gradient of densities per spin channel.
        **kwargs: Throwaway arguments.

    Returns:
        PBE exchange energy density, potential, and vsigma.
    """
    kappa = 0.804

    norm_dn = norm(dn, axis=1)
    kf = (3 * np.pi**2 * n) ** (1 / 3)
    # Handle divisions by zero
    # divkf = 1 / kf
    divkf = np.divide(1, kf, out=np.zeros_like(kf), where=kf > 0)
    # Handle divisions by zero
    # s = norm_dn * divkf / (2 * n)
    s = np.divide(norm_dn * divkf, 2 * n, out=np.zeros_like(n), where=n > 0)
    f1 = 1 + mu * s**2 / kappa
    Fx = kappa - kappa / f1
    exunif = -3 * kf / (4 * np.pi)
    # In Fx a "1 + " is missing, since n * exunif is the Slater exchange that is added later
    sx = exunif * Fx

    dsdn = -4 / 3 * s
    dFxds = 2 * mu * s / f1**2
    dexunif = exunif / 3
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn  # dFx/dn = dFx/ds * ds/dn

    # Handle divisions by zero
    # vsigmax = exunifdFx * divkf / (2 * norm_dn)
    vsigmax = np.divide(
        exunifdFx * divkf, 2 * norm_dn, out=np.zeros_like(norm_dn), where=norm_dn > 0
    )
    return sx * n, np.array([vx]), vsigmax
