#!/usr/bin/env python3
'''Chachiyo GGA exchange.

Reference: Molecules 25, 3485.
'''
import numpy as np
from numpy.linalg import norm

from .lda_x import lda_x


def gga_x_chachiyo(n, exc_only=False, dn_spin=None, **kwargs):
    '''Chachiyo parametrization of the exchange functional (spin-paired).

    Corresponds to the functional with the label GGA_X_CHACHIYO and ID 298 in Libxc.
    Reference: Molecules 25, 3485.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.
        dn_spin (ndarray): Real-space gradient of densities per spin channel.

    Returns:
        tuple[ndarray, ndarray, ndarray]: Chachiyo exchange energy density, potential, and vsigma.
    '''
    norm_dn = norm(dn_spin[0], axis=1)
    ex, vx, _ = lda_x(n, **kwargs)

    x = norm_dn / n**(4 / 3) * 2 / 9 * (np.pi / 3)**(1 / 3)
    x1 = x + 1
    logx1 = np.log(x1)
    div = 3 * x + np.pi**2
    tmpgex = (3 * x**2 + np.pi**2 * logx1)
    gex = tmpgex / (div * logx1)
    if exc_only:
        return ex * gex, None, None

    term1 = 8 * ex / tmpgex * (x**2 + x * np.pi**2 / (6 * x1)) + \
        2 / 3 * norm_dn / n * (1 / div + 1 / (3 * logx1 * x1))
    gvx = (1 + 1 / 3) * ex - term1

    vsigmax = n * 3 * term1 / (8 * norm_dn**2)
    return ex * gex, np.array([gvx]) * gex, np.array([vsigmax]) * gex


def gga_x_chachiyo_spin(n, zeta, exc_only=False, dn_spin=None, **kwargs):
    '''Chachiyo parametrization of the exchange functional (spin-polarized).

    Corresponds to the functional with the label GGA_X_CHACHIYO and ID 298 in Libxc.
    Reference: Molecules 25, 3485.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.
        dn_spin (ndarray): Real-space gradient of densities per spin channel.

    Returns:
        tuple[ndarray, ndarray, ndarray]: Chachiyo exchange energy density, potential, and vsigma.
    '''
    n_up = zeta * n + n   # 2 * n_up
    n_dw = -zeta * n + n  # 2 * n_down
    ex_up, vx_up, vsigma_up = gga_x_chachiyo(n_up, exc_only, np.array([2 * dn_spin[0]]), **kwargs)
    ex_dw, vx_dw, vsigma_dw = gga_x_chachiyo(n_dw, exc_only, np.array([2 * dn_spin[1]]), **kwargs)
    if exc_only:
        return 0.5 * (ex_up * n_up + ex_dw * n_dw) / n, None, None

    vsigmax = np.array([2 * vsigma_up, np.zeros_like(vsigma_up), 2 * vsigma_dw])
    return 0.5 * (ex_up * n_up + ex_dw * n_dw) / n, np.array([vx_up, vx_dw]), vsigmax
