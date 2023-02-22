#!/usr/bin/env python3
'''Slater LDA exchange.

Reference: Phys. Rev. 81, 385.
'''
import numpy as np


def lda_x(n, exc_only=False, **kwargs):
    '''Slater exchange functional (spin-paired).

    Corresponds to the functional with the label LDA_X and ID 1 in Libxc.
    Reference: Phys. Rev. 81, 385.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: Exchange energy density and potential.
    '''
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    f = -3 / 4 * (3 / (2 * np.pi))**(2 / 3)
    rs = pi34 / n**(1 / 3)

    ex = f / rs
    if exc_only:
        return ex, None

    vx = 4 / 3 * ex
    return ex, np.array([vx])


def lda_x_spin(n, zeta, exc_only=False, **kwargs):
    '''Slater exchange functional (spin-polarized).

    Corresponds to the functional with the label LDA_X and ID 1 in Libxc.
    Reference: Phys. Rev. 81, 385.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: Exchange energy density and potential.
    '''
    f = -3 / 4 * (3 / np.pi)**(1 / 3)

    rho13p = ((1 + zeta) * n)**(1 / 3)
    rho13m = ((1 - zeta) * n)**(1 / 3)

    exup = f * rho13p
    exdw = f * rho13m
    ex = 0.5 * ((1 + zeta) * exup + (1 - zeta) * exdw)
    if exc_only:
        return ex, None

    vxup = 4 / 3 * f * rho13p
    vxdw = 4 / 3 * f * rho13m
    return ex, np.array([vxup, vxdw])