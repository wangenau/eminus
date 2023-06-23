#!/usr/bin/env python3
'''Slater LDA exchange.

Reference: Phys. Rev. 81, 385.
'''
import numpy as np


def lda_x(n, **kwargs):
    '''Slater exchange functional (spin-paired).

    Corresponds to the functional with the label LDA_X and ID 1 in Libxc.

    Reference: Phys. Rev. 81, 385.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        tuple[ndarray, ndarray]: Exchange energy density and potential.
    '''
    f = -3 / 4 * (3 / (2 * np.pi))**(2 / 3)
    rs = (3 / (4 * np.pi * n))**(1 / 3)

    ex = f / rs

    vx = 4 / 3 * ex
    return ex, np.array([vx]), None


def lda_x_spin(n, zeta, **kwargs):
    '''Slater exchange functional (spin-polarized).

    Corresponds to the functional with the label LDA_X and ID 1 in Libxc.

    Reference: Phys. Rev. 81, 385.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        tuple[ndarray, ndarray]: Exchange energy density and potential.
    '''
    f = -3 / 4 * (3 / np.pi)**(1 / 3)

    rho13p = ((1 + zeta) * n)**(1 / 3)
    rho13m = ((1 - zeta) * n)**(1 / 3)

    exup = f * rho13p
    exdw = f * rho13m
    ex = 0.5 * ((1 + zeta) * exup + (1 - zeta) * exdw)

    vxup = 4 / 3 * exup
    vxdw = 4 / 3 * exdw
    return ex, np.array([vxup, vxdw]), None
