#!/usr/bin/env python3
'''Slater LDA exchange.

Reference: Phys. Rev. 81, 385.
'''
import numpy as np


def lda_x(n, alpha=2 / 3, **kwargs):
    '''Slater exchange functional (spin-paired).

    Corresponds to the functional with the label LDA_C_PW and ID 1 in LibXC.
    Reference: Phys. Rev. 81, 385.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        alpha (float): Scaling factor.

    Returns:
        tuple[ndarray, ndarray]: Exchange energy density and potential.
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**third
    f = -9 / 8 * (3 / (2 * np.pi))**(2 / 3)
    rs = pi34 / n**third

    ex = f * alpha / rs
    vx = 4 / 3 * ex
    return ex, np.array([vx])


def lda_x_spin(n, zeta, alpha=2 / 3, **kwargs):
    '''Slater exchange functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW and ID 1 in LibXC.
    Reference: Phys. Rev. 81, 385.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        alpha (float): Scaling factor.

    Returns:
        tuple[ndarray, ndarray]: Exchange energy density and potential.
    '''
    third = 1 / 3
    p43 = 4 / 3
    f = -9 / 8 * (3 / np.pi)**third

    rho13p = ((1 + zeta) * n)**third
    rho13m = ((1 - zeta) * n)**third

    exup = f * alpha * rho13p
    exdw = f * alpha * rho13m
    ex = 0.5 * ((1 + zeta) * exup + (1 - zeta) * exdw)

    vxup = p43 * f * alpha * rho13p
    vxdw = p43 * f * alpha * rho13m
    return ex, np.array([vxup, vxdw])
