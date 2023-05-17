#!/usr/bin/env python3
'''Vosko-Wilk-Nusair LDA correlation.

Reference: Phys. Rev. B 22, 3812.
'''
import numpy as np


def lda_c_vwn(n, A=0.0310907, b=3.72744, c=12.9352, x0=-0.10498, exc_only=False, **kwargs):
    '''Vosko-Wilk-Nusair parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_VWN and ID 7 in Libxc.
    Reference: Phys. Rev. B 22, 3812.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.
        A (float): Functional parameter.
        b (float): Functional parameter.
        c (float): Functional parameter.
        x0 (float): Functional parameter.

    Returns:
        tuple[ndarray, ndarray]: VWN correlation energy density and potential.
    '''
    rs = (3 / (4 * np.pi * n))**(1 / 3)

    x = np.sqrt(rs)
    X = rs + b * x + c
    Q = np.sqrt(4 * c - b**2)
    fx0 = b * x0 / (x0**2 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / Q
    tx = 2 * x + b
    tanx = np.arctan(Q / tx)

    ec = A * (np.log(rs / X) + 2 * b / Q * tanx - fx0 * (np.log((x - x0)**2 / X) + f3 * tanx))
    if exc_only:
        return ec, None, None

    tt = tx**2 + Q**2
    vc = ec - x * A / 6 * (2 / x - tx / X - 4 * b / tt -
                           fx0 * (2 / (x - x0) - tx / X - 4 * (2 * x0 + b) / tt))
    return ec, np.array([vc]), None


def lda_c_vwn_spin(n, zeta, exc_only=False, **kwargs):
    '''Vosko-Wilk-Nusair parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_VWN and ID 7 in Libxc.
    Reference: Phys. Rev. B 22, 3812.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: VWN correlation energy density and potential.
    '''
    A = (0.0310907, 0.01554535, -1 / (6 * np.pi**2))
    b = (3.72744, 7.06042, 1.13107)
    c = (12.9352, 18.0578, 13.0045)
    x0 = (-0.10498, -0.325, -0.0047584)

    ec0, vc0, _ = lda_c_vwn(n, A[0], b[0], c[0], x0[0], exc_only)  # Paramagnetic
    ec1, vc1, _ = lda_c_vwn(n, A[1], b[1], c[1], x0[1], exc_only)  # Ferromagnetic
    ac, dac, _ = lda_c_vwn(n, A[2], b[2], c[2], x0[2], exc_only)   # Spin stiffness

    d2fzeta0 = 4 / 9 / (2**(1 / 3) - 1)
    fzeta = 0.5 * ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) / (2**(1 / 3) - 1)
    zeta4 = zeta**4
    fzetaz4 = fzeta * zeta4
    De = ec1 - ec0 - ac / d2fzeta0

    ec = ec0 + ac / d2fzeta0 * fzeta + De * fzetaz4
    if exc_only:
        return ec, None, None

    dfzeta = 4 / 6 * ((1 + zeta)**(1 / 3) - (1 - zeta)**(1 / 3)) / (2**(1 / 3) - 1)
    dec1 = vc0 + dac / d2fzeta0 * fzeta + (vc1 - vc0 - dac / d2fzeta0) * fzetaz4
    dec2 = ac / d2fzeta0 * dfzeta + De * (4 * zeta**3 * fzeta + zeta4 * dfzeta)

    vcup = dec1 + (1 - zeta) * dec2
    vcdw = dec1 - (1 + zeta) * dec2
    return ec, np.array([vcup, vcdw]), None
