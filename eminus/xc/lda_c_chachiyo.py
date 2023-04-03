#!/usr/bin/env python3
'''Chachiyo LDA correlation.

Reference: J. Chem. Phys. 145, 021101.
'''
import numpy as np


def lda_c_chachiyo(n, exc_only=False, **kwargs):
    '''Chachiyo parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_CHACHIYO and ID 287 in Libxc.
    Reference: J. Chem. Phys. 145, 021101.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: Chachiyo correlation energy density and potential.
    '''
    a = -0.01554535  # (np.log(2) - 1) / (2 * np.pi**2)
    b = 20.4562557

    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 * n**(-1 / 3)
    rs2 = rs**2

    ec = a * np.log(1 + b / rs + b / rs2)
    if exc_only:
        return ec, None

    vc = ec + a * b * (2 + rs) / (3 * (b + b * rs + rs2))
    return ec, np.array([vc])


def chachiyo_scaling(zeta, exc_only=False):
    '''Weighting factor between the paramagnetic and the ferromagnetic case.

    Reference: J. Chem. Phys. 145, 021101.

    Args:
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: Weighting factor and its derivative.
    '''
    fzeta = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) / (2 * (2**(1 / 3) - 1))
    if exc_only:
        return fzeta, None

    dfdzeta = (2 * (1 - zeta)**(1 / 3) - 2 * (1 + zeta)**(1 / 3)) / (3 - 3 * 2**(1 / 3))
    return fzeta, dfdzeta


def lda_c_chachiyo_spin(n, zeta, weight_function=chachiyo_scaling, exc_only=False, **kwargs):
    '''Chachiyo parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_CHACHIYO and ID 287 in Libxc.
    Reference: J. Chem. Phys. 145, 021101.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        weight_function (Callable): Functional function.
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: Chachiyo correlation energy density and potential.
    '''
    a0 = -0.01554535   # (np.log(2) - 1) / (2 * np.pi**2)
    a1 = -0.007772675  # (np.log(2) - 1) / (4 * np.pi**2)
    b0 = 20.4562557
    b1 = 27.4203609

    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 * n**(-1 / 3)
    rs2 = rs**2

    fzeta, dfdzeta = weight_function(zeta, exc_only=exc_only)

    ec0 = a0 * np.log(1 + b0 / rs + b0 / rs2)
    ec1 = a1 * np.log(1 + b1 / rs + b1 / rs2)
    ec = ec0 + (ec1 - ec0) * fzeta
    if exc_only:
        return ec, None

    factor = -1 / rs2 - 2 / rs**3
    dec0drs = a0 / (1 + b0 / rs + b0 / rs2) * b0 * factor
    dec1drs = a1 / (1 + b1 / rs + b1 / rs2) * b1 * factor
    prefactor = ec - rs / 3 * (dec0drs + (dec1drs - dec0drs) * fzeta)

    vcup = prefactor + (ec1 - ec0) * dfdzeta * (1 - zeta)
    vcdw = prefactor - (ec1 - ec0) * dfdzeta * (1 + zeta)
    return ec, np.array([vcup, vcdw])
