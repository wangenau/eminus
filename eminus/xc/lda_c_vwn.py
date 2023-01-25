#!/usr/bin/env python3
'''Vosko-Wilk-Nusair LDA correlation.

Reference: Phys. Rev. B 22, 3812.
'''
import numpy as np


def lda_c_vwn(n, exc_only=False, **kwargs):
    '''Vosko-Wilk-Nusair parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_VWN and ID 7 in LibXC.
    Reference: Phys. Rev. B 22, 3812.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: VWN correlation energy density and potential.
    '''
    a = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498

    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**(1 / 3)

    q = np.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = np.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = np.arctan(q / (2 * rs12 + b))

    ec = a * (np.log(rs / fx) + f1 * qx - f2 * (np.log((rs12 - x0)**2 / fx) + f3 * qx))
    if exc_only:
        return ec, None

    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * a / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
                              f2 * (2 / (rs12 - x0) - tx / fx - 4 * (2 * x0 + b) / tt))
    return ec, np.array([vc])


def lda_c_vwn_spin(n, zeta, exc_only=False, **kwargs):
    '''Vosko-Wilk-Nusair parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_VWN and ID 7 in LibXC.
    Reference: Phys. Rev. B 22, 3812.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: VWN correlation energy density and potential.
    '''
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**(1 / 3)
    zeta4 = zeta**4

    cfz = 2**(4 / 3) - 2
    iddfz0 = 9 / 8 * cfz
    trup = 1 + zeta
    trdw = 1 - zeta
    trup13 = trup**(1 / 3)
    trdw13 = trdw**(1 / 3)

    def pade_fit(i):
        '''Calculate correlation energies by Pade approximation interpolation.

        Args:
            i (int): Index to choose paramagnetic (0), ferromagnetic (1), or spin stiffness (2) fit.

        Returns:
            tuple[ndarray, ndarray]: Pade fit and the derivative.
        '''
        a = (0.0310907, 0.01554535, -0.01688686394039)
        b = (3.72744, 7.06042, 1.13107)
        c = (12.9352, 18.0578, 13.0045)
        x0 = (-0.10498, -0.325, -0.0047584)
        Q = (6.15199081975908, 4.73092690956011, 7.12310891781812)
        tbQ = (1.21178334272806, 2.98479352354082, 0.31757762321188)
        bx0fx0 = (-0.03116760867894, -0.14460061018521, -0.00041403379428)

        sqrtrs = np.sqrt(rs)

        xx0 = sqrtrs - x0[i]
        Qtxb = Q[i] / (2 * sqrtrs + b[i])
        atg = np.arctan(Qtxb)
        fx = rs + b[i] * sqrtrs + c[i]
        fit = a[i] * (np.log(rs / fx) + tbQ[i] * atg -
                      bx0fx0[i] * (np.log(xx0 * xx0 / fx) + (tbQ[i] + 4 * x0[i] / Q[i]) * atg))
        if exc_only:
            return fit, None

        txb = 2 * sqrtrs + b[i]
        txbfx = txb / fx
        itxbQ = 1 / (txb * txb + Q[i] * Q[i])
        dfit = fit - a[i] / 3 + \
            a[i] * sqrtrs / 6 * (txbfx + 4 * b[i] * itxbQ + bx0fx0[i] *
                                 (2 / xx0 - txbfx - 4 * (b[i] + 2 * x0[i]) * itxbQ))
        return fit, dfit

    ecP, vcP = pade_fit(0)  # Paramagnetic fit
    ecF, vcF = pade_fit(1)  # Ferromagnetic fit
    ac, dac = pade_fit(2)   # Spin stiffness

    fz = (trup13 * trup + trdw13 * trdw - 2) / cfz  # f(zeta)
    ac *= iddfz0
    De = ecF - ecP - ac  # e_c[F] - e_c[P] - alpha_c/(ddf/ddz(z=0))
    fzz4 = fz * zeta4

    ec = ecP + ac * fz + De * fzz4
    if exc_only:
        return ec, None

    dfz = 4 / 3 * (trup13 - trdw13) / cfz  # df/dzeta
    dac *= iddfz0
    dec1 = vcP + dac * fz + (vcF - vcP - dac) * fzz4  # e_c-(r_s/3)*(de_c/dr_s)
    dec2 = ac * dfz + De * (4 * zeta**3 * fz + zeta4 * dfz)  # de_c/dzeta

    vcup = dec1 + (1 - zeta) * dec2
    vcdw = dec1 - (1 + zeta) * dec2
    return ec, np.array([vcup, vcdw])
