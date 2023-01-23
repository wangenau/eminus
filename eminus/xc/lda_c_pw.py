#!/usr/bin/env python3
'''Perdew-Wang LDA correlation.

Reference: Phys. Rev. B 45, 13244.
'''
import numpy as np


def lda_c_pw(n, a=0.031091, **kwargs):
    '''Perdew-Wang parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_PW and ID 12 in LibXC.
    Reference: Phys. Rev. B 45, 13244.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        a (float): Functional parameter.

    Returns:
        tuple[ndarray, ndarray]: PW correlation energy density and potential.
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**third
    rs = pi34 / n**third

    a1 = 0.2137
    b1 = 7.5957
    b2 = 3.5876
    b3 = 1.6382
    b4 = 0.49294

    rs12 = np.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    om = 2 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    olog = np.log(1 + 1 / om)
    ec = -2 * a * (1 + a1 * rs) * olog

    dom = 2 * a * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    vc = -2 * a * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * a * (1 + a1 * rs) * dom / (om * (om + 1))
    return ec, np.array([vc])


def lda_c_pw_spin(n, zeta, a=(0.031091, 0.015545, 0.016887), fz0=1.709921, **kwargs):
    '''Perdew-Wang parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW and ID 12 in LibXC.
    Reference: Phys. Rev. B 45, 13244.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        a (tuple): Functional parameters.
        fz0 (float): Functional parameter.

    Returns:
        tuple[ndarray, ndarray]: PW correlation energy density and potential.
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**third
    rs = pi34 / n**third

    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta
    zeta4 = zeta3 * zeta
    rs12 = np.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    def pw_fit(i):
        '''Calculate correlation energies by Perdew-Wang approximation interpolation.

        Args:
            i (int): Index to choose unpolarized (0), polarized (1), or antiferromagnetic (2) fit.

        Returns:
            tuple[ndarray, ndarray]: PW fit and the derivative.
        '''
        a1 = (0.2137, 0.20548, 0.11125)
        b1 = (7.5957, 14.1189, 10.357)
        b2 = (3.5876, 6.1977, 3.6231)
        b3 = (1.6382, 3.3662, 0.88026)
        b4 = (0.49294, 0.62517, 0.49671)

        om = 2 * a[i] * (b1[i] * rs12 + b2[i] * rs + b3[i] * rs32 + b4[i] * rs2)
        dom = 2 * a[i] * (0.5 * b1[i] * rs12 + b2[i] * rs + 1.5 * b3[i] * rs32 + 2 * b4[i] * rs2)
        olog = np.log(1 + 1 / om)

        fit = -2 * a[i] * (1 + a1[i] * rs) * olog
        dfit = -2 * a[i] * (1 + 2 / 3 * a1[i] * rs) * olog - \
            2 / 3 * a[i] * (1 + a1[i] * rs) * dom / (om * (om + 1))
        return fit, dfit

    ecU, vcU = pw_fit(0)  # Unpolarized
    ecP, vcP = pw_fit(1)  # Polarized
    ac, dac = pw_fit(2)   # Spin stiffness
    ac, dac = -ac, -dac   # The PW spin interpolation parametrizes -ac instead of ac

    fz = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) / (2**(4 / 3) - 2)
    ec = ecU + ac * fz * (1 - zeta4) / fz0 + (ecP - ecU) * fz * zeta4

    dfz = ((1 + zeta)**third - (1 - zeta)**third) * 4 / (3 * (2**(4 / 3) - 2))
    factor1 = vcU + dac * fz * (1 - zeta4) / fz0 + (vcP - vcU) * fz * zeta4
    factor2 = (ac / fz0 * (dfz * (1 - zeta4) - 4 * fz * zeta3) +
               (ecP - ecU) * (dfz * zeta4 + 4 * fz * zeta3))

    vcup = factor1 + factor2 * (1 - zeta)
    vcdw = factor1 - factor2 * (1 + zeta)
    return ec, np.array([vcup, vcdw])
