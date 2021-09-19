#!/usr/bin/env python3
'''
Parametizations of density functionals.
'''
import numpy as np


def get_exc(exc, n, spinpol):
    '''Handle and get exchange-correlation functionals.

    Args:
        exc : str
            Exchange and correlation identifier, separated by a comma.

        n : array
            Real-space electronic density.

        spinpol : bool
            Choose if a spin-polarized exchange-correlation functional will be used.

    Returns:
        Exchange-correlation energy density and potential as a tuple(array, array).
    '''
    exch, corr = exc.split(',')
    if spinpol:
        exc_map = {
            'lda': lda_slater_x_spin,
            'pw': lda_pw_c_spin,
            'vwn': lda_vwn_c_spin
        }
        # FIXME: WARNING: For unpolarized calculations we insert ones as zeta, fix this for later
        ex, vx = exc_map.get(exch, dummy_exc)(n, np.ones((n.shape)))
        ec, vc = exc_map.get(corr, dummy_exc)(n, np.ones((n.shape)))
    else:
        exc_map = {
            'lda': lda_slater_x,
            'pw': lda_pw_c,
            'vwn': lda_vwn_c
        }
        ex, vx = exc_map.get(exch, dummy_exc)(n)
        ec, vc = exc_map.get(corr, dummy_exc)(n)
    return ex + ec, vx + vc


def dummy_exc(n, **kwargs):
    '''Dummy exchange-correlation functional with no effect.

    Args:
        n : array
            Real-space electronic density.

    Returns:
        Dummy exchange-correlation energy density and potential as a tuple(array, array).
    '''
    zero = np.zeros((n.shape))
    return zero, zero


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_x_slater.jl
def lda_slater_x(n, alpha=2 / 3):
    '''Slater exchange functional (spin paired).

    Args:
        n : array
            Real-space electronic density.

    Kwargs:
        alpha : float
            Scaling factor.

    Returns:
        Exchange energy density and potential as a tuple(array, array).
    '''
    third = 1 / 3
    p43 = 4 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    f = -9 / 8 * (3 / (2 * np.pi))**(2 / 3)
    rs = pi34 / n**third

    ex = f * alpha / rs
    vx = p43 * f * alpha / rs
    return ex, vx


# Adapted from
# https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_x_slater_spin.jl
def lda_slater_x_spin(n, zeta, alpha=2 / 3):
    '''Slater exchange functional (spin polarized).

    Args:
        n : array
            Real-space electronic density.

        zeta : array
            Relative spin polarization.

    Kwargs:
        alpha : float
            Scaling factor.

    Returns:
        Exchange energy density and potential as a tuple(array, array).
    '''
    third = 1 / 3
    p43 = 4 / 3
    f = -9 / 8 * (3 / np.pi)**(1 / 3)

    rho13 = ((1 + zeta) * n)**third
    exup = f * alpha * rho13
    vxup = p43 * f * alpha * rho13

    rho13 = ((1 - zeta) * n)**third
    exdw = f * alpha * rho13
    vxdw = p43 * f * alpha * rho13
    ex = 0.5 * ((1 + zeta) * exup + (1 - zeta) * exdw)
    return ex, [vxup, vxdw]


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_pw.jl
def lda_pw_c(n):
    '''PW parameterization of the correlation functional (spin paired).

    Args:
        n : array
            Real-space electronic density.

    Returns:
        Correlation energy density and potential as a tuple(array, array).
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    a = 0.031091
    a1 = 0.21370
    b1 = 7.5957
    b2 = 3.5876
    b3 = 1.6382
    b4 = 0.49294

    rs12 = np.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    om = 2 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    dom = 2 * a * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    olog = np.log(1 + 1 / om)
    ec = -2 * a * (1 + a1 * rs) * olog
    vc = -2 * a * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * a * (1 + a1 * rs) * dom / (om * (om + 1))
    return ec, vc


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_pw_spin.jl
def lda_pw_c_spin(n, zeta):
    '''PW parameterization of the correlation functional (spin polarized).

    Args:
        n : array
            Real-space electronic density.

        zeta : array
            Relative spin polarization.

    Returns:
        Correlation energy density and potential as a tuple(array, array).
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    # Unpolarised parameters
    a = 0.031091
    a1 = 0.21370
    b1 = 7.5957
    b2 = 3.5876
    b3 = 1.6382
    b4 = 0.49294

    # Polarised parameters
    ap = 0.015545
    a1p = 0.20548
    b1p = 14.1189
    b2p = 6.1977
    b3p = 3.3662
    b4p = 0.62517

    # Antiferromagnetic parameters
    aa = 0.016887
    a1a = 0.11125
    b1a = 10.357
    b2a = 3.6231
    b3a = 0.88026
    b4a = 0.49671

    fz0 = 1.709921

    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta
    zeta4 = zeta3 * zeta
    rs12 = np.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    # Unpolarised
    om = 2 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    dom = 2 * a * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    olog = np.log(1 + 1 / om)
    epwc = -2 * a * (1 + a1 * rs) * olog
    vpwc = -2 * a * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * a * (1 + a1 * rs) * dom / (om * (om + 1))

    # Polarized
    omp = 2 * ap * (b1p * rs12 + b2p * rs + b3p * rs32 + b4p * rs2)
    domp = 2 * ap * (0.5 * b1p * rs12 + b2p * rs + 1.5 * b3p * rs32 + 2 * b4p * rs2)
    ologp = np.log(1 + 1 / omp)
    epwcp = -2 * ap * (1 + a1p * rs) * ologp
    vpwcp = -2 * ap * (1 + 2 / 3 * a1p * rs) * ologp - \
             2 / 3 * ap * (1 + a1p * rs) * domp / (omp * (omp + 1))

    # Antiferromagnetic
    oma = 2 * aa * (b1a * rs12 + b2a * rs + b3a * rs32 + b4a * rs2)
    doma = 2 * aa * (0.5 * b1a * rs12 + b2a * rs + 1.5 * b3a * rs32 + 2 * b4a * rs2)
    ologa = np.log(1 + 1 / oma)
    alpha = 2 * aa * (1 + a1a * rs) * ologa
    vpwca = 2 * aa * (1 + 2 / 3 * a1a * rs) * ologa + \
            2 / 3 * aa * (1 + a1a * rs) * doma / (oma * (oma + 1))

    fz = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) / (2**(4 / 3) - 2)
    dfz = ((1 + zeta)**(1 / 3) - (1 - zeta)**(1 / 3)) * 4 / (3 * (2**(4 / 3) - 2))

    ec = epwc + alpha * fz * (1 - zeta4) / fz0 + (epwcp - epwc) * fz * zeta4

    vcup = vpwc + vpwca * fz * (1 - zeta4) / fz0 + (vpwcp - vpwc) * fz * zeta4 + \
           (alpha / fz0 * (dfz * (1 - zeta4) - 4 * fz * zeta3) +
           (epwcp - epwc) * (dfz * zeta4 + 4 * fz * zeta3)) * (1 - zeta)

    vcdw = vpwc + vpwca * fz * (1 - zeta4) / fz0 + (vpwcp - vpwc) * fz * zeta4 - \
           (alpha / fz0 * (dfz * (1 - zeta4) - 4 * fz * zeta3) +
           (epwcp - epwc) * (dfz * zeta4 + 4 * fz * zeta3)) * (1 + zeta)
    return ec, [vcup, vcdw]


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn.jl
def lda_vwn_c(n):
    '''VWN parameterization of the correlation functional (spin paired).

    Args:
        n : array
            Real-space electronic density.

    Returns:
        Correlation energy density and potential as a tuple(array, array).
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    A = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498

    q = np.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = np.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = np.arctan(q / (2 * rs12 + b))
    ec = A * (np.log(rs / fx) + f1 * qx - f2 * (np.log((rs12 - x0)**2 / fx) + f3 * qx))

    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * A / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
         f2 * (2 / (rs12 - x0) - tx / fx - 4.0 * (2 * x0 + b) / tt))
    return ec, vc


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn_spin.jl
def lda_vwn_c_spin(n, zeta):
    '''VWN parameterization of the correlation functional (spin polarized).

    Args:
        n : array
            Real-space electronic density.

        zeta : array
            Relative spin polarization.

    Returns:
        Correlation energy density and potential as a tuple(array, array).
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    cfz = 2**(4 / 3) - 2
    cfz1 = 1 / cfz
    cfz2 = 4 / 3 * cfz1
    iddfz0 = 9 / 8 * cfz
    sqrtrs = np.sqrt(rs)
    zeta3 = zeta**3
    zeta4 = zeta3 * zeta
    trup = 1 + zeta
    trdw = 1 - zeta
    trup13 = trup**(1 / 3)
    trdw13 = trdw**(1 / 3)
    fz = cfz1 * (trup13 * trup + trdw13 * trdw - 2)  # f(zeta)
    dfz = cfz2 * (trup13 - trdw13)  # df/dzeta

    ecP, vcP = padefit(sqrtrs, 0)  # Paramagnetic fit
    ecF, vcF = padefit(sqrtrs, 1)  # Ferromagnetic fit
    ac, dac = padefit(sqrtrs, 2)  # ac = "spin stiffness"

    ac = ac * iddfz0
    dac = dac * iddfz0
    De = ecF - ecP - ac  # e_c[F] - e_c[P] - alpha_c/(ddf/ddz(z=0))
    fzz4 = fz * zeta4
    ec = ecP + ac * fz + De * fzz4

    dec1 = vcP + dac * fz + (vcF - vcP - dac) * fzz4  # e_c-(r_s/3)*(de_c/dr_s)
    dec2 = ac * dfz + De * (4 * zeta3 * fz + zeta4 * dfz)  # de_c/dzeta

    vcup = dec1 + (1 - zeta) * dec2
    vcdw = dec1 - (1 + zeta) * dec2
    return ec, [vcup, vcdw]


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn_spin.jl
def padefit(x, i):
    '''Calculate correlation energies by Pade approximation interpolation.

    Args:
        x : array
              Square root of the density.

        i : int
            Index weather to use a paramagnetic (0), ferromagnetic (1), or spin stiffness (2) fit.

    Returns:
        Pade fit and its derivative as a tuple(array, array).
    '''
    A = (0.0310907, 0.01554535, -0.01688686394039)
    b = (3.72744, 7.06042, 1.13107)
    c = (12.9352, 18.0578, 13.0045)
    x0 = (-0.10498, -0.32500, -0.0047584)
    Q = (6.15199081975908, 4.73092690956011, 7.12310891781812)
    tbQ = (1.21178334272806, 2.98479352354082, 0.31757762321188)
    bx0fx0 = (-0.03116760867894, -0.14460061018521, -0.00041403379428)

    # Pade fit calculated in x and its derivative with respect to rho
    # rs = inv((rho*)^(1/3)) = x^2
    sqx = x * x                   # x^2 = r_s
    xx0 = x - x0[i]               # x - x_0
    Qtxb = Q[i] / (2 * x + b[i])  # Q / (2x+b)
    atg = np.arctan(Qtxb)         # tan^-1(Q/(2x+b))
    fx = sqx + b[i] * x + c[i]    # X(x) = x^2 + b*x + c

    fit = A[i] * (np.log(sqx / fx) + tbQ[i] * atg -
          bx0fx0[i] * (np.log(xx0 * xx0 / fx) + (tbQ[i] + 4 * x0[i] / Q[i]) * atg))

    txb = 2 * x + b[i]
    txbfx = txb / fx
    itxbQ = 1 / (txb * txb + Q[i] * Q[i])

    dfit = fit - A[i] / 3 + A[i] * x / 6 * (txbfx + 4 * b[i] * itxbQ +
           bx0fx0[i] * (2 / xx0 - txbfx - 4 * (b[i] + 2 * x0[i]) * itxbQ))
    return fit, dfit
