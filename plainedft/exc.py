#!/usr/bin/env python3
'''
Parametizations of density functionals.
'''
import numpy as np


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_x_slater.jl
def lda_slater_x(n, alpha=2 / 3):
    '''Slater exchange functional (spin paired).'''
    third = 1 / 3
    p43 = 4 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    f = -9 / 8 * (3 / (2 * np.pi))**(2 / 3)
    rs = pi34 / n**third

    ex = f * alpha / rs
    vx = p43 * f * alpha / rs
    # In PWDFT they return Vx that is ex+n*dex/dn, but we need dex/dn, so return (vx-ex)/n instead
    return ex, (vx - ex) / n


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn.jl
def lda_vwn_c(n):
    '''VWN parameterization of the correlation functional (spin paired).'''
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
    # In PWDFT they return Vc that is ec+n*dec/dn, but we need dec/dn, so return (vc-ec)/n instead
    return ec, (vc - ec) / n


# Adapted from
# https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_x_slater_spin.jl
def lda_slater_x_spin(n, zeta, alpha=2 / 3):
    '''Slater exchange functional (spin polarized).'''
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
    # FIXME: exp is probably wrong
    return ex, [(vxup - exup) / n, (vxdw - exdw) / n]


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn_spin.jl
def lda_vwn_c_spin(n, zeta):
    '''VWN parameterization of the correlation functional (spin polarized).'''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    A = (0.0310907, 0.01554535, -0.01688686394039)
    b = (3.72744, 7.06042, 1.13107)
    c = (12.9352, 18.0578, 13.0045)
    x0 = (-0.10498, -0.32500, -0.0047584)
    Q = (6.15199081975908, 4.73092690956011, 7.12310891781812)
    tbQ = (1.21178334272806, 2.98479352354082, 0.31757762321188)
    bx0fx0 = (-0.03116760867894, -0.14460061018521, -0.00041403379428)

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

    ecP, vcP = padefit(sqrtrs, 0, x0, Q, b, c, A, tbQ, bx0fx0)  # ecF = e_c Paramagnetic
    ecF, vcF = padefit(sqrtrs, 1, x0, Q, b, c, A, tbQ, bx0fx0)  # ecP = e_c Ferromagnetic
    ac, dac = padefit(sqrtrs, 2, x0, Q, b, c, A, tbQ, bx0fx0)  # ac = "spin stiffness"

    ac = ac * iddfz0
    dac = dac * iddfz0
    De = ecF - ecP - ac  # e_c[F] - e_c[P] - alpha_c/(ddf/ddz(z=0))
    fzz4 = fz * zeta4
    ec = ecP + ac * fz + De * fzz4

    dec1 = vcP + dac * fz + (vcF - vcP - dac) * fzz4  # e_c-(r_s/3)*(de_c/dr_s)
    dec2 = ac * dfz + De * (4 * zeta3 * fz + zeta4 * dfz)  # de_c/dzeta

    vcup = dec1 + (1 - zeta) * dec2
    vcdw = dec1 - (1 + zeta) * dec2
    # FIXME: ecp is probably wrong
    return ec, [(vcup - ec) / n, (vcdw - ec) / n]


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn_spin.jl
def padefit(x, i, x0, Q, b, c, A, tbQ, bx0fx0):
    '''Implement eq. 4.4 in S.H. Vosko, L. Wilk, and M. Nusair, Can. J. Phys. 58, 1200 (1980).'''
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
