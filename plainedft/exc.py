#!/usr/bin/env python3
'''
Parametization of the VWN local density approximation functional.
'''
import numpy as np


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_x_slater.jl
def lda_vwn_x(n):
    '''VWN parameterization of the exchange energy functional (spin paired).'''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    f = -0.687247939924714
    alpha = 2 / 3

    ex = f * alpha / rs
    vx = 4 / 3 * f * alpha / rs
    # In PWDFT they return Vx that is ex+n*dex/dn, but we need dex/dn, so return (vx-ex)/n instead
    return ex, (vx - ex) / n


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn.jl
def lda_vwn_c(n):
    '''VWN parameterization of the correlation energy functional (spin paired).'''
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


# Adapted from https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_c_vwn_spin.jl
# and https://github.com/f-fathurrahman/PWDFT.jl/blob/master/src/XC_funcs/XC_x_slater_spin.jl
def lda_vwn_spin(n, zeta):
    '''VWN parameterization of the exchange-correlation energy functional (spin polarized).'''
    # Exchange contribution
    f = -9 / 8 * (3 / np.pi)**(1 / 3)
    alpha = 2 / 3
    third = 1 / 3

    rho13 = ((1 + zeta) * n)**third
    exup = f * alpha * rho13

    rho13 = ((1 - zeta) * n)**third
    exdw = f * alpha * rho13
    ex = 0.5 * ((1 + zeta) * exup + (1 - zeta) * exdw)

    # Correlation contribution
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 / n**third

    A = (0.0310907, 0.01554535, -0.01688686394039)
    x0 = (-0.10498, -0.32500, -0.0047584)
    b = (3.72744, 7.06042, 1.13107)
    c = (12.9352, 18.0578, 13.0045)
    Q = (6.15199081975908, 4.73092690956011, 7.12310891781812)
    tbQ = (1.21178334272806, 2.98479352354082, 0.31757762321188)
    bx0fx0 = (-0.03116760867894, -0.14460061018521, -0.00041403379428)

    cfz = 2**(4 / 3) - 2
    cfz1 = 1 / cfz
    iddfz0 = 9 / 8 * cfz
    sqrtrs = np.sqrt(rs)
    zeta3 = zeta**3
    zeta4 = zeta3 * zeta
    trup = 1 + zeta
    trdw = 1 - zeta
    trup13 = trup**(1 / 3)
    trdw13 = trdw**(1 / 3)
    fz = cfz1 * (trup13 * trup + trdw13 * trdw - 2)  # f(zeta)

    ecP = padefit(sqrtrs, 0, x0, Q, b, c, A, tbQ, bx0fx0)  # ecF = e_c Paramagnetic
    ecF = padefit(sqrtrs, 1, x0, Q, b, c, A, tbQ, bx0fx0)  # ecP = e_c Ferromagnetic
    ac = padefit(sqrtrs, 2, x0, Q, b, c, A, tbQ, bx0fx0)   # ac = "spin stiffness"

    ac = ac * iddfz0
    De = ecF - ecP - ac  # e_c[F] - e_c[P] - alpha_c/(ddf/ddz(z=0))
    fzz4 = fz * zeta4
    ec = ecP + ac * fz + De * fzz4

    return ex + ec


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
    return fit
