#!/usr/bin/env python3
'''
Parametization of the VWN local density approximation functional.
'''
import numpy as np


def excVWN(n):
    '''VWN parameterization of the exchange correlation energy functional.'''
    X1 = 0.75 * (3 / (2 * np.pi))**(2 / 3)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c
    rs = (4 * np.pi / 3 * n)**(-1 / 3)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    out = -X1 / rs + A * (np.log(x * x / X) + 2 * b / Q * np.arctan(Q / (2 * x + b)) -
          (b * x0) / X0 * (np.log((x - x0) * (x - x0) / X) + 2 * (2 * x0 + b) / Q *
          np.arctan(Q / (2 * x + b))))
    return out


def excpVWN(n):
    '''Derivation with respect to n of the VWN exchange correlation energy functional.'''
    X1 = 0.75 * (3 / (2 * np.pi))**(2 / 3)
    A = 0.0310907
    x0 = -0.10498
    b = 3.72744
    c = 12.9352
    Q = np.sqrt(4 * c - b * b)
    X0 = x0 * x0 + b * x0 + c
    rs = (4 * np.pi / 3 * n)**(-1 / 3)
    x = np.sqrt(rs)
    X = x * x + b * x + c
    dx = 0.5 / x
    out = dx * (2 * X1 / (rs * x) + A * (2 / x - (2 * x + b) / X - 4 * b / (Q * Q + (2 * x + b) *
          (2 * x + b)) - (b * x0) / X0 * (2 / (x - x0) - (2 * x + b) / X - 4 * (2 * x0 + b) /
          (Q * Q + (2 * x + b) * (2 * x + b)))))
    return (-rs / (3 * n)) * out


# For debugging, taken from PWDFT
def xc_vwn(n):
    third = 1 / 3
    pi34 = 0.6203504908994  # pi34=(3/4pi)^(1/3)
    rs = pi34 / n**third

    a = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498

    q = np.sqrt(4.0 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = np.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = np.arctan(q / (2 * rs12 + b))
    ec = a * (np.log(rs / fx) + f1 * qx - f2 * (np.log((rs12 - x0)**2 / fx) + f3 * qx))

    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * a / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
         f2 * (2 / (rs12 - x0) - tx / fx - 4 * (2 * x0 + b) / tt))

    f = -0.687247939924714
    alpha = 2 / 3

    ex = f * alpha / rs
    vx = 4 / 3 * f * alpha / rs

    return ex + ec, vx + vc
