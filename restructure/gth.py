#!/usr/bin/env python3
import numpy as np
from numpy.linalg import det
from scipy.special import erf, sph_harm

GTH_LDA_Abinit = {
    'H':    {'rloc': 0.2000000,
             'Znuc': 1,
             'Zion': 1,
             'C1'  : -4.0663326,
             'C2'  : 0.6778322,
             'C3'  : 0,
             'C4'  : 0,
             'rs'  : 0,
             'h1s' : 0,
             'h2s' : 0,
             'rp'  : 0,
             'h1p' : 0
            },
    'He':   {'rloc' : 0.2000000,
             'Znuc' : 2,
             'Zion' : 2,
             'C1'   : -9.11202340,
             'C2'   : 1.69836797,
             'C3'   : 0,
             'C4'   : 0,
             'rs'   : 0,
             'h1s'  : 0,
             'h2s'  : 0,
             'rp'   : 0,
             'h1p'  : 0
            },
    'Li':   {'rloc' : 0.7875530500,  #Li2: replace me!
             'Znuc' : 1,
             'Zion' : 1,
             'C1'   : -1.8926124700,
             'C2'   :  0.2860596800,
             'C3'   : 0,
             'C4'   : 0,
             'rs'   : 0.666375,
             'h1s'  : 1.8588111100,
             'h2s'  : 0,
             'rp'   : 1.079306,
             'h1p'  : -0.0058950400
            }
}


def Vloc(a):
    '''Local contribution for the GTH pseudopotential.'''
    rloc = GTH_LDA_Abinit[a.atom]['rloc']
    Zion = GTH_LDA_Abinit[a.atom]['Zion']
    C1 = GTH_LDA_Abinit[a.atom]['C1']
    C2 = GTH_LDA_Abinit[a.atom]['C2']
    C3 = GTH_LDA_Abinit[a.atom]['C3']
    C4 = GTH_LDA_Abinit[a.atom]['C4']

    omega = 1  # det(a.R)
    rlocG2 = a.G2[1:] * rloc**2

    Vps = -4 * np.pi * Zion / omega * np.exp(-0.5 * rlocG2) / a.G2[1:] \
        + np.sqrt((2 * np.pi)**3) * rloc**3 / omega * np.exp(-0.5 * rlocG2) * \
        (C1 + C2 * (3 - rlocG2) + C3 * (15 - 10 * rlocG2 + rlocG2**2) + C4 * (105 - 105 * rlocG2 + 21 * rlocG2**2 - rlocG2**3))
    return np.concatenate(([0], Vps))  # Why zero?


def Ylm(l, m, r):
    '''Spherical harmonics for cartesian coordinates.'''
    theta = np.arctan(np.sqrt(r[:,0]**2 + r[:,1]**2) / r[:,2])
    phi = np.arctan(r[:,1] / r[:,0])
    return sph_harm(m, l, theta, phi)


def p1s(a):
    '''Needed projector for the nonlocal potential.'''
    rs = GTH_LDA_Abinit[a.atom]['rs']
    omega = 1  # det(a.R)
    return 4 / np.sqrt(omega) * rs * np.sqrt(2 * rs) * np.pi**(5 / 4) * np.exp(-0.5 * a.G2 * rs**2)


def p2s(a):
    '''Needed projector for the nonlocal potential.'''
    rs = GTH_LDA_Abinit[a.atom]['rs']
    omega = 1  # det(a.R)
    rlocG2 = a.G2 * rs**2
    return 8 / np.sqrt(omega) * rs * np.sqrt(2 * rs / 15) * np.pi**(5 / 4) * np.exp(-0.5 * rlocG2) * (3 - rlocG2)


def p1p(a):
    '''Needed projector for the nonlocal potential.'''
    rp = GTH_LDA_Abinit[a.atom]['rp']
    omega = 1  # det(a.R)
    rlocG2 = a.G2 * rp**2
    return 8 / np.sqrt(omega) * rp**2 * np.sqrt(rp / 3) * np.pi**(5 / 4) * np.exp(-0.5 * rlocG2) * np.sqrt(a.G2)


def Vnonloc(a):
    '''Nonlocal potential for the GTH pseudopotentials.'''
    h1s = GTH_LDA_Abinit[a.atom]['h1s']
    h2s = GTH_LDA_Abinit[a.atom]['h2s']
    h1p = GTH_LDA_Abinit[a.atom]['h1p']

    s = 0
    m = 0
    H = Ylm(s, m, a.G) * p1s(a) * h1s * p1s(a) * Ylm(s, m, a.G).conj() \
      + Ylm(s, m, a.G) * p2s(a) * h2s * p2s(a) * Ylm(s, m, a.G).conj()

    p = 1
    for m in (-1, 0, 1):
        H -= Ylm(p, m, a.G) * p1p(a) * h1p * p1p(a) * Ylm(p, m, a.G).conj()
    #H[0] = 0
    return H


def GTH_pot(a):
    '''Goedecker, Teter, Hutter pseudopotentials.'''
    if a.verbose >= 4:
        print(f'Potential used: GTH for {a.atom}')
    Vps = Vloc(a)# + Vnonloc(a)
    Vdual = a.J(Vps * a.Sf)
    return Vps, Vdual
