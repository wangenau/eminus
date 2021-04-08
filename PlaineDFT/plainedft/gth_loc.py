#!/usr/bin/env python3
'''
Calculate the local potential with GTH pseudopotentials. Phys. Rev. B 54, 1703
'''
import numpy as np
# from numpy.linalg import det


def Vloc(a):
    '''Local contribution for the GTH pseudopotential.'''
    Natoms = len(a.X)

    Vps = np.zeros(len(a.G2) - 1)

    for ia in range(Natoms):
        psp = a.GTH[a.atom[ia]]

        rloc = psp['rlocal']
        Zion = psp['Zval']
        C1 = psp['C'][0]
        C2 = psp['C'][1]
        C3 = psp['C'][2]
        C4 = psp['C'][3]

        omega = 1  # det(a.R)  # FIXME: is this correct?
        rlocG2 = a.G2[1:] * rloc**2

        Vps += -4 * np.pi * Zion / omega * np.exp(-0.5 * rlocG2) / a.G2[1:] \
            + np.sqrt((2 * np.pi)**3) * rloc**3 / omega * np.exp(-0.5 * rlocG2) * \
            (C1 + C2 * (3 - rlocG2) + C3 * (15 - 10 * rlocG2 + rlocG2**2) + C4 * (105 - 105 * rlocG2 + 21 * rlocG2**2 - rlocG2**3))
    return np.concatenate(([0], Vps))


def init_gth_loc(a):
    '''Goedecker, Teter, Hutter pseudopotentials.'''
    return a.J(Vloc(a) * a.Sf)
