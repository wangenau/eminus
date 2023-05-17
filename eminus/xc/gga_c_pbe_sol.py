#!/usr/bin/env python3
'''Perdew-Burke-Ernzerhof GGA correlation for solids and surfaces.

Reference: Phys. Rev. Lett. 102, 039902.
'''
from .gga_c_pbe import gga_c_pbe, gga_c_pbe_spin


def gga_c_pbe_sol(n, **kwargs):
    '''Perdew-Burke-Ernzerhof solid parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label GGA_C_PBE_SOL and ID 133 in Libxc.
    Reference: Phys. Rev. Lett. 102, 039902.

    Args:
        n (ndarray): Real-space electronic density.

    Returns:
        tuple[ndarray, ndarray, ndarray]: PBE correlation energy density, potential, and vsigma.
    '''
    return gga_c_pbe(n, beta=0.046, **kwargs)


def gga_c_pbe_sol_spin(n, zeta, **kwargs):
    '''Perdew-Burke-Ernzerhof solid parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label GGA_C_PBE_SOL and ID 133 in Libxc.
    Reference: Phys. Rev. Lett. 102, 039902.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Returns:
        tuple[ndarray, ndarray, ndarray]: PBE correlation energy density, potential, and vsigma.
    '''
    return gga_c_pbe_spin(n, zeta, beta=0.046, **kwargs)