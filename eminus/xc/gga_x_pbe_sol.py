#!/usr/bin/env python3
"""Perdew-Burke-Ernzerhof GGA exchange for solids and surfaces.

Reference: Phys. Rev. Lett. 100, 136406.
"""

from .gga_x_pbe import gga_x_pbe, gga_x_pbe_spin


def gga_x_pbe_sol(n, **kwargs):
    """Perdew-Burke-Ernzerhof solid parametrization of the exchange functional (spin-paired).

    Corresponds to the functional with the label GGA_X_PBE_SOL and ID 116 in Libxc.

    Reference: Phys. Rev. Lett. 100, 136406.

    Args:
        n (ndarray): Real-space electronic density.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        tuple[ndarray, ndarray, ndarray]: PBE exchange energy density, potential, and vsigma.
    """
    return gga_x_pbe(n, mu=10 / 81, **kwargs)


def gga_x_pbe_sol_spin(n, zeta, **kwargs):
    """Perdew-Burke-Ernzerhof solid parametrization of the exchange functional (spin-polarized).

    Corresponds to the functional with the label GGA_X_PBE_SOL and ID 116 in Libxc.

    Reference: Phys. Rev. Lett. 100, 136406.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        tuple[ndarray, ndarray, ndarray]: PBE exchange energy density, potential, and vsigma.
    """
    return gga_x_pbe_spin(n, zeta, mu=10 / 81, **kwargs)
