#!/usr/bin/env python3
'''Utility functions for exchange-correlation functionals.'''
import numpy as np

from .lda_c_chachiyo import lda_c_chachiyo, lda_c_chachiyo_spin
from .lda_c_pw import lda_c_pw, lda_c_pw_spin
from .lda_c_vwn import lda_c_vwn, lda_c_vwn_spin
from .lda_x import lda_x, lda_x_spin


def get_xc(xc, n_spin, Nspin, dens_threshold=0):
    '''Handle and get exchange-correlation functionals.

    Args:
        xc (str): Exchange and correlation identifier, separated by a comma.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Keyword Args:
        dens_threshold (float): Do not treat densities smaller than the threshold.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    exch, corr = xc.split(',')

    # Only use non-zero values of the density
    n = np.sum(n_spin, axis=0)
    nz_mask = np.where(n > dens_threshold)
    n_nz = n[nz_mask]

    # Zeta is only needed for non-zero values of the density
    zeta_nz = get_zeta(n_spin[:, nz_mask])

    # Only import libxc interface if necessary
    if 'libxc' in xc:
        from .extras.libxc import libxc_functional

    # Handle exchange part
    if 'libxc' in exch:
        exch = exch.split(':')[1]
        ex, vx = libxc_functional(exch, n_spin, Nspin)
    else:
        # If the desired functional is not implemented use a mock functional
        try:
            f_exch = XC_MAP[exch]
            if Nspin == 2:
                f_exch += '_spin'
        except KeyError:
            f_exch = 'mock_xc'
        ex_nz, vx_nz = IMPLEMENTED[f_exch](n_nz, zeta=zeta_nz, Nspin=Nspin)

        # Map the non-zero values back to the right dimension
        ex = np.zeros_like(n)
        ex[nz_mask] = ex_nz
        vx = np.zeros_like(n_spin)
        for spin in range(Nspin):
            vx[spin, nz_mask] = vx_nz[spin]

    # Handle correlation part
    if 'libxc' in corr:
        corr = corr.split(':')[1]
        ec, vc = libxc_functional(corr, n_spin, Nspin)
    else:
        # If the desired functional is not implemented use a mock functional
        try:
            f_corr = XC_MAP[corr]
            if Nspin == 2:
                f_corr += '_spin'
        except KeyError:
            f_corr = 'mock_xc'
        ec_nz, vc_nz = IMPLEMENTED[f_corr](n_nz, zeta=zeta_nz, Nspin=Nspin)

        # Map the non-zero values back to the right dimension
        ec = np.zeros_like(n)
        ec[nz_mask] = ec_nz
        vc = np.zeros_like(n_spin)
        for spin in range(Nspin):
            vc[spin, nz_mask] = vc_nz[spin]

    return ex + ec, vx + vc


def get_zeta(n_spin):
    '''Calculate the relative spin polarization.

    Args:
        n_spin (ndarray): Real-space electronic densities per spin channel.

    Returns:
        ndarray: Relative spin polarization.
    '''
    # If only one spin is given return an array of ones as if the density only is in one channel
    if len(n_spin) == 1:
        return np.ones_like(n_spin[0])
    return (n_spin[0] - n_spin[1]) / (n_spin[0] + n_spin[1])


def mock_xc(n, Nspin=1, **kwargs):
    '''Mock exchange-correlation functional with no effect (spin-paired).

    Args:
        n (ndarray): Real-space electronic density.

    Returns:
        tuple[ndarray, ndarray]: Mock exchange-correlation energy density and potential.
    '''
    zeros = np.zeros_like(n)
    return zeros, np.array([zeros] * Nspin)


IMPLEMENTED = {
    'mock_xc': mock_xc,
    'lda_x': lda_x,
    'lda_x_spin': lda_x_spin,
    'lda_c_chachiyo': lda_c_chachiyo,
    'lda_c_chachiyo_spin': lda_c_chachiyo_spin,
    'lda_c_pw': lda_c_pw,
    'lda_c_pw_spin': lda_c_pw_spin,
    'lda_c_vwn': lda_c_vwn,
    'lda_c_vwn_spin': lda_c_vwn_spin
}


XC_MAP = {
    # lda_x
    '1': 'lda_x',
    'lda': 'lda_x',
    's': 'lda_x',
    'slater': 'lda_x',
    # lda_c_chachiyo
    '287': 'lda_c_chachiyo',
    'chachiyo': 'lda_c_chachiyo',
    # lda_c_pw
    '12': 'lda_c_pw',
    'pw': 'lda_c_pw',
    'pw92': 'lda_c_pw',
    # lda_c_vwn
    '7': 'lda_c_vwn',
    'vwn': 'lda_c_vwn',
    'vwn5': 'lda_c_vwn'
}
