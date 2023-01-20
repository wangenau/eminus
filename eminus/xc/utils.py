#!/usr/bin/env python3
'''Utility functions for exchange-correlation functionals.'''
import numpy as np

from .lda_c_chachiyo import lda_c_chachiyo, lda_c_chachiyo_spin
from .lda_c_pw import lda_c_pw, lda_c_pw_spin
from .lda_c_pw_mod import lda_c_pw_mod, lda_c_pw_mod_spin
from .lda_c_vwn import lda_c_vwn, lda_c_vwn_spin
from .lda_x import lda_x, lda_x_spin
from ..logger import log


def get_xc(xc, n_spin, Nspin, dens_threshold=0):
    '''Handle and get exchange-correlation functionals.

    Args:
        xc (list): Exchange and correlation identifier.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Keyword Args:
        dens_threshold (float): Do not treat densities smaller than the threshold.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    f_exch, f_corr = xc

    # Only use non-zero values of the density
    n = np.sum(n_spin, axis=0)
    nz_mask = np.where(n > dens_threshold)
    n_nz = n[nz_mask]
    # Zeta is only needed for non-zero values of the density
    zeta_nz = get_zeta(n_spin[:, nz_mask])

    # Only import libxc interface if necessary
    if 'libxc' in str(xc):
        from ..extras.libxc import libxc_functional

    # Handle exchange part
    if 'libxc' in f_exch:
        f_exch = f_exch.split(':')[1]
        ex, vx = libxc_functional(f_exch, n_spin, Nspin)
    else:
        if Nspin == 2 and f_exch != 'mock_xc':
            f_exch += '_spin'
        ex_nz, vx_nz = IMPLEMENTED[f_exch](n_nz, zeta=zeta_nz, Nspin=Nspin)
        # Map the non-zero values back to the right dimension
        ex = np.zeros_like(n)
        ex[nz_mask] = ex_nz
        vx = np.zeros_like(n_spin)
        for spin in range(Nspin):
            vx[spin, nz_mask] = vx_nz[spin]

    # Handle correlation part
    if 'libxc' in f_corr:
        f_corr = f_corr.split(':')[1]
        ec, vc = libxc_functional(f_corr, n_spin, Nspin)
    else:
        if Nspin == 2 and f_corr != 'mock_xc':
            f_corr += '_spin'
        ec_nz, vc_nz = IMPLEMENTED[f_corr](n_nz, zeta=zeta_nz, Nspin=Nspin)
        # Map the non-zero values back to the right dimension
        ec = np.zeros_like(n)
        ec[nz_mask] = ec_nz
        vc = np.zeros_like(n_spin)
        for spin in range(Nspin):
            vc[spin, nz_mask] = vc_nz[spin]

    return ex + ec, vx + vc


def parse_functionals(xc):
    '''Parse exchange-correlation functional strings to the internal format.

    Args:
        xc (str): Exchange and correlation identifier, separated by a comma.

    Returns:
        list: Exchange and correlation string.
    '''
    # Check for combined aliases
    try:
        xc = ALIAS[xc]
    except KeyError:
        pass

    # Parse functionals
    functionals = []
    for f in xc.split(','):
        if 'libxc' in f or f in IMPLEMENTED.keys():
            f_xc = f
        elif f == '':
            f_xc = 'mock_xc'
        else:
            try:
                f_xc = XC_MAP[f]
            except KeyError:
                log.exception(f'No functional found for "{f}"')
                raise
        functionals.append(f_xc)

    # If only one or no functional has been parsed append with mock functionals
    for i in range(2 - len(functionals)):
        functionals.append('mock_xc')
    return functionals


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
        Nspin (int): Number of spin states.

    Returns:
        tuple[ndarray, ndarray]: Mock exchange-correlation energy density and potential.
    '''
    zeros = np.zeros_like(n)
    return zeros, np.array([zeros] * Nspin)


IMPLEMENTED = {
    'mock_xc': mock_xc,
    'lda_x': lda_x,
    'lda_x_spin': lda_x_spin,
    'lda_c_pw': lda_c_pw,
    'lda_c_pw_spin': lda_c_pw_spin,
    'lda_c_pw_mod': lda_c_pw_mod,
    'lda_c_pw_mod_spin': lda_c_pw_mod_spin,
    'lda_c_vwn': lda_c_vwn,
    'lda_c_vwn_spin': lda_c_vwn_spin,
    'lda_c_chachiyo': lda_c_chachiyo,
    'lda_c_chachiyo_spin': lda_c_chachiyo_spin
}


XC_MAP = {
    # lda_x
    '1': 'lda_x',
    's': 'lda_x',
    'lda': 'lda_x',
    'slater': 'lda_x',
    # lda_c_pw
    '12': 'lda_c_pw',
    'pw': 'lda_c_pw',
    'pw92': 'lda_c_pw',
    # lda_c_pw_mod
    '13': 'lda_c_pw_mod',
    'pw_mod': 'lda_c_pw_mod',
    'pw92_mod': 'lda_c_pw_mod',
    # lda_c_vwn
    '7': 'lda_c_vwn',
    'vwn': 'lda_c_vwn',
    'vwn5': 'lda_c_vwn',
    # lda_c_chachiyo
    '287': 'lda_c_chachiyo',
    'chachiyo': 'lda_c_chachiyo'
}

ALIAS = {
    'spw92': 'slater,pw_mod',
    'svwn': 'slater,vwn'
}
