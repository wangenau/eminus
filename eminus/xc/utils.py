#!/usr/bin/env python3
'''Utility functions for exchange-correlation functionals.'''
import numpy as np

from .. import config
from ..logger import log
from ..utils import add_maybe_none
from .gga_c_chachiyo import gga_c_chachiyo, gga_c_chachiyo_spin
from .gga_c_pbe import gga_c_pbe, gga_c_pbe_spin
from .gga_c_pbe_sol import gga_c_pbe_sol, gga_c_pbe_sol_spin
from .gga_x_chachiyo import gga_x_chachiyo, gga_x_chachiyo_spin
from .gga_x_pbe import gga_x_pbe, gga_x_pbe_spin
from .gga_x_pbe_sol import gga_x_pbe_sol, gga_x_pbe_sol_spin
from .lda_c_chachiyo import lda_c_chachiyo, lda_c_chachiyo_spin
from .lda_c_chachiyo_mod import lda_c_chachiyo_mod, lda_c_chachiyo_mod_spin
from .lda_c_pw import lda_c_pw, lda_c_pw_spin
from .lda_c_pw_mod import lda_c_pw_mod, lda_c_pw_mod_spin
from .lda_c_vwn import lda_c_vwn, lda_c_vwn_spin
from .lda_x import lda_x, lda_x_spin


def get_xc(xc, n_spin, Nspin, dn_spin=None, tau=None, dens_threshold=0, exc_only=False):
    '''Handle and get exchange-correlation functionals.

    Args:
        xc (list | str): Exchange and correlation identifier.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Keyword Args:
        dn_spin (ndarray): Real-space gradient of densities per spin channel.
        tau (ndarray): Real-space kinetic energy densities per spin channel.
        dens_threshold (float): Do not treat densities smaller than the threshold.
        exc_only (bool): Only calculate the exchange-correlation energy density.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    if isinstance(xc, str):
        xc = parse_functionals(xc)
    f_exch, f_corr = xc

    # Only use non-zero values of the density
    n = np.sum(n_spin, axis=0)
    nz_mask = np.where(n > dens_threshold)
    n_nz = n[nz_mask]
    # Zeta is only needed for non-zero values of the density
    zeta_nz = get_zeta(n_spin[:, nz_mask])
    # dn_spin is only needed for non-zero values of the density
    if dn_spin is not None:
        dn_spin_nz = dn_spin[:, nz_mask[0], :]
    else:
        dn_spin_nz = None

    # Only import libxc interface if necessary
    if ':' in str(xc):
        from ..extras.libxc import libxc_functional

    # Handle exchange part
    if ':' in f_exch:
        f_exch = f_exch.split(':')[-1]
        ex, vx, vsigmax, vtaux = libxc_functional(f_exch, n_spin, Nspin, dn_spin, tau, exc_only)
    else:
        if Nspin == 2 and f_exch != 'mock_xc':
            f_exch += '_spin'
        ex_nz, vx_nz, vsigmax_nz = IMPLEMENTED[f_exch](n_nz, zeta=zeta_nz, Nspin=Nspin,
                                                       exc_only=exc_only, dn_spin=dn_spin_nz)
        # Map the non-zero values back to the right dimension
        ex = np.zeros_like(n)
        ex[nz_mask] = ex_nz
        # Only map vx and vsigmax if necessary
        if not exc_only:
            vx = np.zeros_like(n_spin)
            for s in range(Nspin):
                vx[s, nz_mask] = vx_nz[s]
            if vsigmax_nz is not None:
                vsigmax = np.zeros((len(vsigmax_nz), len(ex)))
                for i in range(len(vsigmax)):
                    vsigmax[i, nz_mask] = vsigmax_nz[i]
            else:
                vsigmax = None
        # There are no internal meta-GGAs
        vtaux = None

    # Handle correlation part
    if ':' in f_corr:
        f_corr = f_corr.split(':')[-1]
        ec, vc, vsigmac, vtauc = libxc_functional(f_corr, n_spin, Nspin, dn_spin, tau, exc_only)
    else:
        if Nspin == 2 and f_corr != 'mock_xc':
            f_corr += '_spin'
        ec_nz, vc_nz, vsigmac_nz = IMPLEMENTED[f_corr](n_nz, zeta=zeta_nz, Nspin=Nspin,
                                                       exc_only=exc_only, dn_spin=dn_spin_nz)
        # Map the non-zero values back to the right dimension
        ec = np.zeros_like(n)
        ec[nz_mask] = ec_nz
        # Only map vc and vsigmac if necessary
        if not exc_only:
            vc = np.zeros_like(n_spin)
            for s in range(Nspin):
                vc[s, nz_mask] = vc_nz[s]
            if vsigmac_nz is not None:
                vsigmac = np.zeros((len(vsigmac_nz), len(ex)))
                for i in range(len(vsigmac)):
                    vsigmac[i, nz_mask] = vsigmac_nz[i]
            else:
                vsigmac = None
        # There are no internal meta-GGAs
        vtauc = None

    if exc_only:
        return ex + ec, None, None, None
    return ex + ec, vx + vc, add_maybe_none(vsigmax, vsigmac), add_maybe_none(vtaux, vtauc)


def get_exc(*args, **kwargs):
    '''Get the exchange-correlation energy density.

    This is a convenience function to interface :func:`~eminus.xc.utils.get_xc`.
    '''
    exc, _, _, _ = get_xc(*args, **kwargs, exc_only=True)
    return exc


def get_vxc(*args, **kwargs):
    '''Get the exchange-correlation potential.

    This is a convenience function to interface :func:`~eminus.xc.utils.get_xc`.
    '''
    _, vxc, vsigma, vtau = get_xc(*args, **kwargs)
    return vxc, vsigma, vtau


def parse_functionals(xc):
    '''Parse exchange-correlation functional strings to the internal format.

    Args:
        xc (str): Exchange and correlation identifier, separated by a comma.

    Returns:
        list: Exchange and correlation string.
    '''
    # Check for combined aliases
    try:
        # Remove underscores when looking up in the dictionary
        xc_ = xc.replace('_', '')
        xc = ALIAS[xc_]
    except KeyError:
        pass

    # Parse functionals
    functionals = []
    for f in xc.split(','):
        if ':' in f or f in IMPLEMENTED:
            f_xc = f
        elif not f:
            f_xc = 'mock_xc'
        else:
            try:
                # Remove underscores when looking up in the dictionary
                f_ = f.replace('_', '')
                f_xc = XC_MAP[f_]
            except KeyError:
                log.exception(f'No functional found for "{f}".')
                raise
        functionals.append(f_xc)

    # If only one or no functional has been parsed append with mock functionals
    for _ in range(2 - len(functionals)):
        functionals.append('mock_xc')
    return functionals


def parse_xc_type(xc):
    '''Parse functional strings to identify the corresponding functional type.

    Args:
        xc (list): Exchange and correlation identifier, separated by a comma.

    Returns:
        str: Functional type.
    '''
    xc_type = []
    for func in xc:
        if ':' in func:
            xc_id = func.split(':')[-1]
            # Try to parse the functional using pylibxc at first
            try:
                family = parse_xc_libxc(xc_id)
            # Otherwise parse it with PySCF
            except (ImportError, AssertionError):
                family = parse_xc_pyscf(xc_id)

            if family == 1:
                xc_type.append('lda')
            elif family == 2:
                xc_type.append('gga')
            elif family == 4:
                xc_type.append('meta-gga')
            else:
                log.exception('Unsupported functional family.')
                raise
        # Fall back to internal xc functionals
        elif 'gga' in func:
            xc_type.append('gga')
        else:
            xc_type.append('lda')

    # When mixing functional types use the higher level of theory
    if xc_type[0] != xc_type[1]:
        log.warning('Detected mixing of different functional types.')
        if 'meta-gga' in xc_type:
            return 'meta-gga'
        return 'gga'
    return xc_type[0]


def parse_xc_libxc(xc_id):
    '''Parse functional type by its ID using pylibxc.

    Args:
        xc_id (int | string): Functional ID or identifier.

    Returns:
        str: Functional type.
    '''
    assert config.use_pylibxc
    import pylibxc
    if not xc_id.isdigit():
        xc_id = pylibxc.util.xc_functional_get_number(xc_id)

    func = pylibxc.LibXCFunctional(int(xc_id), 1)
    if func._needs_laplacian:
        log.exception('meta-GGAs that need a laplacian are not supported.')
        raise
    return func.get_family()


def parse_xc_pyscf(xc_id):
    '''Parse functional type by its ID using PySCF.

    Args:
        xc_id (int | string): Functional ID or identifier.

    Returns:
        str: Functional type.
    '''
    from pyscf.dft.libxc import is_gga, is_lda, is_meta_gga, needs_laplacian, XC_CODES
    if not xc_id.isdigit():
        xc_id = XC_CODES[xc_id.upper()]

    if needs_laplacian(int(xc_id)):
        log.exception('meta-GGAs that need a laplacian are not supported.')
        raise
    if is_lda(xc_id):
        return 1
    elif is_gga(xc_id):
        return 2
    elif is_meta_gga(xc_id):
        return 4
    return -1


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

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        tuple[ndarray, ndarray]: Mock exchange-correlation energy density and potential.
    '''
    zeros = np.zeros_like(n)
    return zeros, np.array([zeros] * Nspin), None


IMPLEMENTED = {
    'mock_xc': mock_xc,
    'gga_c_chachiyo': gga_c_chachiyo,
    'gga_c_chachiyo_spin': gga_c_chachiyo_spin,
    'gga_c_pbe': gga_c_pbe,
    'gga_c_pbe_spin': gga_c_pbe_spin,
    'gga_c_pbe_sol': gga_c_pbe_sol,
    'gga_c_pbe_sol_spin': gga_c_pbe_sol_spin,
    'gga_x_chachiyo': gga_x_chachiyo,
    'gga_x_chachiyo_spin': gga_x_chachiyo_spin,
    'gga_x_pbe': gga_x_pbe,
    'gga_x_pbe_spin': gga_x_pbe_spin,
    'gga_x_pbe_sol': gga_x_pbe_sol,
    'gga_x_pbe_sol_spin': gga_x_pbe_sol_spin,
    'lda_x': lda_x,
    'lda_x_spin': lda_x_spin,
    'lda_c_pw': lda_c_pw,
    'lda_c_pw_spin': lda_c_pw_spin,
    'lda_c_pw_mod': lda_c_pw_mod,
    'lda_c_pw_mod_spin': lda_c_pw_mod_spin,
    'lda_c_vwn': lda_c_vwn,
    'lda_c_vwn_spin': lda_c_vwn_spin,
    'lda_c_chachiyo': lda_c_chachiyo,
    'lda_c_chachiyo_spin': lda_c_chachiyo_spin,
    'lda_c_chachiyo_mod': lda_c_chachiyo_mod,
    'lda_c_chachiyo_mod_spin': lda_c_chachiyo_mod_spin
}

XC_MAP = {
    # lda_x
    '1': 'lda_x',
    's': 'lda_x',
    'lda': 'lda_x',
    'slater': 'lda_x',
    # lda_c_vwn
    '7': 'lda_c_vwn',
    'vwn': 'lda_c_vwn',
    'vwn5': 'lda_c_vwn',
    # lda_c_pw
    '12': 'lda_c_pw',
    'pw': 'lda_c_pw',
    'pw92': 'lda_c_pw',
    # lda_c_pw_mod
    '13': 'lda_c_pw_mod',
    'pwmod': 'lda_c_pw_mod',
    'pw92mod': 'lda_c_pw_mod',
    # gga_x_pbe
    '101': 'gga_x_pbe',
    'pbex': 'gga_x_pbe',
    # gga_x_pbe_sol
    '116': 'gga_x_pbe_sol',
    'pbesolx': 'gga_x_pbe_sol',
    # gga_c_pbe
    '130': 'gga_c_pbe',
    'pbec': 'gga_c_pbe',
    # gga_c_pbe_sol
    '133': 'gga_c_pbe_sol',
    'pbesolc': 'gga_c_pbe_sol',
    # lda_c_chachiyo
    '287': 'lda_c_chachiyo',
    'chachiyo': 'lda_c_chachiyo',
    # gga_x_chachiyo
    '298': 'gga_x_chachiyo',
    'chachiyox': 'gga_x_chachiyo',
    # lda_c_chachiyo_mod
    '307': 'lda_c_chachiyo_mod',
    'chachiyomod': 'lda_c_chachiyo_mod',
    # gga_c_chachiyo
    '309': 'gga_c_chachiyo',
    'chachiyoc': 'gga_c_chachiyo'
}

ALIAS = {
    'svwn': 'slater,vwn5',
    'svwn5': 'slater,vwn5',
    'spw92': 'slater,pw92mod',
    'pbe': 'pbex,pbec',
    'pbesol': 'pbesolx,pbesolc',
    'chachiyo': 'chachiyox,chachiyoc'
}
