#!/usr/bin/env python3
'''Interface to use LibXC functionals.

For a list of available functionals, see: https://www.tddft.org/programs/libxc/functionals/
'''
try:
    from pylibxc import LibXCFunctional
except ImportError:
    print('ERROR: Necessary addon dependencies not found. To use this module,\n'
          '       install the package with addons, e.g., with "pip install eminus[addons]"')

from ..logger import log


def libxc_functional(xc, n, spinpol):
    '''Handle LibXC exchange-correlation functionals.

    Only LDA functionals should be used.

    Reference: SoftwareX 7, 1.

    Args:
        xc (str | int): Exchange or correlation identifier.
        n (ndarray): Real-space electronic density.
        spinpol (bool): Choose if a spin-polarized exchange-correlation functional will be used.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    if spinpol:
        log.warning('The LibXC routine will still use an unpolarized functional.')
    # FIXME: Compare LibXC and internal polarized functionals
    spin = 'unpolarized'

    inp = {'rho': n}
    # LibXC functionals have one integer and one string identifier
    try:
        func = LibXCFunctional(int(xc), spin)
    except ValueError:
        func = LibXCFunctional(xc, spin)
    out = func.compute(inp)

    exc = out['zk'].ravel()
    vxc = out['vrho'].ravel()
    return exc, vxc
