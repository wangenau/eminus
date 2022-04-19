#!/usr/bin/env python3
'''Interface to use LibXC functionals.

For a list of available functionals, see: https://www.tddft.org/programs/libxc/functionals/
'''
try:
    from pylibxc import LibXCFunctional
except ImportError:
    print('ERROR: Necessary addon dependencies not found. '
          'To use this module, install the package with addons, e.g., with '
          '"pip install eminus[addons]"')


def libxc_functional(exc, n, ret, spinpol):
    '''Handle LibXC exchange-correlation functionals.

    Only LDA functionals should be used.

    Args:
        exc (str or int): Exchange or correlation identifier.
        n (array): Real-space electronic density.
        ret (str): Choose whether to return the energy density or the potential.
        spinpol (bool): Choose if a spin-polarized exchange-correlation functional will be used.

    Returns:
        array: Exchange or correlation energy density or potential.
    '''
    if spinpol:
        spin = 'polarized'
    else:
        spin = 'unpolarized'

    inp = {}
    inp['rho'] = n
    try:
        func = LibXCFunctional(int(exc), spin)
    except KeyError:
        func = LibXCFunctional(exc, spin)
    out = func.compute(inp)
    if ret == 'density':
        return out['zk'].flatten()
    else:
        return out['vrho'].flatten()
