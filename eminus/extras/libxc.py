#!/usr/bin/env python3
'''Interface to use LibXC functionals.

For a list of available functionals, see: https://www.tddft.org/programs/libxc/functionals

One can install pylibxc by compiling LibXC according to:
https://tddft.org/programs/libxc/installation/#python-library

Alternatively, one can use the PySCF LibXC interface with::

    pip install eminus[libxc]
'''
import numpy as np

from ..logger import log


def libxc_functional(xc, n_spin, Nspin):
    '''Handle LibXC exchange-correlation functionals via pylibxc.

    Only LDA functionals should be used.
    Reference: SoftwareX 7, 1.

    Args:
        xc (str | int): Exchange or correlation identifier.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    try:
        from pylibxc import LibXCFunctional
    except ImportError:
        return pyscf_functional(xc, n_spin, Nspin)

    # LibXC functionals have one integer and one string identifier
    try:
        func = LibXCFunctional(int(xc), Nspin)
    except ValueError:
        func = LibXCFunctional(xc, Nspin)

    # LibXC expects a 1d array, so reshape n_spin (same as n_spin.ravel(order='F'))
    out = func.compute({'rho': n_spin.T.ravel()})
    # zk is a column vector, flatten it to a 1d row vector
    exc = out['zk'].ravel()
    # vrho is exactly transposed from what we need
    vxc = out['vrho'].T
    return exc, vxc


def pyscf_functional(xc, n_spin, Nspin):
    '''Handle LibXC exchange-correlation functionals via PySCF.

    Only LDA functionals should be used.
    Reference: WIREs Comput. Mol. Sci. 8, e1340.

    Args:
        xc (str | int): Exchange or correlation identifier.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    try:
        from pyscf.dft.libxc import eval_xc
    except ImportError:
        log.exception('Necessary dependencies not found. To use this module, '
                      'install them with "pip install eminus[libxc]".\n\n')
        raise
    # The function also returns fxc and kxc (higher derivatives) that are not needed at this point
    # Spin in PySCF is the number of unpaired electrons, not the number of spin channels
    exc, vxc, _, _ = eval_xc(xc, n_spin, spin=Nspin - 1)
    # The first entry of vxc is vrho as a column array
    return exc, np.atleast_2d(vxc[0].T)
