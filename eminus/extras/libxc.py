#!/usr/bin/env python3
'''Interface to use Libxc functionals.

For a list of available functionals, see: https://www.tddft.org/programs/libxc/functionals

One can install pylibxc by compiling Libxc according to:
https://tddft.org/programs/libxc/installation/#python-library

Alternatively, one can use the PySCF Libxc interface with::

    pip install eminus[libxc]
'''
import numpy as np
from scipy.linalg import norm

from .. import config
from ..logger import log


def libxc_functional(xc, n_spin, Nspin, dn_spin=None):
    '''Handle Libxc exchange-correlation functionals via pylibxc.

    Only LDA and GGA functionals can be used.
    Reference: SoftwareX 7, 1.

    Args:
        xc (str | int): Exchange or correlation identifier.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Keyword Args:
        dn_spin (ndarray): Real-space gradient of densities per spin channel.

    Returns:
        tuple[ndarray, ndarray]: Exchange-correlation energy density and potential.
    '''
    try:
        assert config.use_pylibxc
        from pylibxc import LibXCFunctional
    except (ImportError, AssertionError):
        return pyscf_functional(xc, n_spin, Nspin, dn_spin)

    # Libxc functionals have one integer and one string identifier
    try:
        func = LibXCFunctional(int(xc), Nspin)
    except ValueError:
        func = LibXCFunctional(xc, Nspin)

    if dn_spin is None:
        # Libxc expects a 1d array, so reshape n_spin (same as n_spin.ravel(order='F'))
        out = func.compute({'rho': n_spin.T.ravel()})
    else:
        # The gradients have to be reshaped as well, but also squared
        if Nspin == 1:
            dn2 = norm(dn_spin, axis=2)**2
        else:
            # For the spin-polairzed case the gradients of spin-up and -down are mixed together
            dn2 = np.vstack((norm(dn_spin[0], axis=1)**2, np.sum(dn_spin[0] * dn_spin[1], axis=1),
                            norm(dn_spin[1], axis=1)**2))
        out = func.compute({'rho': n_spin.T.ravel(), 'sigma': dn2.T.ravel()})
    # zk is a column vector, flatten it to a 1d row vector
    exc = out['zk'].ravel()
    # vrho (and vsigma) is exactly transposed from what we need
    vxc = out['vrho'].T
    if dn_spin is not None:
        vsigma = out['vsigma'].T
        return exc, vxc, np.atleast_2d(vsigma)
    return exc, vxc, None


def pyscf_functional(xc, n_spin, Nspin, dn_spin=None):
    '''Handle Libxc exchange-correlation functionals via PySCF.

    Only LDA and GGA functionals can be used.
    Reference: WIREs Comput. Mol. Sci. 8, e1340.

    Args:
        xc (str | int): Exchange or correlation identifier.
        n_spin (ndarray): Real-space electronic densities per spin channel.
        Nspin (int): Number of spin states.

    Keyword Args:
        dn_spin (ndarray): Real-space gradient of densities per spin channel.

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
    if dn_spin is None:
        # For LDAs we only need the spin densities that are already in the needed shape
        exc, vxc, _, _ = eval_xc(xc, n_spin, spin=Nspin - 1)
        # The first entry of vxc is vrho as a 1d array for Nspin=1 and as a column array for Nspin=2
        return exc, np.atleast_2d(vxc[0].T), None
    else:
        # For GGAs we have to append the density gradients
        # The input "density" rho is sorted as (n,grad_x n,grad_y n,grad_z n)
        if Nspin == 1:
            # For spin-paired systems we have to remove the spin indexing, i.e., the outermost shape
            rho = np.vstack((n_spin[0], dn_spin[0].T))
        else:
            rho = np.array([np.vstack((n_spin[0], dn_spin[0].T)),
                            np.vstack((n_spin[1], dn_spin[1].T))])
        exc, vxc, _, _ = eval_xc(xc, rho, spin=Nspin - 1)
        # The second entry of the second entry is vsigma
        # vsigma can be a 1d or 2d array depending on the spin, reshape it as needed
        return exc, np.atleast_2d(vxc[0].T), np.atleast_2d(vxc[1].T)
