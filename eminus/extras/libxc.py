# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Interface to use Libxc functionals.

For a list of available functionals, see: https://libxc.gitlab.io/functionals

One can install pylibxc by compiling Libxc according to
https://libxc.gitlab.io/installation/#python-library

Alternatively, one can use the PySCF Libxc interface with::

    pip install eminus[libxc]
"""

import numpy as np
from scipy.linalg import norm

from .. import config
from ..logger import log


def libxc_functional(xc, n_spin, Nspin, dn_spin=None, tau=None, xc_params=None):
    """Handle Libxc exchange-correlation functionals using pylibxc.

    Reference: SoftwareX 7, 1.

    Args:
        xc: Exchange or correlation identifier.
        n_spin: Real-space electronic densities per spin channel.
        Nspin: Number of spin states.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        tau: Real-space kinetic energy densities per spin channel.
        xc_params: Exchange-correlation functional parameters.

    Returns:
        Exchange-correlationenergy density and potentials.
    """
    try:
        if not config.use_pylibxc:
            raise AssertionError  # noqa: TRY301
        from pylibxc import LibXCFunctional
    except (ImportError, AssertionError):
        return pyscf_functional(xc, n_spin, Nspin, dn_spin, tau, xc_params)

    # Libxc functionals have one integer and one string identifier
    try:
        func = LibXCFunctional(int(xc), Nspin)
    except ValueError:
        func = LibXCFunctional(xc, Nspin)

    # Set the external Libxc parameters
    if xc_params is not None and xc_params != {}:
        param_names = func.get_ext_param_names()
        param_defaults = func.get_ext_param_default_values()
        params = []
        for i, name in enumerate(param_names):
            if name in xc_params:
                # pylibxc uses lists as parameters, unravel the parameters from our dictionary
                params.append(xc_params[name])
            else:
                # If parameters are not given use the defaults since selections are not allowed
                params.append(param_defaults[i])
        func.set_ext_params(params)

    if dn_spin is None:
        # Libxc expects a 1d array, so reshape n_spin (same as n_spin.ravel(order="F"))
        out = func.compute({"rho": n_spin.T.ravel()})
    else:
        # The gradients have to be reshaped as well but also squared
        if Nspin == 1:
            dn2 = norm(dn_spin, axis=2) ** 2
        else:
            # For the spin-polarized case the gradients of spin-up and -down are mixed
            dn2 = np.vstack(
                (
                    norm(dn_spin[0], axis=1) ** 2,
                    np.sum(dn_spin[0] * dn_spin[1], axis=1),
                    norm(dn_spin[1], axis=1) ** 2,
                )
            )
        if tau is None:
            out = func.compute({"rho": n_spin.T.ravel(), "sigma": dn2.T.ravel()})
        else:
            out = func.compute(
                {"rho": n_spin.T.ravel(), "sigma": dn2.T.ravel(), "tau": tau.T.ravel()}
            )
    # zk is a column vector, flatten it to a 1d row vector
    exc = out["zk"].ravel()

    # vrho (and vsigma) is exactly transposed from what we need
    vxc = out["vrho"].T
    if dn_spin is not None:
        vsigma = np.atleast_2d(out["vsigma"].T)
        if tau is not None:
            vtau = out["vtau"].T
            return exc, vxc, vsigma, vtau
        return exc, vxc, vsigma, None
    return exc, vxc, None, None


def pyscf_functional(xc, n_spin, Nspin, dn_spin=None, tau=None, xc_params=None):
    """Handle Libxc exchange-correlation functionals using PySCF.

    Reference: WIREs Comput. Mol. Sci. 8, e1340.

    Args:
        xc: Exchange or correlation identifier.
        n_spin: Real-space electronic densities per spin channel.
        Nspin: Number of spin states.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        tau: Real-space kinetic energy densities per spin channel.
        xc_params: Exchange-correlation functional parameters.

    Returns:
        Exchange-correlation energy density and potentials.
    """
    try:
        from pyscf.dft.libxc import eval_xc
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[libxc]".\n\n'
        )
        raise

    if xc_params is not None:
        log.warning("xc_params are not supported when using PySCF as the Libxc backend.")

    if dn_spin is None:
        # For LDAs we only need the spin densities that are already in the needed shape
        rho = n_spin
    elif tau is None:
        # For GGAs we have to append the density gradients
        # The input "density" rho is sorted as (n,grad_x n,grad_y n,grad_z n)
        if Nspin == 1:
            # For spin-paired systems we have to remove the spin indexing (the outermost shape)
            rho = np.vstack((n_spin[0], dn_spin[0].T))
        else:
            rho = np.array(
                [
                    np.vstack((n_spin[0], dn_spin[0].T)),
                    np.vstack((n_spin[1], dn_spin[1].T)),
                ]
            )
    else:
        # For meta-GGAs we have to append the kinetic energy densities as well
        # The input "density" rho is sorted as (n,grad_x n,grad_y n,grad_z n,lapl,tau)
        # We do not support meta-GGAs with a dependency of the laplacian so set it to zero
        lapl = np.zeros(len(n_spin[0]))
        if Nspin == 1:
            rho = np.vstack((n_spin[0], dn_spin[0].T, lapl, tau[0]))
        else:
            rho = np.array(
                [
                    np.vstack((n_spin[0], dn_spin[0].T, lapl, tau[0])),
                    np.vstack((n_spin[1], dn_spin[1].T, lapl, tau[1])),
                ]
            )

    # Spin in PySCF is the number of unpaired electrons, not the number of spin channels
    exc, vxc, _, _ = eval_xc(xc, rho, spin=Nspin - 1)
    # The first entry of vxc is vrho
    # The second entry of the second entry is vsigma
    # The fourth entry of the second entry is vtau (the third would be vlapl)
    # These arrays are 1d for Nspin=1 and a 2d column array for Nspin=2, reshape them as needed
    if dn_spin is not None:
        if tau is not None:
            return exc, np.atleast_2d(vxc[0].T), np.atleast_2d(vxc[1].T), np.atleast_2d(vxc[3].T)
        return exc, np.atleast_2d(vxc[0].T), np.atleast_2d(vxc[1].T), None
    return exc, np.atleast_2d(vxc[0].T), None, None
