# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for exchange-correlation functionals."""

import inspect
import sys

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
from .lda_xc_gdsmfb import lda_xc_gdsmfb, lda_xc_gdsmfb_spin


def get_xc(xc, n_spin, Nspin, dn_spin=None, tau=None, xc_params=None, dens_threshold=0):
    """Handle and get exchange-correlation functionals.

    Args:
        xc: Exchange and correlation identifier.
        n_spin: Real-space electronic densities per spin channel.
        Nspin: Number of spin states.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        tau: Real-space kinetic energy densities per spin channel.
        xc_params: Exchange-correlation functional parameters.
        dens_threshold: Do not treat densities smaller than the threshold.

    Returns:
        Exchange-correlation energy density and potential.
    """
    if isinstance(xc, str):
        xc = parse_functionals(xc)
    f_exch, f_corr = xc
    if xc_params is None:
        xc_params = {}

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

    def handle_functional(fxc):
        """Calculate a given functional fxc, same for exchange and correlation."""
        # Calculate with the libxc extra...
        if ":" in fxc:
            from ..extras.libxc import libxc_functional

            fxc = fxc.split(":")[-1]
            exc, vxc, vsigma, vtau = libxc_functional(fxc, n_spin, Nspin, dn_spin, tau, xc_params)
        # ...or use an internal functional
        else:
            if Nspin == 2 and fxc != "mock_xc":
                fxc += "_spin"
            exc_nz, vxc_nz, vsigma_nz = IMPLEMENTED[fxc](
                n_nz, zeta=zeta_nz, dn_spin=dn_spin_nz, Nspin=Nspin, **xc_params
            )
            # Map the non-zero values back to the right dimension
            exc = np.zeros_like(n)
            exc[nz_mask] = exc_nz
            vxc = np.zeros_like(n_spin)
            for s in range(Nspin):
                vxc[s, nz_mask] = vxc_nz[s]
            if vsigma_nz is not None:
                vsigma = np.zeros((len(vsigma_nz), len(exc)))
                for i in range(len(vsigma)):
                    vsigma[i, nz_mask] = vsigma_nz[i]
            else:
                vsigma = None
            # There are no internal meta-GGAs
            vtau = None
        return exc, vxc, vsigma, vtau

    ex, vx, vsigmax, vtaux = handle_functional(f_exch)  # Calculate the exchange part
    ec, vc, vsigmac, vtauc = handle_functional(f_corr)  # Calculate the correlation part
    return ex + ec, vx + vc, add_maybe_none(vsigmax, vsigmac), add_maybe_none(vtaux, vtauc)


def get_exc(xc, n_spin, Nspin, dn_spin=None, tau=None, xc_params=None, dens_threshold=0):
    """Get the exchange-correlation energy density.

    This is a convenience function to interface :func:`~eminus.xc.utils.get_xc`.

    Args:
        xc: Exchange and correlation identifier.
        n_spin: Real-space electronic densities per spin channel.
        Nspin: Number of spin states.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        tau: Real-space kinetic energy densities per spin channel.
        xc_params: Exchange-correlation functional parameters.
        dens_threshold: Do not treat densities smaller than the threshold.

    Returns:
        Exchange-correlation energy potential.
    """
    exc, _, _, _ = get_xc(xc, n_spin, Nspin, dn_spin, tau, xc_params, dens_threshold)
    return exc


def get_vxc(xc, n_spin, Nspin, dn_spin=None, tau=None, xc_params=None, dens_threshold=0):
    """Get the exchange-correlation potential.

    This is a convenience function to interface :func:`~eminus.xc.utils.get_xc`.

    Args:
        xc: Exchange and correlation identifier.
        n_spin: Real-space electronic densities per spin channel.
        Nspin: Number of spin states.

    Keyword Args:
        dn_spin: Real-space gradient of densities per spin channel.
        tau: Real-space kinetic energy densities per spin channel.
        xc_params: Exchange-correlation functional parameters.
        dens_threshold: Do not treat densities smaller than the threshold.

    Returns:
        Exchange-correlation energy density.
    """
    _, vxc, vsigma, vtau = get_xc(xc, n_spin, Nspin, dn_spin, tau, xc_params, dens_threshold)
    return vxc, vsigma, vtau


def parse_functionals(xc):
    """Parse exchange-correlation functional strings to the internal format.

    Args:
        xc: Exchange and correlation identifier, separated by a comma.

    Returns:
        Exchange and correlation string.
    """
    # Check for combined aliases
    try:
        # Remove underscores when looking up in the dictionary
        xc_ = xc.replace("_", "")
        xc = ALIAS[xc_]
    except KeyError:
        pass

    # Parse functionals
    functionals = []
    for f in xc.split(","):
        if ":" in f or f in IMPLEMENTED:
            f_xc = f
        elif not f:
            f_xc = "mock_xc"
        else:
            try:
                # Remove underscores when looking up in the dictionary
                f_ = f.replace("_", "")
                f_xc = XC_MAP[f_]
            except KeyError:
                log.exception(f'No functional found for "{f}".')
                raise
        functionals.append(f_xc)

    # If only one or no functional has been parsed append with mock functionals
    functionals.extend("mock_xc" for _ in range(2 - len(functionals)))
    return functionals


def parse_xc_type(xc):
    """Parse functional strings to identify the corresponding functional type.

    Args:
        xc: Exchange and correlation identifier, separated by a comma.

    Returns:
        Functional type.
    """
    xc_type = []
    for func in xc:
        if ":" in func:
            xc_id = func.split(":")[-1]
            # Try to parse the functional using pylibxc at first
            try:
                family = parse_xc_libxc(xc_id)
            # Otherwise parse it with PySCF
            except (ImportError, AssertionError):
                family = parse_xc_pyscf(xc_id)

            if family == 1:
                xc_type.append("lda")
            elif family == 2:
                xc_type.append("gga")
            elif family == 4:
                xc_type.append("meta-gga")
            else:
                msg = "Unsupported functional family."
                raise NotImplementedError(msg)
        # Fall back to internal xc functionals
        elif "gga" in func:
            xc_type.append("gga")
        else:
            xc_type.append("lda")

    # When mixing functional types use the higher level of theory
    if xc_type[0] != xc_type[1]:
        log.warning("Detected mixing of different functional types.")
        if "meta-gga" in xc_type:
            return "meta-gga"
        return "gga"
    return xc_type[0]


def parse_xc_libxc(xc_id):
    """Parse functional type by its ID using pylibxc.

    Args:
        xc_id: Functional ID or identifier.

    Returns:
        Functional type.
    """
    if not config.use_pylibxc:
        raise AssertionError
    import pylibxc

    if not xc_id.isdigit():
        xc_id = pylibxc.util.xc_functional_get_number(xc_id)

    func = pylibxc.LibXCFunctional(int(xc_id), 1)
    if func._needs_laplacian:
        msg = "meta-GGAs that need a laplacian are not supported."
        raise NotImplementedError(msg)
    return func.get_family()


def parse_xc_pyscf(xc_id):
    """Parse functional type by its ID using PySCF.

    Args:
        xc_id: Functional ID or identifier.

    Returns:
        Functional type.
    """
    from pyscf.dft.libxc import is_gga, is_lda, is_meta_gga, needs_laplacian, XC_CODES

    if not xc_id.isdigit():
        xc_id = XC_CODES[xc_id.upper()]

    if needs_laplacian(int(xc_id)):
        msg = "meta-GGAs that need a laplacian are not supported."
        raise NotImplementedError(msg)
    # Use the same values as in parse_xc_libxc
    if is_lda(xc_id):
        return 1
    if is_gga(xc_id):
        return 2
    if is_meta_gga(xc_id):
        return 4
    return -1


def get_xc_defaults(xc):
    """Get the default parameters and values for a given set of functionals.

    Args:
        xc: Exchange and correlation identifier, separated by a comma.

    Returns:
        Default parameters and values.
    """
    if isinstance(xc, str):
        xc = parse_functionals(xc)

    # Names of special kewyword arguments that should not be used in xc_params
    SPECIAL_NAMES = ["dn_spin", "Nspin"]

    params = {}
    for func in xc:
        # If pylibxc functionals are used determine the default values from it
        if ":" in func:
            # This only works for pylibxc, not with PySCF
            if not config.use_pylibxc or "pylibxc" not in sys.modules:
                msg = "ext_params only work with pylibxc as the libxc backend, not with pyscf."
                raise NotImplementedError(msg)
            from pylibxc import LibXCFunctional

            fxc = func.split(":")[-1]
            try:
                f_xc = LibXCFunctional(int(fxc), 1)
            except ValueError:
                f_xc = LibXCFunctional(fxc, 1)
            fxc_params = dict(zip(f_xc.get_ext_param_names(), f_xc.get_ext_param_default_values()))

        # Analyze the signature for implemented functionals
        if func in IMPLEMENTED:
            sig = inspect.signature(IMPLEMENTED[func])
            fxc_params = {
                param.name: param.default
                for param in sig.parameters.values()
                if param.default is not inspect.Parameter.empty
            }

        # Remove special names from the parsed parameters
        for special in SPECIAL_NAMES:
            if special in fxc_params:
                del fxc_params[special]

        # Append all parameters, warn if a parameter has been used before
        for name in fxc_params:
            if name in params:
                log.warning(
                    f'External parameter "{name}" is used in the exchange AND correlation part. '
                    "It will be passed to both functionals if used in xc_params."
                )
            params[name] = fxc_params[name]
    return params


def get_zeta(n_spin):
    """Calculate the relative spin polarization.

    Args:
        n_spin: Real-space electronic densities per spin channel.

    Returns:
        Relative spin polarization.
    """
    # If only one spin is given return an array of ones as if the density only is in one channel
    if len(n_spin) == 1:
        return np.ones_like(n_spin[0])
    return (n_spin[0] - n_spin[1]) / (n_spin[0] + n_spin[1])


def mock_xc(n, Nspin=1, **kwargs):
    """Mock exchange-correlation functional with no effect (spin-paired).

    Args:
        n: Real-space electronic density.
        Nspin: Number of spin states.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Mock exchange-correlation energy density and potential.
    """
    zeros = np.zeros_like(n)
    return zeros, np.array([zeros] * Nspin), None


#: Map functional names with their respective implementation.
IMPLEMENTED = {
    i.__name__: i
    for i in (
        mock_xc,
        gga_c_chachiyo,
        gga_c_chachiyo_spin,
        gga_c_pbe,
        gga_c_pbe_spin,
        gga_c_pbe_sol,
        gga_c_pbe_sol_spin,
        gga_x_chachiyo,
        gga_x_chachiyo_spin,
        gga_x_pbe,
        gga_x_pbe_spin,
        gga_x_pbe_sol,
        gga_x_pbe_sol_spin,
        lda_x,
        lda_x_spin,
        lda_c_pw,
        lda_c_pw_spin,
        lda_c_pw_mod,
        lda_c_pw_mod_spin,
        lda_c_vwn,
        lda_c_vwn_spin,
        lda_c_chachiyo,
        lda_c_chachiyo_spin,
        lda_c_chachiyo_mod,
        lda_c_chachiyo_mod_spin,
        lda_xc_gdsmfb,
        lda_xc_gdsmfb_spin,
    )
}

#: Map functional shorthands to the actual functional names.
XC_MAP = {
    # lda_x
    "1": "lda_x",
    "s": "lda_x",
    "lda": "lda_x",
    "slater": "lda_x",
    # lda_c_vwn
    "7": "lda_c_vwn",
    "vwn": "lda_c_vwn",
    "vwn5": "lda_c_vwn",
    # lda_c_pw
    "12": "lda_c_pw",
    "pw": "lda_c_pw",
    "pw92": "lda_c_pw",
    # lda_c_pw_mod
    "13": "lda_c_pw_mod",
    "pwmod": "lda_c_pw_mod",
    "pw92mod": "lda_c_pw_mod",
    # gga_x_pbe
    "101": "gga_x_pbe",
    "pbex": "gga_x_pbe",
    # gga_x_pbe_sol
    "116": "gga_x_pbe_sol",
    "pbesolx": "gga_x_pbe_sol",
    # gga_c_pbe
    "130": "gga_c_pbe",
    "pbec": "gga_c_pbe",
    # gga_c_pbe_sol
    "133": "gga_c_pbe_sol",
    "pbesolc": "gga_c_pbe_sol",
    # lda_c_chachiyo
    "287": "lda_c_chachiyo",
    "chachiyo": "lda_c_chachiyo",
    # gga_x_chachiyo
    "298": "gga_x_chachiyo",
    "chachiyox": "gga_x_chachiyo",
    # lda_c_chachiyo_mod
    "307": "lda_c_chachiyo_mod",
    "chachiyomod": "lda_c_chachiyo_mod",
    # gga_c_chachiyo
    "309": "gga_c_chachiyo",
    "chachiyoc": "gga_c_chachiyo",
    # lda_xc_gdsmfb
    "577": "lda_xc_gdsmfb",
    "gdsmfb": "lda_xc_gdsmfb",
}

#: Dictionary of common functional aliases.
ALIAS = {
    "svwn": "slater,vwn5",
    "svwn5": "slater,vwn5",
    "spw92": "slater,pw92mod",
    "pbe": "pbex,pbec",
    "pbesol": "pbesolx,pbesolc",
    "chachiyo": "chachiyox,chachiyoc",
}
