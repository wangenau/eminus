# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Collection of miscellaneous potentials."""

import inspect

import numpy as np
from scipy.linalg import norm

from .gth import init_gth_loc
from .logger import log


def get_pot_defaults(pot):
    """Get the default parameters and values for a given potential.

    Args:
        pot: Potential name.

    Returns:
        Default parameters and values.
    """
    if pot in IMPLEMENTED:
        sig = inspect.signature(IMPLEMENTED[pot])
        return {
            param.name: param.default
            for param in sig.parameters.values()
            if param.default is not inspect.Parameter.empty
        }
    return {}


def harmonic(scf, freq=2, **kwargs):
    """Harmonic potential.

    Can be used for quantum dot calculations.

    Args:
        scf: SCF object.

    Keyword Args:
        freq: Harmonic oscillator frequency.
        **kwargs: Throwaway arguments.

    Returns:
        Harmonic potential in real-space.
    """
    atoms = scf.atoms
    dr = norm(atoms.r - np.sum(atoms.a, axis=1) / 2, axis=1)
    Vharm = 0.5 * freq**2 * dr**2
    return atoms.Jdag(atoms.O(atoms.J(Vharm)))


def coulomb(scf, **kwargs):
    """All-electron Coulomb potential.

    Reference: Bull. Lebedev Phys. Inst. 42, 329.

    Args:
        scf: SCF object.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Coulomb potential in real-space.
    """
    atoms = scf.atoms

    Vcoul = np.zeros_like(atoms.G2)
    for isp in set(atoms.atom):
        # Sum up the structure factor for every species
        # Also get the charge, assuming all species have the same charge
        Sf = np.zeros(len(atoms.Sf[0]), dtype=complex)
        for ia in range(atoms.Natoms):
            if atoms.atom[ia] == isp:
                Sf += atoms.Sf[ia]
                Z = atoms.Z[ia]

        # Ignore the division by zero for the first elements
        # One could do some proper indexing with [1:] but indexing is slow
        with np.errstate(divide="ignore", invalid="ignore"):
            Vsp = -4 * np.pi * Z / atoms.G2
        Vsp[0] = 0
        Vcoul += np.real(atoms.J(Vsp * Sf))
    return Vcoul


def coulomb_lr(scf, alpha=100, **kwargs):
    """Long-range all-electron Coulomb potential.

    Reference: J. Comput. Phys. 117, 171.

    Args:
        scf: SCF object.

    Keyword Args:
        alpha: Convergence parameter.
        **kwargs: Throwaway arguments.

    Returns:
        Long-range Coulomb potential in real-space.
    """
    atoms = scf.atoms

    Vcoul = np.zeros_like(atoms.G2)
    for isp in set(atoms.atom):
        # Sum up the structure factor for every species
        # Also get the charge, assuming all species have the same charge
        Sf = np.zeros(len(atoms.Sf[0]), dtype=complex)
        for ia in range(atoms.Natoms):
            if atoms.atom[ia] == isp:
                Sf += atoms.Sf[ia]
                Z = atoms.Z[ia]

        # Ignore the division by zero for the first elements
        # One could do some proper indexing with [1:] but indexing is slow
        with np.errstate(divide="ignore", invalid="ignore"):
            Vsp = -4 * np.pi * Z * np.exp(-atoms.G2 / (4 * alpha**2)) / atoms.G2
        Vsp[0] = 0
        Vcoul += np.real(atoms.J(Vsp * Sf))
    return Vcoul


def ge(scf, **kwargs):
    """Starkloff-Joannopoulos local pseudopotential for germanium.

    Fourier-transformed by Tomas Arias.

    Reference: Phys. Rev. B 16, 5212.

    Args:
        scf: SCF object.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Germanium pseudopotential in real-space.
    """
    atoms = scf.atoms
    Z = 4  # This potential should only be used for germanium
    lamda = 18.5
    rc = 1.052
    Gm = np.sqrt(atoms.G2)

    with np.errstate(divide="ignore", invalid="ignore"):
        Vps = (
            -2
            * np.pi
            * np.exp(-np.pi * Gm / lamda)
            * np.cos(rc * Gm)
            * (Gm / lamda)
            / (1 - np.exp(-2 * np.pi * Gm / lamda))
        )
        for n in range(5):
            Vps = Vps + (-1) ** n * np.exp(-lamda * rc * n) / (1 + (n * lamda / Gm) ** 2)
        Vps = Vps * 4 * np.pi * Z / Gm**2 * (1 + np.exp(-lamda * rc)) - 4 * np.pi * Z / Gm**2

    # Special case for G=(0,0,0)
    n = np.arange(1, 5)
    Vps[0] = (
        4
        * np.pi
        * Z
        * (1 + np.exp(-lamda * rc))
        * (
            rc**2 / 2
            + 1 / lamda**2 * (np.pi**2 / 6 + np.sum((-1) ** n * np.exp(-lamda * rc * n) / n**2))
        )
    )

    Sf = np.sum(atoms.Sf, axis=0)
    return atoms.J(Vps * Sf)


def init_pot(scf, pot_params=None):
    """Handle and initialize potentials.

    Args:
        scf: SCF object.

    Keywords Args:
        pot_params: Potential parameters.

    Returns:
        Potential in real-space.
    """
    if pot_params is None:
        pot_params = {}
    try:
        pot = IMPLEMENTED[scf.pot](scf, **pot_params)
    except KeyError:
        log.exception(f'No potential found for "{scf.pot}".')
        raise
    return pot


#: Map potential names with their respective implementation.
IMPLEMENTED = {
    "harmonic": harmonic,
    "coulomb": coulomb,
    "lr": coulomb_lr,
    "ge": ge,
    "gth": init_gth_loc,
}
