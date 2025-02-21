# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Dispersion correction interface.

All necessary dependencies to use this extra can be installed with::

    pip install eminus[dispersion]
"""

# Import the DFT-D3 C extension beforehand, if one doesn't do this the dispersion energies are wrong
try:  # noqa: SIM105
    import dftd3._libdftd3  # noqa: F401
except ImportError:
    pass
import numpy as np

from ..data import SYMBOL2NUMBER
from ..logger import log


def get_Edisp(scf, version="d3bj", atm=True, xc=None):
    """Calculate the DFT-D3 dispersion correction energy.

    Reference: J. Chem. Phys. 132, 154104.

    Args:
        scf: SCF object.

    Keyword Args:
        version: Damping function, can be one of "d3bj", "d3bjm", "d3zero", "d3zerom", "d3op".
        atm: Whether to use three-body dispersion energies.
        xc: Overwrite the functional keyword if the automatic detection does not work.

    Returns:
        Dispersion correction energy.
    """
    try:
        from dftd3.interface import (
            DispersionModel,
            ModifiedRationalDampingParam,
            ModifiedZeroDampingParam,
            OptimizedPowerDampingParam,
            RationalDampingParam,
            ZeroDampingParam,
        )
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[dispersion]".\n\n'
        )
        raise
    # Dictionary of implemented dispersion models
    dispersion_version = {
        "d3bj": RationalDampingParam,
        "d3bjm": ModifiedRationalDampingParam,
        "d3zero": ZeroDampingParam,
        "d3zerom": ModifiedZeroDampingParam,
        "d3op": OptimizedPowerDampingParam,
    }

    # Set up input parameters
    # The dispersion correction is only geometry dependent which makes handling the input easy
    atoms = scf.atoms
    positions = atoms.pos
    numbers = np.array([SYMBOL2NUMBER[ia] for ia in atoms.atom])
    # Try to determine the method keyword
    if xc is None:
        if scf.xc_type == "lda":
            method = "slaterdiracexchange"
            version = "d3zero"
            log.warning("Dispersion correction for LDA functionals only support d3zero.")
        elif "pbe_sol" in "".join(scf.xc):
            method = "pbesol"
        elif "pbe" in "".join(scf.xc):
            method = "pbe"
        else:
            method = "pbe"
            log.warning(
                "Functional for the dispersion correction could not be detected, continue"
                " with pbe. You may need to overwrite the functional keyword manually."
            )
    else:
        method = xc

    # Set up the dispersion model object with periodic boundary conditions
    model = DispersionModel(numbers, positions, lattice=atoms.a, periodic=np.array([1, 1, 1]))
    # Calculate the dispersion energy, neglecting the gradient
    res = model.get_dispersion(dispersion_version[version](method=method, atm=atm), grad=False)
    scf.energies.Edisp = res["energy"]
    return res["energy"]
