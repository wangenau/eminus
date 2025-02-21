# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Fermi-orbital descriptor generation.

All necessary dependencies to use this extra can be installed with::

    pip install eminus[fods]
"""

import numpy as np
from scipy.linalg import norm

from ..data import SYMBOL2NUMBER
from ..logger import log
from ..units import bohr2ang


def get_localized_orbitals(mf, loc, Nit=1000, seed=1234):
    """Generate localized orbitals with additional simple stability analysis.

    Same as implemented in PyFLOSIC2.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        mf: PySCF SCF object.
        loc: Localization method.

    Keyword Args:
        Nit: Number of tries to get a solution with positive eigenvalues.
        seed: Seed to initialize the random number generator.

    Returns:
        Localized occupied orbital coefficients per spin channel.
    """
    from pyscf.lo import boys, edmiston, pipek

    loc_dict = {
        "ER": edmiston.EdmistonRuedenberg,
        "FB": boys.Boys,
        "GPM": pipek.PipekMezey,
        "PM": pipek.PipekMezey,
    }

    rng = np.random.default_rng(seed=seed)
    Nspin = mf.mo_occ.ndim

    loc_orb = []
    # Localize each spin channel separately
    for s in range(Nspin):
        # Initialize the localizer object
        if Nspin == 2:
            localizer = loc_dict[loc](mf.mol, mf.mo_coeff[s][:, mf.mo_occ[s] > 0])
        else:
            localizer = loc_dict[loc](mf.mol, mf.mo_coeff[:, mf.mo_occ > 0])

        # Set the population method in generalized PM to Becke charges
        if loc == "GPM":
            localizer.pop_method = "becke"

        for _ in range(Nit):
            tmp_orb = localizer.kernel()
            # Calculate the eigenvalues of the Hessian
            _, _, h_diag = localizer.gen_g_hop(u=np.eye(len(tmp_orb[0])))
            # If there are no eigenvalues or all of them are positive break the loop
            if len(h_diag) == 0 or np.min(h_diag) > 0:
                break
            # If not continue with randomly "displaced" orbitals
            noise = rng.normal(scale=5e-4, size=tmp_orb.shape)
            localizer.mo_coeff = tmp_orb + noise
        loc_orb.append(tmp_orb)
    return loc_orb


def get_fods(obj, basis="pc-1", loc="FB"):
    """Generate FOD positions using the PyCOM method.

    Reference: J. Comput. Chem. 40, 2843.

    Args:
        obj: Atoms or SCF object.

    Keyword Args:
        basis: Basis set for the DFT calculation.
        loc: Localization method.
        elec_symbols: Identifier for up and down FODs.

    Returns:
        FOD positions.
    """
    try:
        from pyscf.gto import Mole
        from pyscf.scf import RKS, UKS
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[fods]".\n\n'
        )
        raise

    atoms = obj._atoms
    loc = loc.upper()

    # Convert to Angstrom for PySCF
    pos = bohr2ang(atoms.pos)
    # Build the PySCF input format
    atom_pyscf = list(zip(atoms.atom, pos))

    # Do the PySCF DFT calculation
    # Use Mole.build() over M() since the parse_arg option breaks testing with pytest
    mol = Mole(atom=atom_pyscf, basis=basis, spin=atoms.spin).build(parse_arg=False)
    if atoms.unrestricted:
        mf = UKS(mol=mol)
    else:
        mf = RKS(mol=mol)
    mf.verbose = 0
    mf.kernel()

    # Get the localized orbital coefficients
    loc_orb = get_localized_orbitals(mf, loc)
    # Calculate the COMs
    loc_com = []
    ao = mf._numint.eval_ao(mf.mol, mf.grids.coords)
    for s in range(atoms.occ.Nspin):
        phi = ao @ loc_orb[s]
        dens = phi.conj() * phi * mf.grids.weights[:, None]
        loc_com.append(dens.T @ mf.grids.coords)
    return loc_com


def split_fods(atom, pos, elec_symbols=("X", "He")):
    """Split atom and FOD coordinates.

    Args:
        atom: Atom symbols.
        pos: Atom positions.

    Keyword Args:
        elec_symbols: Identifier for up and down FODs.

    Returns:
        Atom types, the respective coordinates, and FOD positions.
    """
    pos_fod_up = []
    pos_fod_dn = []
    # Iterate in reverse order because we may delete elements
    for ia in range(len(pos) - 1, -1, -1):
        if atom[ia] in elec_symbols:
            if atom[ia] in elec_symbols[0]:
                pos_fod_up.append(pos[ia])
            if len(elec_symbols) > 1 and atom[ia] in elec_symbols[1]:
                pos_fod_dn.append(pos[ia])
            pos = np.delete(pos, ia, axis=0)
            del atom[ia]

    pos_fod = [np.asarray(pos_fod_up), np.asarray(pos_fod_dn)]
    return atom, pos, pos_fod


def remove_core_fods(obj, fods):
    """Remove core FODs from a set of FOD coordinates.

    Args:
        obj: Atoms or SCF object.
        fods: FOD positions.

    Returns:
        Valence FOD positions.
    """
    atoms = obj._atoms

    # If the number of valence electrons is the same as the number of FODs, do nothing
    atoms.kpts._assert_gamma_only()
    if not atoms.unrestricted and len(fods[0]) * 2 == np.sum(atoms.occ.f[0]):
        return fods
    if (
        atoms.unrestricted
        and len(fods[0]) == np.sum(atoms.occ.f[0, 0])
        and len(fods[1]) == np.sum(atoms.occ.f[0, 1])
    ):
        return fods

    for s in range(atoms.occ.Nspin):
        for ia in range(atoms.Natoms):
            n_core = SYMBOL2NUMBER[atoms.atom[ia]] - atoms.Z[ia]
            # In the spin-paired case two electrons are one state
            # Since only core states are removed in pseudopotentials this value is divisible by 2
            # +1 to account for uneven amounts of core FODs (like in hydrogen)
            n_core = (n_core + 1) // 2
            dist = norm(fods[s] - atoms.pos[ia], axis=1)
            idx = np.argsort(dist)
            # Remove core FODs with the smallest distance to the core
            fods[s] = np.delete(fods[s], idx[:n_core], axis=0)
    return fods
