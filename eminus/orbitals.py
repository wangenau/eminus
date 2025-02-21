# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Workflow functions that combine functions to generate orbitals."""

from .dft import get_psi
from .io import write_cube
from .localizer import get_FLO, get_FO, get_scdm, get_wannier
from .logger import log
from .tools import orbital_center


def KSO(scf, write_cubes=False, **kwargs):
    """Generate Kohn-Sham orbitals and optionally save them as CUBE files.

    Reference: Phys. Rev. 140, A1133.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes: Write orbitals to CUBE files.
        **kwargs: Throwaway arguments.

    Returns:
        Real-space Kohn-Sham orbitals.
    """
    atoms = scf.atoms

    # Calculate eigenfunctions and transform to real-space
    kso = atoms.I(get_psi(scf, scf.W))
    if write_cubes:
        cube_writer(atoms, "KSO", kso)
    return kso


def FO(scf, write_cubes=False, fods=None, guess="wannier"):
    """Generate Fermi orbitals and optionally save them as CUBE files.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes: Write orbitals to CUBE files.
        fods: Fermi-orbital descriptors.
        guess: Guess to generate FODs if none are given. Can be "Wannier" (default) or "PyCOM".

    Returns:
        Real-space Fermi orbitals.
    """
    # Lazy import extras
    from .extras.fods import get_fods, remove_core_fods

    atoms = scf.atoms

    # Calculate eigenfunctions
    kso = get_psi(scf, scf.W)
    if fods is None and guess == "wannier":
        wo = WO(scf)
        fods = orbital_center(atoms, wo[0])
    if fods is None and guess == "pycom":
        fods = get_fods(atoms)
        fods = remove_core_fods(atoms, fods)

    # The FO functions need orbitals in reciprocal space as input
    fo = get_FO(atoms, kso, fods)
    if write_cubes:
        cube_writer(atoms, "FO", fo)
    return fo


def FLO(scf, write_cubes=False, fods=None, guess="wannier"):
    """Generate Fermi-Loewdin orbitals and optionally save them as CUBE files.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes: Write orbitals to CUBE files.
        fods: Fermi-orbital descriptors.
        guess: Guess to generate FODs if none are given. Can be "Wannier" (default) or "PyCOM".

    Returns:
        Real-space Fermi-Loewdin orbitals.
    """
    # Lazy import extras
    from .extras.fods import get_fods, remove_core_fods

    atoms = scf.atoms

    # Calculate eigenfunctions
    kso = get_psi(scf, scf.W)

    guess = guess.lower()
    if fods is None and guess == "wannier":
        wo = WO(scf)
        fods = orbital_center(atoms, wo[0])
    if fods is None and guess == "pycom":
        fods = get_fods(atoms)
        fods = remove_core_fods(atoms, fods)

    # The FLO functions need orbitals in reciprocal space as input
    flo = get_FLO(atoms, kso, fods)
    if write_cubes:
        cube_writer(atoms, "FLO", flo)
    return flo


def WO(scf, write_cubes=False, precondition=True):
    """Generate Wannier orbitals and optionally save them as CUBE files.

    Reference: Phys. Rev. B 59, 9703.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes: Write orbitals to CUBE files.
        precondition: Precondition by calculating SCDMs as the initial guess.

    Returns:
        Real-space Wannier orbitals.
    """
    atoms = scf.atoms

    # Calculate eigenfunctions/initial guess orbitals and transform to real-space
    if precondition:
        psi = SCDM(scf)
    else:
        psi = atoms.I(get_psi(scf, scf.W))

    wo = get_wannier(atoms, psi)
    if write_cubes:
        cube_writer(atoms, "WO", wo)
    return wo


def SCDM(scf, write_cubes=False):
    """Generate SCDM orbitals and optionally save them as CUBE files.

    Reference: J. Chem. Theory Comput. 11, 1463.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes: Write orbitals to CUBE files.

    Returns:
         Real-space SCDM orbitals.
    """
    atoms = scf.atoms

    # Calculate eigenfunctions
    kso = get_psi(scf, scf.W)
    scdm = get_scdm(atoms, kso)
    if write_cubes:
        cube_writer(atoms, "SCDM", scdm)
    return scdm


def cube_writer(atoms, orb_type, orbitals):
    """Simple CUBE file writer function.

    Args:
        atoms: Atoms object.
        orb_type: Orbital orb_type for the CUBE file names.
        orbitals: Real-space orbitals.
    """
    # Create the system name
    name = ""
    for ia in sorted(set(atoms.atom)):
        # Skip the number of atoms if it is equal to one
        na = atoms.atom.count(ia)
        name += f"{ia}{na if na > 1 else ''}"

    n_spin = ""
    for ik in range(atoms.kpts.Nk):
        for spin in range(atoms.occ.Nspin):
            for i in range(atoms.occ.Nstate):
                if atoms.occ.f[ik, spin, i] > 0:
                    if atoms.unrestricted:
                        n_spin = f"_spin_{spin}"
                    filename = f"{name}_{orb_type}_k{ik}_{i}{n_spin}.cube"
                    log.info(f"Write {filename}...")
                    write_cube(atoms, filename, orbitals[ik][spin, :, i])
