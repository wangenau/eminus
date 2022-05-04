#!/usr/bin/env python3
'''Workflow functions that combine functions to generate orbitals.'''
from .dft import get_psi
from .filehandler import write_cube
from .localizer import get_FLO, get_FO


def KSO(scf, write_cubes=False, **kwargs):
    '''Generate Kohn-Sham orbitals and optionally save them as cube files.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.

    Returns:
        ndarray: Real-space Kohn-Sham orbitals.
    '''
    atoms = scf.atoms
    # Calculate eigenfunctions and transform to real-space
    KSO = atoms.I(get_psi(scf, scf.W))
    if write_cubes:
        name = ''
        for ia in set(atoms.atom):
            name += f'{ia}{atoms.atom.count(ia)}'
        for i in range(atoms.Ns):
            write_cube(atoms, KSO[:, i], f'{name}_KSO_{i}.cube')
    return KSO


def FO(scf, write_cubes=False, fods=None):
    '''Generate Fermi orbitals and optionally save them as cube files.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.
        fods (ndarray): Fermi-orbital descriptors.

    Returns:
        ndarray: Real-space Fermi orbitals.
    '''
    # Lazy import addons
    from .addons.fods import get_fods, remove_core_fods
    atoms = scf.atoms
    # Calculate eigenfunctions
    KSO = get_psi(scf, scf.W)
    if fods is None:
        fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)
    # The FO functions needs orbitals in reciprocal space as input
    FO = get_FO(atoms, KSO, fods)
    if write_cubes:
        name = ''
        for ia in set(atoms.atom):
            name += f'{ia}{atoms.atom.count(ia)}'
        for i in range(atoms.Ns):
            write_cube(atoms, FO[:, i], f'{name}_FO_{i}.cube')
    return FO


def FLO(scf, write_cubes=False, fods=None):
    '''Generate Fermi-Löwdin orbitals and optionally save them as cube files.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.
        fods (ndarray): Fermi-orbital descriptors.

    Returns:
        ndarray: Real-space Fermi-Löwdin orbitals.
    '''
    # Lazy import addons
    from .addons.fods import get_fods, remove_core_fods
    atoms = scf.atoms
    # Calculate eigenfunctions
    KSO = get_psi(scf, scf.W)
    if fods is None:
        fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)
    # The FLO functions needs orbitals in reciprocal space as input
    FLO = get_FLO(atoms, KSO, fods)
    if write_cubes:
        name = ''
        for ia in set(atoms.atom):
            name += f'{ia}{atoms.atom.count(ia)}'
        for i in range(atoms.Ns):
            write_cube(atoms, FLO[:, i], f'{name}_FLO_{i}.cube')
    return FLO
