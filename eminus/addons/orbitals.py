#!/usr/bin/env python3
'''Workflow functions that combine functions to generate orbitals.'''
from .fods import get_fods, remove_core_fods
from ..filehandler import write_cube
from ..localizer import get_FLO, get_FO
from ..scf import get_psi


def KSO(atoms, write_cubes=False, **kwargs):
    '''Generate Kohn-Sham orbitals and optionally save them as cube files.

    Args:
        atoms: Atoms object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.

    Returns:
        array: Real-space Kohn-Sham orbitals.
    '''
    # Calculate eigenfunctions and transform to real-space
    KSO = atoms.I(get_psi(atoms, atoms.W))
    if write_cubes:
        name = ''
        for ia in set(atoms.atom):
            name += f'{ia}{atoms.atom.count(ia)}'
        for i in range(atoms.Ns):
            write_cube(atoms, KSO[:, i], f'{name}_KSO_{i}.cube')
    return KSO


def FO(atoms, write_cubes=False, fods=None):
    '''Generate Fermi orbitals and optionally save them as cube files.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.
        fods (array): Fermi-orbital descriptors.

    Returns:
        array: Real-space Fermi orbitals.
    '''
    # Calculate eigenfunctions
    KSO = get_psi(atoms, atoms.W)
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


def FLO(atoms, write_cubes=False, fods=None):
    '''Generate Fermi-Löwdin orbitals and optionally save them as cube files.

    Reference: J. Chem. Phys. 153, 084104.

    Args:
        atoms: Atoms object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.
        fods (array): Fermi-orbital descriptors.

    Returns:
        array: Real-space Fermi-Löwdin orbitals.
    '''
    # Calculate eigenfunctions
    KSO = get_psi(atoms, atoms.W)
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
