#!/usr/bin/env python3
'''
Workflow functions that combine functions to get one property.
'''
from eminus.atoms_io import write_cube
from eminus.localizer import get_FLOs, get_FOs
from eminus.scf import get_psi
from .fods import get_fods, remove_core_fods


def KSO(atoms, write_cubes=True, **kwargs):
    '''Generate Kohn-Sham orbitals and optionally save them as cube files.

    Args:
        atoms :
            Atoms object.

    Kwargs:
        write_cubes : bool
            Write orbitals to cube files.

    Returns:
        Kohn-Sham orbitals as an array.
    '''
    KSOs = get_psi(atoms, atoms.W)
    name = ''
    for ia in set(atoms.atom):
        name += f'{ia}{atoms.atom.count(ia)}'
    if write_cubes:
        for i in range(atoms.Ns):
            write_cube(atoms, KSOs[:, i], f'{name}_KSO_{i}.cube')
    return KSOs


def FO(atoms, write_cubes=True, fods=None):
    '''Generate Fermi orbitals and optionally save them as cube files.

    Args:
        atoms :
            Atoms object.

    Kwargs:
        write_cubes : bool
            Write orbitals to cube files.

        fods : array
            Fermi-orbital descriptors.

    Returns:
        Fermi orbitals as an array.
    '''
    KSOs = get_psi(atoms, atoms.W)
    if fods is None:
        fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)
    FOs = get_FOs(atoms, KSOs, fods)
    name = ''
    for ia in set(atoms.atom):
        name += f'{ia}{atoms.atom.count(ia)}'
    if write_cubes:
        for i in range(atoms.Ns):
            write_cube(atoms, FOs[:, i], f'{name}_FO_{i}.cube')
    return FOs


def FLO(atoms, write_cubes=True, fods=None):
    '''Generate Fermi-Löwdin orbitals and optionally save them as cube files.

    Args:
        atoms :
            Atoms object.

    Kwargs:
        write_cubes : bool
            Write orbitals to cube files.

        fods : array
            Fermi-orbital descriptors.

    Returns:
        Fermi-Löwdin orbitals as an array.
    '''
    KSOs = get_psi(atoms, atoms.W)
    if fods is None:
        fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)
    FLOs = get_FLOs(atoms, KSOs, fods)
    name = ''
    for ia in set(atoms.atom):
        name += f'{ia}{atoms.atom.count(ia)}'
    if write_cubes:
        for i in range(atoms.Ns):
            write_cube(atoms, FLOs[:, i], f'{name}_FLO_{i}.cube')
    return FLOs
