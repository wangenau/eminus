#!/usr/bin/env python3
'''
Workflow functions that combine functions to get one property.
'''
from .atoms_io import write_cube
from .localizer import get_FLOs, get_FOs
from .scf import get_psi
from eminus.addons import get_fods, remove_core_fods


def FO(atoms, write_cubes=True):
    psi = get_psi(atoms, atoms.W)
    fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)
    FLOs = get_FOs(atoms, psi, fods)
    name = ''
    for ia in set(atoms.atom):
        name += f'{ia}{atoms.atom.count(ia)}'
    if write_cubes:
        for i in range(atoms.Ns):
            write_cube(atoms, FLOs[:, i], f'{name}_FO_{i}.cube')
    return FLOs


def FLO(atoms, write_cubes=True):
    psi = get_psi(atoms, atoms.W)
    fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)
    FLOs = get_FLOs(atoms, psi, fods)
    name = ''
    for ia in set(atoms.atom):
        name += f'{ia}{atoms.atom.count(ia)}'
    if write_cubes:
        for i in range(atoms.Ns):
            write_cube(atoms, FLOs[:, i], f'{name}_FLO_{i}.cube')
    return FLOs
