#!/usr/bin/env python3
'''Workflow functions that combine functions to generate orbitals.'''
from .dft import get_psi
from .io import write_cube
from .localizer import get_FLO, get_FO
from .logger import log


def KSO(scf, write_cubes=False, **kwargs):
    '''Generate Kohn-Sham orbitals and optionally save them as cube files.

    Reference: Phys. Rev. 140, A1133.

    Args:
        scf: SCF object.

    Keyword Args:
        write_cubes (bool): Write orbitals to cube files.

    Returns:
        ndarray: Real-space Kohn-Sham orbitals.
    '''
    atoms = scf.atoms

    # Calculate eigenfunctions and transform to real-space
    kso = atoms.I(get_psi(scf, scf.W))
    if write_cubes:
        cube_writer(atoms, 'KSO', kso)
    return kso


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
    # Lazy import extras
    from .extras.fods import get_fods, remove_core_fods
    atoms = scf.atoms

    # Calculate eigenfunctions
    kso = get_psi(scf, scf.W)
    if fods is None:
        fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)

    # The FO functions needs orbitals in reciprocal space as input
    fo = get_FO(atoms, kso, fods)
    if write_cubes:
        cube_writer(atoms, 'FO', fo)
    return fo


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
    # Lazy import extras
    from .extras.fods import get_fods, remove_core_fods
    atoms = scf.atoms

    # Calculate eigenfunctions
    kso = get_psi(scf, scf.W)
    if fods is None:
        fods = get_fods(atoms)
    fods = remove_core_fods(atoms, fods)

    # The FLO functions needs orbitals in reciprocal space as input
    flo = get_FLO(atoms, kso, fods)
    if write_cubes:
        cube_writer(atoms, 'FLO', flo)
    return flo


def cube_writer(atoms, type, orbitals):
    '''Simple cube file writer function.

    Args:
        atoms: Atoms object.
        type (str): Orbital type for the cube file names.
        orbitals (ndarray): Real-space orbitals.
    '''
    # Create the system name
    name = ''
    for ia in sorted(set(atoms.atom)):
        # Skip the number of atoms if it is equal to one
        na = atoms.atom.count(ia)
        name += f'{ia}{na if na > 1 else ""}'

    n_spin = ''
    for spin in range(atoms.Nspin):
        for i in range(atoms.Nstate):
            if atoms.f[spin, i] > 0:
                if atoms.Nspin > 1:
                    n_spin = f'_spin_{spin}'
                filename = f'{name}_{type}_{i}{n_spin}.cube'
                log.info(f'Write {filename}...')
                write_cube(atoms, filename, orbitals[spin, :, i])
    return
