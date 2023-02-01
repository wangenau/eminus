#!/usr/bin/env python3
'''GTH file handling.'''
import inspect
import pathlib

import numpy as np

from ..logger import log


def read_gth(atom, charge=None, psp_path=None):
    '''Read GTH files for a given atom.

    Reference: Phys. Rev. B 54, 1703.

    Args:
        atom (str): Atom name.

    Keyword Args:
        charge (int): Valence charge.
        psp_path (str): Path to GTH pseudopotential files. Defaults to installation_path/pade/.

    Returns:
        dict: GTH parameters.
    '''
    if psp_path is None:
        file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
        psp_path = file_path.parent.joinpath('pade')

    if charge is not None:
        f_psp = psp_path.joinpath(f'{atom}-q{charge}')
    else:
        files = sorted(psp_path.glob(f'{atom}-q*'))
        try:
            f_psp = pathlib.Path(files[0])
        except IndexError:
            log.warning(f'There is no GTH pseudopotential in {psp_path} for "{atom}"')
            return mock_gth()
        if len(files) > 1:
            log.info(f'Multiple pseudopotentials found for "{atom}". '
                     f'Continue with "{f_psp.name}".')

    psp = {}
    cloc = np.zeros(4)
    rp = np.zeros(4)
    Nproj_l = np.zeros(4, dtype=int)
    h = np.zeros((4, 3, 3))
    try:
        with open(f_psp, 'r') as fh:
            atom = fh.readline()
            # Skip the first line, since we don't need the atom symbol here. If needed, use
            # psp['atom'] = atom.split()[0]  # Atom symbol
            N_all = fh.readline().split()
            psp['Zion'] = sum([int(N) for N in N_all])  # Ionic charge
            loc = fh.readline().split()
            psp['rloc'] = float(loc[0])  # Range of local Gaussian charge distribution
            # Skip the number of local coefficients, since we don't need it. If needed, use
            # psp['n_c_local'] = int(loc[1])  # Number of local coefficients
            for i, val in enumerate(loc[2:]):
                cloc[i] = float(val)
            psp['cloc'] = cloc  # Coefficients for the local part
            lmax = int(fh.readline().split()[0])
            psp['lmax'] = lmax  # Maximal angular momentum in the non-local part
            for i in range(lmax):
                proj = fh.readline().split()
                rp[i], Nproj_l[i] = float(proj[0]), int(proj[1])
                for k, val in enumerate(proj[2:]):
                    h[i, 0, k] = float(val)
                for j in range(1, Nproj_l[i]):
                    proj = fh.readline().split()
                    for k, val in enumerate(proj):
                        h[i, j, j + k] = float(val)
                # Copy upper triangle elements to the lower triangle
                for jtmp in range(3):
                    for ktmp in range(i, 3):
                        h[i, ktmp, jtmp] = h[i, jtmp, ktmp]
            psp['rp'] = rp  # Projector radius for each angular momentum
            psp['Nproj_l'] = Nproj_l  # Number of non-local projectors
            psp['h'] = h  # Projector coupling coefficients per AM channel
    except FileNotFoundError:
        log.warning(f'There is no GTH pseudopotential for "{f_psp.name}"')
        return mock_gth()
    return psp


def mock_gth():
    '''Create a mock dictionary with all-zeros, for atom species with no pseudopotential file.

    Returns:
        dict: GTH parameters (all zero).
    '''
    psp = {}
    psp['Zion'] = 0
    psp['rloc'] = 0
    psp['cloc'] = np.zeros(4)
    psp['lmax'] = 0
    psp['rp'] = np.zeros(4)
    psp['Nproj_l'] = np.zeros(4, dtype=int)
    psp['h'] = np.zeros((4, 3, 3))
    return psp
