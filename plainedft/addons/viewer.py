#!/usr/bin/env python3
'''
Viewer functions for Jupyter notebooks.
'''
import numpy as np
try:
    from nglview import NGLWidget, TextStructure
except ImportError:
    print('ERROR: Necessary addon dependecies not found. '
          'To use this module, install the package with addons, e.g., with "pip install .[addons]"')

from plainedft.atoms import create_pdb, read_cube, read_xyz


# Adapted from https://github.com/MolSSI/QCFractal/issues/374
def view(filename, isovalue=0.01, **kwargs):
    '''Display molecules and orbitals.

    Args:
        filename : str
            Input file path/name.

    Kwargs:
        isovalue : float
            Isovalue for sizing orbitals.

    Returns:
        Viewable object as a NGLWidget.
    '''
    if isinstance(isovalue, str):
        isovalue = float(isovalue)
    view = NGLWidget(**kwargs)
    view._set_size('400px', '400px')

    if filename.endswith('.xyz'):
        # Atoms
        atom, X = read_xyz(filename)
        atom, X, X_fod = split_atom_and_fod(atom, X)
        view.add_component(TextStructure(create_pdb(atom, X)))
        view[0].clear()
        view[0].add_ball_and_stick()
        # FODs
        if len(X_fod) > 0:
            view.add_component(TextStructure(create_pdb(['X'] * len(X_fod), X_fod)))
            view[1].clear()
            view[1].add_ball_and_stick('_X', color='red', radius=0.1)
        view.center()

    if filename.endswith('.cube'):
        # Atoms and unit cell
        atom, X, _, a, _ = read_cube(filename)
        atom, X, X_fod = split_atom_and_fod(atom, X)
        view.add_component(TextStructure(create_pdb(atom, X, a)))
        view[0].clear()
        view[0].add_ball_and_stick()
        view.add_unitcell()
        view.center()
        # Spin up
        view.add_component(filename)
        view[1].clear()
        view[1].add_surface(negateIsolevel=False,
                            isolevelType='value',
                            isolevel=isovalue,
                            color='lightgreen',
                            opacity=0.75,
                            side='front')
        # Spin down
        view.add_component(filename)
        view[2].clear()
        view[2].add_surface(negateIsolevel=True,
                            isolevelType='value',
                            isolevel=isovalue,
                            color='red',
                            opacity=0.75,
                            side='front')
        # FODs
        if len(X_fod) > 0:
            view.add_component(TextStructure(create_pdb(['X'] * len(X_fod), X_fod)))
            view[3].clear()
            view[3].add_ball_and_stick('_X', color='red', radius=0.1)
    return view


def save_view(view, filename, **kwargs):
    '''Save the current view as a png.

    Args:
        view :
            NGLWidget object.

        filename : str
            Output file path/name.
    '''
    if not filename.endswith('.png'):
        filename = f'{filename}.png'
    view.download_image(filename, trim=True, **kwargs)
    return


def split_atom_and_fod(atom, X):
    '''Split atom and FOD coordinates.

    Args:
        atom : list
            Atom symbols.

        X : array
            Atom positions.

    Returns:
        Shortened atom types and coordinates, with FOD coordinates as a tuple(list, array, array).
    '''
    X_fod = []
    # Iterate in reverted order, because we may delete elements
    for ia in range(len(X) - 1, -1, -1):
        if atom[ia] == 'X':
            X_fod.append(X[ia])
            X = np.delete(X, ia, axis=0)
            del atom[ia]
    X_fod = np.asarray(X_fod)
    return atom, X, X_fod
