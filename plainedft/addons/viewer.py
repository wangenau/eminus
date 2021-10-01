#!/usr/bin/env python3
'''
Viewer functions for Jupyter notebooks.
'''
try:
    from nglview import NGLWidget, TextStructure
except ImportError:
    print('ERROR: Necessary addon dependecies not found. '
          'To use this module, install the package with addons, e.g., with "pip install .[addons]"')

from ..atoms import create_pdb, read_cube, read_xyz
from .fods import split_atom_and_fod


# Adapted from https://github.com/MolSSI/QCFractal/issues/374
def view_mol(filename, isovalue=0.01, **kwargs):
    '''Display molecules and orbitals.

    Args:
        filename : str
            Input file path/name. This can be either a cube or xyz file.

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
        if X_fod:
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
        # Negate isovalue here as a workaround for some display bugs
        view[1].add_surface(negateIsolevel=True,
                            isolevelType='value',
                            isolevel=-isovalue,
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
        if X_fod:
            view.add_component(TextStructure(create_pdb(['X'] * len(X_fod), X_fod)))
            view[3].clear()
            view[3].add_ball_and_stick('_X', color='red', radius=0.1)
    return view
