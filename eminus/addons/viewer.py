#!/usr/bin/env python3
'''
Viewer functions for Jupyter notebooks.
'''
import numpy as np
from numpy.linalg import norm
try:
    from nglview import NGLWidget, TextStructure
    from vispy import scene
except ImportError:
    print('ERROR: Necessary addon dependecies not found. '
          'To use this module, install the package with addons, e.g., with '
          '"pip install eminus[addons]"')

from eminus.atoms_io import create_pdb, read_cube, read_xyz
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
        if len(X_fod) > 0:
            view.add_component(TextStructure(create_pdb(['X'] * len(X_fod), X_fod)))
            view[3].clear()
            view[3].add_ball_and_stick('_X', color='red', radius=0.1)
    return view


def view_grid(coords, extra=None):
    '''Display 3D-coordinates, e.g., grid points.

    Args:
        coords : array
            Grid points.

    Kwargs:
        extra : array
            Extra coordinates to display.

    Returns:
        Viewable object as a SceneCanvas.
    '''
    # Set up view
    canvas = scene.SceneCanvas(keys='interactive', show=True, size=(400, 400))
    view = canvas.central_widget.add_view()
    view.size = (400, 400)
    view.camera = 'arcball'
    view.camera.center = (np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2]))
    view.camera.distance = np.max(norm(coords, axis=1)) * 2

    # Add data
    grid = scene.visuals.Markers()
    grid.set_data(coords, face_color='lightgreen', edge_width=0, size=2)
    view.add(grid)
    if extra is not None:
        grid_extra = scene.visuals.Markers()
        grid_extra.set_data(extra, face_color='red', edge_width=0, size=8)
        view.add(grid_extra)

    canvas.app.run()
    return canvas
