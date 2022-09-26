#!/usr/bin/env python3
'''Viewer functions for Jupyter notebooks.'''
import numpy as np
from scipy.linalg import norm
try:
    from nglview import NGLWidget, TextStructure
    from vispy import scene
except ImportError:
    print('ERROR: Necessary addon dependencies not found. To use this module,\n'
          '       install the package with addons, e.g., with "pip install eminus[addons]"')

from .fods import split_fods
from ..io import create_pdb_str, read_cube, read_xyz


# Adapted from https://github.com/MolSSI/QCFractal/issues/374
def view_mol(filename, isovalue=0.01, gui=False, elec_symbols=None, **kwargs):
    '''Display molecules and orbitals.

    Reference: Bioinformatics 34, 1241.

    Args:
        filename (str): Input file path/name. This can be either a cube or xyz file.

    Keyword Args:
        isovalue (float): Isovalue for sizing orbitals.
        gui (bool): Turn on the NGLView GUI.
        elec_symbols (list): Identifier for up and down FODs.

    Returns:
        NGLWidget: Viewable object.
    '''
    if elec_symbols is None:
        elec_symbols = ['X', 'He']

    if isinstance(isovalue, str):
        isovalue = float(isovalue)
    view = NGLWidget(**kwargs)
    view._set_size('400px', '400px')

    if filename.endswith('.xyz'):
        # Atoms
        atom, X = read_xyz(filename)
        atom, X, X_fod = split_fods(atom, X, elec_symbols)
        view.add_component(TextStructure(create_pdb_str(atom, X)))
        view[0].clear()
        view[0].add_ball_and_stick()
        # Spin up FODs
        if len(X_fod[0]) > 0:
            view.add_component(TextStructure(create_pdb_str([elec_symbols[0]] * len(X_fod[0]),
                               X_fod[0])))
            view[1].clear()
            view[1].add_ball_and_stick(f'_{elec_symbols[0]}', color='red', radius=0.1)
        # Spin down FODs
        if len(X_fod[1]) > 0:
            view.add_component(TextStructure(create_pdb_str([elec_symbols[1]] * len(X_fod[1]),
                               X_fod[1])))
            view[2].clear()
            view[2].add_ball_and_stick(f'_{elec_symbols[1]}', color='green', radius=0.1)
        view.center()

    if filename.endswith('.cube'):
        # Atoms and cell
        atom, X, _, a, _ = read_cube(filename)
        atom, X, X_fod = split_fods(atom, X, elec_symbols)
        view.add_component(TextStructure(create_pdb_str(atom, X, a)))
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
        # Spin up FODs
        if len(X_fod[0]) > 0:
            view.add_component(TextStructure(create_pdb_str([elec_symbols[0]] * len(X_fod[0]),
                               X_fod[0])))
            view[3].clear()
            view[3].add_ball_and_stick(f'_{elec_symbols[0]}', color='red', radius=0.1)
        # Spin down FODs
        if len(X_fod[1]) > 0:
            view.add_component(TextStructure(create_pdb_str([elec_symbols[1]] * len(X_fod[1]),
                               X_fod[1])))
            view[4].clear()
            view[4].add_ball_and_stick(f'_{elec_symbols[1]}', color='green', radius=0.1)
    return view.display(gui=gui)


def view_grid(coords, extra=None):
    '''Display 3D-coordinates, e.g., grid points.

    Args:
        coords (ndarray): Grid points.

    Keyword Args:
        extra (ndarray): Extra coordinates to display.

    Returns:
        SceneCanvas: Viewable object.
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
