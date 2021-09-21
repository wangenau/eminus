#!/usr/bin/env python3
'''
Viewer functions for Jupyter notebooks.
'''
try:
    from nglview import NGLWidget, TextStructure
except ImportError:
    print('ERROR: Necessary addon dependecies not found. '
          'To use this module, install the package with addons, e.g., "pip install .[addons]"')

from plainedft.atoms import create_pdb, read_cube, read_xyz


# Adapted from https://github.com/MolSSI/QCFractal/issues/374
def view(filename, isovalue=0.01):
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
    view = NGLWidget()
    view._set_size('400px', '400px')

    if filename.endswith('.xyz'):
        atom, X = read_xyz(filename)
        view.add_component(TextStructure(create_pdb(atom, X)))
        view.center()

    if filename.endswith('.cube'):
        # Atoms and unit cell
        atom, X, _, a, _ = read_cube(filename)
        view.add_component(TextStructure(create_pdb(atom, X, a)))
        view.add_unitcell()
        view.center()
        # Spin up
        view.add_component(filename)
        view[1].clear()
        view[1].add_surface(negateIsolevel=False,
                            isolevelType='value',
                            isolevel=isovalue,
                            color='lightgreen',
                            opacity=0.5,
                            side='front')
        # Spin down
        view.add_component(filename)
        view[2].clear()
        view[2].add_surface(negateIsolevel=True,
                            isolevelType='value',
                            isolevel=isovalue,
                            color='red',
                            opacity=0.5,
                            side='front')
    return view


def save_view(view, filename):
    '''Save the current view as a png.

    Args:
        view :
            NGLWidget object.

        filename : str
            Output file path/name.
    '''
    if not filename.endswith('.png'):
        filename = f'{filename}.png'
    view.download_image(filename, trim=True)
    return
