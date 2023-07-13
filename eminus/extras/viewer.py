#!/usr/bin/env python3
"""Viewer functions for Jupyter notebooks."""
import io
import pathlib

import numpy as np

from ..data import COVALENT_RADII, CPK_COLORS
from ..io import create_pdb_str, read_cube, read_xyz
from ..logger import log
from ..tools import get_isovalue
from .fods import split_fods


def view(*args, **kwargs):
    """Unified display function."""
    if isinstance(args[0], str):
        return view_file(*args, **kwargs)
    return view_atoms(*args, **kwargs)


def view_atoms(object, extra=None, plot_n=False, percent=85, surfaces=20):
    """Display atoms and 3D-coordinates, e.g., FODs or grid points, or even densities.

    Reference: https://plotly.com/python/

    Args:
        object: Atoms or SCF object.

    Keyword Args:
        extra (ndarray | list ): Extra coordinates or FODs to display.
        plot_n (bool): Weather to plot the electronic density (only for executed scf objects).
        percent (float): Amount of density that should be contained.
        surfaces (int): Number of surfaces to display in density plots (reduce for performance).

    Returns:
        None.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.exception('Necessary dependencies not found. To use this module, '
                      'install them with "pip install eminus[viewer]".\n\n')
        raise
    try:
        atoms = object.atoms
    except AttributeError:
        atoms = object

    fig = go.Figure()
    # Add species one by one to be able to have them named and be selectable in the legend
    # Note: The size scaling is mostly arbitray and has no meaning
    for ia in sorted(set(atoms.atom)):
        mask = np.where(np.asarray(atoms.atom) == ia)[0]
        atom_data = go.Scatter3d(x=atoms.X[mask, 0], y=atoms.X[mask, 1], z=atoms.X[mask, 2],
                                 name=ia,
                                 mode='markers',
                                 marker={'size': 2 * np.pi * np.sqrt(COVALENT_RADII[ia]),
                                         'color': CPK_COLORS[ia],
                                         'line': {'color': 'black', 'width': 2}})
        fig.add_trace(atom_data)
    if extra is not None:
        # If a list has been provided with the length of two it has to be FODs
        if isinstance(extra, list):
            if len(extra[0]) != 0:
                extra_data = go.Scatter3d(x=extra[0][:, 0], y=extra[0][:, 1], z=extra[0][:, 2],
                                          name='up-FOD',
                                          mode='markers',
                                          marker={'size': np.pi, 'color': 'red'})
                fig.add_trace(extra_data)
            if len(extra) > 1 and len(extra[1]) != 0:
                extra_data = go.Scatter3d(x=extra[1][:, 0], y=extra[1][:, 1], z=extra[1][:, 2],
                                          name='down-FOD',
                                          mode='markers',
                                          marker={'size': np.pi, 'color': 'green'})
                fig.add_trace(extra_data)
        # Treat extra as normal coordinates otherwise
        else:
            extra_data = go.Scatter3d(x=extra[:, 0], y=extra[:, 1], z=extra[:, 2],
                                      name='Coordinates',
                                      mode='markers',
                                      marker={'size': 1, 'color': 'red'})
            fig.add_trace(extra_data)

    # A density can be plotted for an scf object
    if isinstance(plot_n, np.ndarray) or plot_n:
        # Plot a given array or use the density from an scf object
        if isinstance(plot_n, np.ndarray):
            density = plot_n
        else:
            density = object.n
        den_data = go.Volume(x=atoms.r[:, 0], y=atoms.r[:, 1], z=atoms.r[:, 2], value=density,
                             name='Density',
                             colorbar_title=f'Density ({percent}%)',
                             colorscale='Inferno',
                             isomin=get_isovalue(density, percent=percent),
                             isomax=np.max(density),
                             surface_count=surfaces,
                             opacity=0.1,
                             showlegend=True)
        fig.add_trace(den_data)
        # Move colorbar to the left
        fig.data[-1].colorbar.x = -0.15

    # Theming
    fig.update_layout(scene={'xaxis': {'range': [0, atoms.a[0]], 'title': 'x [a<sub>0</sub>]'},
                             'yaxis': {'range': [0, atoms.a[1]], 'title': 'y [a<sub>0</sub>]'},
                             'zaxis': {'range': [0, atoms.a[2]], 'title': 'z [a<sub>0</sub>]'},
                             'aspectmode': 'cube'},
                      legend={'itemsizing': 'constant', 'title': 'Selection'},
                      hoverlabel_bgcolor='black',
                      template='none')
    fig.show()


def view_file(filename, isovalue=0.01, gui=False, elec_symbols=None, **kwargs):
    """Display molecules and orbitals.

    Reference: Bioinformatics 34, 1241.

    Args:
        filename (str): Input file path/name. This can be either a cube or xyz file.

    Keyword Args:
        isovalue (float): Isovalue for sizing orbitals.
        gui (bool): Turn on the NGLView GUI.
        elec_symbols (list): Identifier for up and down FODs.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        NGLWidget: Viewable object.
    """
    try:
        from nglview import NGLWidget, TextStructure, write_html
    except ImportError:
        log.exception('Necessary dependencies not found. To use this module, '
                      'install them with "pip install eminus[viewer]".\n\n')
        raise

    if elec_symbols is None:
        elec_symbols = ('X', 'He')
    if isinstance(isovalue, str):
        isovalue = float(isovalue)

    view = NGLWidget(**kwargs)
    view._set_size('400px', '400px')
    # Set the gui to the view
    if gui:
        view.gui_style = 'ngl'

    # Handle XYZ files
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
    # Handle CUBE files
    elif filename.endswith('.cube'):
        # Atoms and cell
        atom, X, _, a, _, _ = read_cube(filename)
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
    # Handle other files (mainly PDB)
    else:
        # If no xyz or cube file is used try a more generic file viewer
        ext = pathlib.Path(filename).suffix.replace('.', '')
        # It seems that only pdb works with this method
        if ext != 'pdb':
            log.warning('Only XYZ, CUBE, and PDB files are support, but others might work.')
        with open(filename, 'r') as fh:
            view.add_component(fh, ext=ext)
        view[0].clear()
        view[0].add_ball_and_stick()
        view.center()

    # Check if the function has been executed in a notebook
    # If yes, just return the view
    if executed_in_notebook():
        return view
    # Otherwise the viewer would display nothing
    # But if plotly is installed on can write the view to a StringIO object and display it with the
    # smart open_html_in_browser function from plotly that automatically opens a new browser tab
    try:
        from plotly.io._base_renderers import open_html_in_browser
        # Use StringIO object instead of a temporary file
        with io.StringIO() as html:
            write_html(html, view)
            open_html_in_browser(html.getvalue())
    except ImportError:
        log.error('This function only works in notebooks or with Plotly installed.')
    return None


def executed_in_notebook():
    """Check if the code runs in a notebook.

    Reference: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    Returns:
        bool: Weather in a notebook or not.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        # Terminal running IPython or other type
        return False
    except NameError:
        return False
