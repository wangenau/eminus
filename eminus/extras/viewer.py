#!/usr/bin/env python3
'''Viewer functions for Jupyter notebooks.'''
import numpy as np

from .fods import split_fods
from ..data import COVALENT_RADII, CPK_COLORS
from ..io import create_pdb_str, read_cube, read_xyz
from ..logger import log
from ..tools import get_isovalue


def view(*args, **kwargs):
    '''Unified display function.'''
    if isinstance(args[0], str):
        return view_file(*args, **kwargs)
    else:
        return view_atoms(*args, **kwargs)


def view_atoms(object, extra=None, plot_n=False, percent=85, surfaces=20):
    '''Display atoms and 3D-coordinates, e.g., FODs or grid points, or even densities.

    Args:
        object: Atoms or SCF object.

    Keyword Args:
        extra (ndarray | list ): Extra coordinates or FODs to display.
        plot_n (bool): Weather to plot the electronic density (only for executed scf objects).
        percent (float): Amount of density that should be contained.
        surfaces (int): Number of surfaces to display in density plots (reduce for performance).

    Returns:
        None.
    '''
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
    if plot_n:
        try:
            den_data = go.Volume(x=atoms.r[:, 0], y=atoms.r[:, 1], z=atoms.r[:, 2], value=object.n,
                                 name='Density',
                                 colorbar_title=f'Density ({percent}%)',
                                 colorscale='Inferno',
                                 isomin=get_isovalue(object.n, percent=percent),
                                 isomax=np.max(object.n),
                                 surface_count=surfaces,
                                 opacity=0.1,
                                 showlegend=True)
            fig.add_trace(den_data)
            # Move colorbar to the left
            fig.data[-1].colorbar.x = -0.15
        except (AttributeError, TypeError):
            log.warning('Density plots are only possible for executed SCF objects.')

    # Theming
    fig.update_layout(scene={'xaxis': {'range': [0, atoms.a[0]], 'title': 'x [a<sub>0</sub>]'},
                             'yaxis': {'range': [0, atoms.a[1]], 'title': 'y [a<sub>0</sub>]'},
                             'zaxis': {'range': [0, atoms.a[2]], 'title': 'z [a<sub>0</sub>]'},
                             'aspectmode': 'cube'},
                      legend={'itemsizing': 'constant', 'title': 'Selection'},
                      hoverlabel_bgcolor='black',
                      template='none')
    fig.show()
    return


def view_file(filename, isovalue=0.01, gui=False, elec_symbols=None, **kwargs):
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
    try:
        from nglview import NGLWidget, TextStructure
    except ImportError:
        log.exception('Necessary dependencies not found. To use this module, '
                      'install them with "pip install eminus[viewer]".\n\n')
        raise

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
