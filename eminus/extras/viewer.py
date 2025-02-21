# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Viewer functions for Jupyter notebooks.

All necessary dependencies to use this extra can be installed with::

    pip install eminus[viewer]

Note that if one wants to use Jupyter one has to install it separately.
"""

import io
import pathlib
import uuid

import numpy as np
from scipy.interpolate import griddata

from ..atoms import Atoms
from ..data import COVALENT_RADII, CPK_COLORS, SPECIAL_POINTS
from ..dft import get_epsilon, get_epsilon_unocc
from ..io import create_pdb_str, read_cube, read_traj, read_xyz
from ..kpoints import get_brillouin_zone, kpoint_convert, KPoints, kpoints2axis
from ..logger import log
from ..tools import get_dos, get_Efermi, get_isovalue
from ..units import ha2ev
from ..utils import get_lattice
from .fods import split_fods


def view(*args, **kwargs):
    """Unified display function.

    Args:
        *args: Pass-through arguments.

    Keyword Args:
        **kwargs: Pass-through keyword arguments.

    Returns:
        Viewable object.
    """
    if isinstance(args[0], (str, list, tuple)):
        return view_file(*args, **kwargs)
    if isinstance(args[0], KPoints):
        return view_kpts(*args, **kwargs)
    return view_atoms(*args, **kwargs)


def _uniform_density_data(n, r, s):
    """Interpolate a density on an orthogonal grid.

    Args:
        n: Real-space electronic density.
        r: Real-space sampling points.
        s: Real-space sampling of the cell.

    Returns:
        Interpolated density and new grid points.
    """
    # Do not use a super fine grid to display the interpolated data
    x = np.linspace(np.min(r[:, 0]), np.max(r[:, 0]), np.min([s[0], 25]))
    y = np.linspace(np.min(r[:, 1]), np.max(r[:, 1]), np.min([s[1], 25]))
    z = np.linspace(np.min(r[:, 2]), np.max(r[:, 2]), np.min([s[2], 25]))
    r_new = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    n_new = griddata(r, n, r_new)
    return np.nan_to_num(n_new), r_new


def view_atoms(  # noqa: PLR0915
    obj, fods=None, plot_n=False, percent=85, isovalue=None, surfaces=10, size=(600, 600)
):
    """Display atoms with optional FODs or grid points, or even densities.

    Reference: https://plotly.com/python

    Args:
        obj: Atoms or SCF object.

    Keyword Args:
        fods: FODs or extra coordinates or to display.
        plot_n: Whether to plot the electronic density (only for executed scf objects).
        percent: Amount of density that should be contained.
        isovalue: Isovalue for sizing densities.
        surfaces: Number of surfaces to display in density plots (reduce for better performance).
        size: Widget size.

    Returns:
        Viewable object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise
    atoms = obj._atoms

    fig = go.Figure()
    # Calculate vertices of the Brillouin zone and plot them
    lattice = get_lattice(atoms.a)
    for xx in lattice:
        bz_data = go.Scatter3d(
            x=xx[:, 0],
            y=xx[:, 1],
            z=xx[:, 2],
            name="Unit cell",
            showlegend=False,
            marker={"size": 0.1, "color": "black"},
        )
        fig.add_trace(bz_data)
    # Add species one by one to be able to have them named and be selectable in the legend
    # Note: The size scaling is mostly arbitrary and has no meaning
    for ia in sorted(set(atoms.atom)):
        mask = np.where(np.asarray(atoms.atom) == ia)[0]
        atom_data = go.Scatter3d(
            x=atoms.pos[mask, 0],
            y=atoms.pos[mask, 1],
            z=atoms.pos[mask, 2],
            name=ia,
            mode="markers",
            marker={
                "size": 2 * np.pi * np.sqrt(COVALENT_RADII[ia]),
                "color": CPK_COLORS[ia],
                "line": {"color": "black", "width": 2},
            },
        )
        fig.add_trace(atom_data)
    if fods is not None:
        # If a list has been provided with a length of two it has to be FODs
        if isinstance(fods, list):
            if len(fods[0]) != 0:
                fods_data = go.Scatter3d(
                    x=fods[0][:, 0],
                    y=fods[0][:, 1],
                    z=fods[0][:, 2],
                    name="up-FOD",
                    mode="markers",
                    marker={"size": np.pi, "color": "red"},
                )
                fig.add_trace(fods_data)
            if len(fods) > 1 and len(fods[1]) != 0:
                fods_data = go.Scatter3d(
                    x=fods[1][:, 0],
                    y=fods[1][:, 1],
                    z=fods[1][:, 2],
                    name="down-FOD",
                    mode="markers",
                    marker={"size": np.pi, "color": "green"},
                )
                fig.add_trace(fods_data)
        # Treat fods as normal coordinates otherwise
        else:
            fods_data = go.Scatter3d(
                x=fods[:, 0],
                y=fods[:, 1],
                z=fods[:, 2],
                name="Coordinates",
                mode="markers",
                marker={"size": 1, "color": "red"},
            )
            fig.add_trace(fods_data)

    # A density can be plotted for an SCF object
    if isinstance(plot_n, np.ndarray) or plot_n:
        # Use the density from an SCF object...
        density = obj.n
        name = "Density"
        # ...or use the quantity if an array is given
        if isinstance(plot_n, np.ndarray):
            density = plot_n
            name = "Quantity"

        # If the unit cell is not diagonal we have to interpolate the density data
        # Plotly will not display volumes that are represented on non-orthogonal grids, which is
        # normally the case for non-diagonal unit cells
        if (np.diag(np.diag(atoms.a)) != atoms.a).any():
            density, r = _uniform_density_data(density, atoms.r, atoms.s)
            # We will get data points outside of the unit cell that can contain NaNs
            # Do not use the get_isovalue function here but simply use the minimum value
            percent = 100
            isomin = np.min(density)
            log.warning(
                "The density data for non-orthogonal grids will be interpolated. The "
                '"percent" keyword has no effect and all of the density data will be displayed.'
            )
        else:
            r = atoms.r
            isomin = get_isovalue(density, percent=percent)

        if isovalue is not None:
            isomin = isovalue
            log.warning('The "density" keyword will be ignored when setting "isovalue".')

        den_data = go.Volume(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            value=density,
            name=name,
            colorbar_title=name,
            colorscale="Inferno",
            isomin=isomin,
            isomax=np.max(density),
            surface_count=surfaces,
            opacity=0.1,
            showlegend=True,
        )
        fig.add_trace(den_data)
        # Move colorbar to the left
        fig.data[-1].colorbar.x = -0.15

    # Theming
    scene = {
        "xaxis": {"title": "x [a<sub>0</sub>]"},
        "yaxis": {"title": "y [a<sub>0</sub>]"},
        "zaxis": {"title": "z [a<sub>0</sub>]"},
        "aspectmode": "data",
    }
    # If the unit cell is diagonal and we scale the plot, otherwise let plotly decide
    if (np.diag(np.diag(atoms.a)) == atoms.a).all():
        scene["xaxis"]["range"] = [0, atoms.a[0, 0]]
        scene["yaxis"]["range"] = [0, atoms.a[1, 1]]
        scene["zaxis"]["range"] = [0, atoms.a[2, 2]]
    fig.update_layout(
        scene=scene,
        width=size[0],
        height=size[1],
        legend={"itemsizing": "constant", "title": "Selection"},
        hoverlabel_bgcolor="black",
        template="none",
    )
    if not executed_in_notebook():
        fig.show()
    return fig


def view_contour(
    obj, field, axis=2, value=0.5, lines=10, limits=(-1, 1), zoom=1, linewidth=1, size=(600, 600)
):
    """Display contour lines of field data like electronic densities.

    Reference: https://plotly.com/python

    Args:
        obj: Atoms or SCF object.
        field: Real-space field data.

    Keyword Args:
        axis: Axis to slice through.
        value:: Slice value scaled by the axis size.
        lines: Number of contour lines.
        limits: Minimal and maximal truncation values for the field data.
        zoom: Initial zoom.
        linewidth: Contour line width.
        size: Widget size.

    Returns:
        Viewable object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise
    atoms = obj._atoms

    # Get the axes indices
    axes = [(axis + 1) % 3, (axis + 2) % 3, axis]
    # Create an index mask to obtain a clean slice through the cell
    # Since the expression is a bit involved it can be understood with the following pseudo-code:
    # mask = |axis_values - slice_value| < axis_value_dist
    mask = np.abs(atoms.r[:, axes[2]] - value * atoms.a[axes[2], axes[2]]) < (
        atoms.a[axes[2], axes[2]] / atoms.s[axes[2]]
    )

    if field is None:
        log.warning('The provided field is "None".')
        return None
    # Create a copy of the field data to not overwrite the input
    field = np.copy(field)
    # Remove large and small values (similar to VESTA)
    field[field < limits[0]] = limits[0]
    field[field > limits[1]] = limits[1]

    # Create the contour lines
    fig = go.Figure()
    contours = go.Contour(
        x=atoms.r[:, axes[0]][mask],
        y=atoms.r[:, axes[1]][mask],
        z=field[mask],
        contours_coloring="none",
        ncontours=lines,
        line_width=linewidth,
    )
    fig.add_trace(contours)

    # Theming
    fig.update_layout(
        width=size[0],
        height=size[1],
        margin={"b": 0, "l": 0, "r": 0, "t": 0},
        xaxis={
            "range": [
                (1 - 1 / zoom) * atoms.a[axes[0], axes[0]],
                1 / zoom * atoms.a[axes[0], axes[0]],
            ],
            "visible": False,
        },
        yaxis={
            "range": [
                (1 - 1 / zoom) * atoms.a[axes[1], axes[1]],
                1 / zoom * atoms.a[axes[1], axes[1]],
            ],
            "visible": False,
        },
        template="none",
    )
    if not executed_in_notebook():
        fig.show()
    return fig


def view_file(
    filename, isovalue=0.01, gui=False, elec_symbols=("X", "He"), size=(400, 400), **kwargs
):
    """Display molecules and orbitals.

    Reference: Bioinformatics 34, 1241.

    Args:
        filename: Input filename(s). This can be either CUBE or XYZ files.

    Keyword Args:
        isovalue: Isovalue for sizing orbitals.
        gui: Turn on the NGLView GUI.
        elec_symbols: Identifier for up and down FODs.
        size: Widget size.

    Keyword Args:
        **kwargs: NGLWidget keyword arguments.

    Returns:
        Viewable object.
    """
    try:
        from nglview import NGLWidget, write_html
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise

    # If multiple files are given, try to open them with an interact drop-down menu
    # If all files are XYZ files they will be displayed as a trajectory instead
    if isinstance(filename, (list, tuple)) and not np.all([f.endswith(".xyz") for f in filename]):
        if executed_in_notebook():
            from ipywidgets import interact

            interact(
                lambda filename: view_file(filename, isovalue, gui, elec_symbols, size, **kwargs),
                filename=filename,
            )
            return None
        # If we are not in a notebook open the files one by one
        for f in filename:
            view_file(f, isovalue, gui, elec_symbols, size, **kwargs)
            return None

    view = NGLWidget(**kwargs)
    view._set_size(f"{size[0]}px", f"{size[1]}px")
    # Set the gui to the view
    if gui:
        view.gui_style = "ngl"

    # Handle TRAJ files (or multiple XYZ files)
    if isinstance(filename, (list, tuple)) or filename.endswith((".trj", ".traj")):
        view = _traj_view(view, filename)
    # Handle XYZ files
    elif filename.endswith(".xyz"):
        view = _xyz_view(view, filename, elec_symbols)
    # Handle CUBE files
    elif filename.endswith((".cub", ".cube")):
        view = _cube_view(view, filename, isovalue, elec_symbols)
    # Handle other files (mainly for PDBs)
    else:
        view = _generic_view(view, filename)

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
        log.error("This function only works in notebooks or with Plotly installed.")
    return None


def executed_in_notebook():
    """Check if the code runs in a notebook.

    Reference: https://stackoverflow.com/a/22424821

    Returns:
        Whether in a notebook or not.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except (AttributeError, ImportError):
        return False
    return True


def _generic_view(view, filename):
    """Modify the view for a generic (probably PDB) file.

    Args:
        view: Viewable object.
        filename: Input filename.

    Returns:
        Viewable object.
    """
    # If no XYZ or CUBE file is used try a more generic file viewer
    ext = pathlib.Path(filename).suffix.replace(".", "")
    # It seems that only PDB works with this method
    if ext != "pdb":
        log.warning("Only XYZ, CUBE, and PDB files are support, but others might work.")
    with open(filename, encoding="utf-8") as fh:
        view.add_component(fh, ext=ext)
    view[0].clear()
    view[0].add_ball_and_stick()
    view.center()
    return view


def _cube_view(view, filename, isovalue, elec_symbols):
    """Modify the view for a given CUBE file.

    Args:
        view: Viewable object.
        filename: Input filename(s).
        isovalue: Isovalue for sizing orbitals.
        elec_symbols: Identifier for up and down FODs.

    Returns:
        Viewable object.
    """
    from nglview import TextStructure

    # Atoms and cell
    atom, pos, _, a, _, _ = read_cube(filename)
    atom, pos, pos_fod = split_fods(atom, pos, elec_symbols)
    view.add_component(TextStructure(create_pdb_str(atom, pos, a)))
    view[0].clear()
    view[0].add_ball_and_stick()
    view.add_unitcell()
    view.center()
    # Spin up
    view.add_component(filename)
    view[1].clear()
    view[1].add_surface(
        negateIsolevel=False,
        isolevelType="value",
        isolevel=isovalue,
        color="lightgreen",
        opacity=0.75,
        side="front",
        depthWrite=False,
    )
    # Spin down
    view.add_component(filename)
    view[2].clear()
    view[2].add_surface(
        negateIsolevel=True,
        isolevelType="value",
        isolevel=isovalue,
        color="red",
        opacity=0.75,
        side="front",
        depthWrite=False,
    )
    # Spin up FODs
    if len(pos_fod[0]) > 0:
        view.add_component(
            TextStructure(create_pdb_str([elec_symbols[0]] * len(pos_fod[0]), pos_fod[0]))
        )
        view[3].clear()
        view[3].add_ball_and_stick(f"_{elec_symbols[0]}", color="red", radius=0.1)
    # Spin down FODs
    if len(pos_fod[1]) > 0:
        view.add_component(
            TextStructure(create_pdb_str([elec_symbols[1]] * len(pos_fod[1]), pos_fod[1]))
        )
        view[4].clear()
        view[4].add_ball_and_stick(f"_{elec_symbols[1]}", color="green", radius=0.1)
    return view


def _xyz_view(view, filename, elec_symbols):
    """Modify the view for a given XYZ file.

    Args:
        view: Viewable object.
        filename: Input filename(s).
        elec_symbols: Identifier for up and down FODs.

    Returns:
        Viewable object.
    """
    from nglview import TextStructure

    # Atoms
    atom, pos = read_xyz(filename)
    atom, pos, pos_fod = split_fods(atom, pos, elec_symbols)
    view.add_component(TextStructure(create_pdb_str(atom, pos)))
    view[0].clear()
    view[0].add_ball_and_stick()
    # Spin up FODs
    if len(pos_fod[0]) > 0:
        view.add_component(
            TextStructure(create_pdb_str([elec_symbols[0]] * len(pos_fod[0]), pos_fod[0]))
        )
        view[1].clear()
        view[1].add_ball_and_stick(f"_{elec_symbols[0]}", color="red", radius=0.1)
    # Spin down FODs
    if len(pos_fod[1]) > 0:
        view.add_component(
            TextStructure(create_pdb_str([elec_symbols[1]] * len(pos_fod[1]), pos_fod[1]))
        )
        view[2].clear()
        view[2].add_ball_and_stick(f"_{elec_symbols[1]}", color="green", radius=0.1)
    view.center()
    return view


def _traj_view(view, filename):
    """Modify the view for a given TRAJ file.

    Args:
        view: Viewable object.
        filename: Input filename(s).

    Returns:
        Viewable object.
    """
    try:
        from nglview.base_adaptor import Structure, Trajectory
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise

    class EminusTrajectory(Trajectory, Structure):
        """EminusTrajectory object to handle trajectory files.

        The interface replicates the Trajectory classes in
        https://nglviewer.org/nglview/latest/_modules/nglview/adaptor.html

        Args:
            filenames: XYZ input file paths/names.
        """

        __slots__ = ("atoms", "ext", "id", "params")

        def __init__(self, filenames):
            """Initialize the EminusTrajectory object."""
            self.atoms = []
            if isinstance(filenames, str) and filenames.endswith((".trj", ".traj")):
                trajectory = read_traj(filenames)
                for frame in trajectory:
                    self.atoms.append(Atoms(*frame))
            else:
                if isinstance(filenames, str) and filenames.endswith(".xyz"):
                    filenames = [filenames]
                for f in filenames:
                    self.atoms.append(Atoms(*read_xyz(f)))
            self.ext = "pdb"
            self.params = {}
            self.id = str(uuid.uuid4())

        def get_coordinates(self, index):
            """Get the atom coordinates for a given frame.

            Args:
                index: Frame number.

            Returns:
                Atom positions in Bohr.
            """
            return self.atoms[index].pos

        @property
        def n_frames(self):
            """Number of frames."""
            return len(self.atoms)

        def get_structure_string(self, index=0):
            """Get the structure string per frame in the PDB format.

            Keyword Args:
                index: Frame number.

            Returns:
                str: Structure string.
            """
            return create_pdb_str(self.atoms[index].atom, self.atoms[index].pos)

    trajectory = EminusTrajectory(filename)
    view.add_trajectory(trajectory)
    view.center()
    return view


def plot_bandstructure(scf, spin=0, size=(800, 600)):
    """Plot band structures.

    Reference: https://plotly.com/python

    Args:
        scf: SCF object.

    Keyword Args:
        spin: Spin channel.
        size: Widget size.

    Returns:
        Viewable object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise
    k_axis, special, label = kpoints2axis(scf.kpts)
    # Replace "G" with the Greek "Gamma" encoded in unicode
    label = ["\u0393" if l == "G" else l for l in label]
    e_occ = ha2ev(get_epsilon(scf, scf.W, **scf._precomputed))
    Efermi = ha2ev(get_Efermi(scf))

    # If both spin channels are plotted used blue and red colors, use automatic colors otherwise
    spin = np.atleast_1d(spin)
    if len(spin) > 1:
        colors = ("blue", "red")
    else:
        colors = (None, None)

    fig = go.Figure()
    for s in spin:
        # Plot occupied bands
        for i in range(scf.atoms.occ.Nstate):
            fig.add_trace(
                go.Scatter(
                    x=k_axis,
                    y=e_occ[:, s, i] - Efermi,
                    mode="lines+markers",
                    name=f"Band {i + 1}",
                    marker_color=colors[s],
                )
            )

        # Calculate and plot unoccupied bands if available
        if scf.Z is not None:
            e_unocc = ha2ev(get_epsilon_unocc(scf, scf.W, scf.Z, **scf._precomputed))
            for i in range(scf.atoms.occ.Nempty):
                fig.add_trace(
                    go.Scatter(
                        x=k_axis,
                        y=e_unocc[:, s, i] - Efermi,
                        mode="lines+markers",
                        line={"dash": "dash"},
                        name=f"Unocc. band {i + 1}",
                        marker_color=colors[s],
                    )
                )

    fig.update_layout(
        width=size[0],
        height=size[1],
        showlegend=False,
        font={"size": 20},
        xaxis={
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": special,
            "ticktext": label,
            "gridcolor": "grey",
            "gridwidth": 2,
        },
        yaxis={
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "ticks": "outside",
            "showgrid": False,
        },
        xaxis_range=(0, k_axis[-1]),
        xaxis_title="k-path",
        yaxis_title="E - E<sub>F</sub> [eV]",
        hoverlabel_bgcolor="black",
        template="none",
    )
    if not executed_in_notebook():
        fig.show()
    return fig


def plot_dos(scf, spin=0, size=(800, 600), **kwargs):
    """Plot density of states.

    Reference: https://plotly.com/python

    Args:
        scf: SCF object.

    Keyword Args:
        spin: Spin channel.
        size: Widget size.
        kwargs: Pass-through keyword arguments.

    Returns:
        Viewable object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise
    e_occ = ha2ev(get_epsilon(scf, scf.W, **scf._precomputed))
    Efermi = ha2ev(get_Efermi(scf))

    spin = np.atleast_1d(spin)
    colors = ("blue", "red")

    fig = go.Figure()
    for s in spin:
        e, dos_e = get_dos(e_occ, scf.kpts.wk, spin=s, **kwargs)
        fig.add_trace(
            go.Scatter(
                x=e - Efermi,
                y=dos_e,
                mode="lines+markers",
                name=f"Spin {s + 1}",
                marker_color=colors[s],
            )
        )

    fig.update_layout(
        width=size[0],
        height=size[1],
        font={"size": 20},
        xaxis={
            "zeroline": False,
            "showline": True,
            "mirror": True,
            "ticks": "outside",
        },
        yaxis={
            "showline": True,
            "mirror": True,
            "ticks": "outside",
        },
        xaxis_title="E - E<sub>F</sub> [eV]",
        yaxis_title="DOS",
        hoverlabel_bgcolor="black",
        template="none",
    )
    if not executed_in_notebook():
        fig.show()
    return fig


def view_kpts(kpts, path=True, special=True, connect=False, size=(600, 600)):
    """Display KPoints objects including Brillouin zones, special- and k-points.

    Reference: https://plotly.com/python

    Args:
        kpts: KPoints object.

    Keyword Args:
        path: Determines if k-points should be plotted.
        special: Determines if special points should be plotted.
        connect: Determines if k-points should connected.
        size: Widget size.

    Returns:
        Viewable object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.exception(
            "Necessary dependencies not found. To use this module, "
            'install them with "pip install eminus[viewer]".\n\n'
        )
        raise

    fig = go.Figure()
    # Calculate vertices of the Brillouin zone and plot them
    bz = get_brillouin_zone(kpts.a)
    for xx in bz:
        bz_data = go.Scatter3d(
            x=xx[:, 0],
            y=xx[:, 1],
            z=xx[:, 2],
            name="Brillouin zone",
            showlegend=False,
            marker={"size": 0.1, "color": "black"},
        )
        fig.add_trace(bz_data)

    # Plot special points if needed
    if special:
        for label, k_scaled in SPECIAL_POINTS[kpts.lattice].items():
            if label == "G":
                label = "\u0393"  # noqa: PLW2901
            v = kpoint_convert(k_scaled, kpts.a)
            special_data = go.Scatter3d(
                x=[v[0]],
                y=[v[1]],
                z=[v[2]],
                name=label,
                mode="markers",
                marker={"size": 4, "opacity": 0.9},
            )
            fig.add_trace(special_data)

    # Plot optional k-points
    if connect:
        mode = "lines+markers"
    else:
        mode = "markers"
    if path:
        path_data = go.Scatter3d(
            x=kpts.k[:, 0],
            y=kpts.k[:, 1],
            z=kpts.k[:, 2],
            name="k-points",
            mode=mode,
            marker={"size": 2, "color": "#1a962b", "opacity": 0.75},
        )
        fig.add_trace(path_data)

    # Theming
    scene = {
        "xaxis": {"title": "b<sub>1</sub>"},
        "yaxis": {"title": "b<sub>2</sub>"},
        "zaxis": {"title": "b<sub>3</sub>"},
        "aspectmode": "data",
    }
    fig.update_layout(
        scene=scene,
        width=size[0],
        height=size[1],
        legend={"itemsizing": "constant", "title": "Selection"},
        hoverlabel_bgcolor="black",
        template="none",
    )
    if not executed_in_notebook():
        fig.show()
    return fig
