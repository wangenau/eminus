# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Generate k-points and sample band paths."""

import numbers

import numpy as np
from scipy.linalg import inv, norm
from scipy.spatial import Voronoi

from .data import LATTICE_VECTORS, SPECIAL_POINTS
from .logger import log
from .utils import BaseObject


class KPoints(BaseObject):
    """KPoints object that holds k-points properties and build methods.

    Args:
        lattice: Lattice system.
        a: Cell size.
    """

    def __init__(self, lattice, a=None):
        """Initialize the KPoints object."""
        self.lattice = lattice  #: Lattice system.
        if a is None:
            a = LATTICE_VECTORS[self.lattice]
        if isinstance(a, numbers.Real):
            a = a * np.asarray(LATTICE_VECTORS[self.lattice])
        self.a = a  #: Cell size.
        self.kmesh = [1, 1, 1]  #: k-point mesh.
        self.wk = [1]  #: k-point weights.
        self.k = [[0, 0, 0]]  #: k-point coordinates.
        self.kshift = [0, 0, 0]  #: k-point shift-vector.
        self._gamma_centered = True  #: Generate a Gamma-point centered grid.
        self.is_built = True  #: Determines the KPoints object build status.

    # ### Class properties ###

    @property
    def kmesh(self):
        """Monkhorst-Pack k-point mesh."""
        return self._kmesh

    @kmesh.setter
    def kmesh(self, value):
        if value is not None:
            if isinstance(value, numbers.Integral):
                value = value * np.ones(3, dtype=int)
            self._kmesh = np.asarray(value)
            self.path = None
            self.is_built = False
        # If we set a band path to the object the k-mesh gets set to None
        else:
            self._kmesh = None

    @property
    def wk(self):
        """k-point weights."""
        return self._wk

    @wk.setter
    def wk(self, value):
        self._wk = np.asarray(value)
        self._Nk = len(self._wk)
        self.is_built = False

    @property
    def k(self):
        """k-point coordinates."""
        return self._k

    @k.setter
    def k(self, value):
        self._k = np.asarray(value)
        self.is_built = False

    @property
    def Nk(self):
        """Number of k-points."""
        return self._Nk

    @Nk.setter
    def Nk(self, value):
        self._Nk = int(value)
        self.is_built = False

    @property
    def kshift(self):
        """k-point shift-vector."""
        return self._kshift

    @kshift.setter
    def kshift(self, value):
        self._kshift = np.asarray(value)
        self.is_built = False

    @property
    def gamma_centered(self):
        """Generate a Gamma-point centered grid."""
        return self._gamma_centered

    @gamma_centered.setter
    def gamma_centered(self, value):
        if value != self._gamma_centered:
            self._gamma_centered = value
            self.is_built = False

    @property
    def path(self):
        """k-point band path."""
        return self._path

    @path.setter
    def path(self, value):
        if value is not None:
            self._path = value.upper()
            self.kmesh = None
            self.is_built = False
        # If we set a k-mesh to the object the band path gets set to None
        else:
            self._path = None

    # ### Read-only properties ###

    @property
    def k_scaled(self):
        """Scaled k-point coordinates."""
        # This will not be set when setting the k-point coordinates manually
        return self._k_scaled

    # ### Class methods ###

    def build(self):
        """Build all parameters of the KPoints object."""
        if self.lattice == "sc" and not (self.a == np.diag(np.diag(self.a))).all():
            log.warning("Lattice system and lattice vectors do not match.")
        if self.is_built:
            return self
        if self.kmesh is not None:
            if self.gamma_centered:
                self._k_scaled = gamma_centered(self.kmesh)
            else:
                self._k_scaled = monkhorst_pack(self.kmesh)
        else:
            self._k_scaled = bandpath(self)
        # Without removing redundancies the weight is the same for all k-points
        self.wk = np.ones(len(self._k_scaled)) / len(self._k_scaled)
        self.k = kpoint_convert(self._k_scaled, self.a) + self.kshift
        self.is_built = True
        return self

    kernel = build

    def _assert_gamma_only(self):
        """Make sure that the object only contains the Gamma point."""
        if not np.all(self.k == 0):
            msg = "The k-points object does not contain only the Gamma point."
            raise NotImplementedError(msg)

    def __repr__(self):
        """Print the parameters stored in the KPoints object."""
        return (
            f"Number of k-points: {self.Nk}\n"
            f"k-mesh: {self.kmesh}\n"
            f"Band path: {self.path}\n"
            f"Shift: {self.kshift}\n"
            f"Weights: {self.wk}"
        )


def kpoint_convert(k_points, lattice_vectors):
    """Convert scaled k-points to cartesian coordinates.

    Reference: https://gitlab.com/ase/ase/-/blob/master/ase/dft/kpoints.py

    Args:
        k_points: k-points.
        lattice_vectors: Lattice vectors.

    Returns:
        k-points in cartesian coordinates.
    """
    inv_cell = 2 * np.pi * inv(lattice_vectors).T
    return k_points @ inv_cell


def monkhorst_pack(nk):
    """Generate a Monkhorst-Pack mesh of k-points, i.e., equally spaced k-points.

    Reference: https://gitlab.com/ase/ase/-/blob/master/ase/dft/kpoints.py

    Args:
        nk: Number of k-points per axis.

    Returns:
        k-points.
    """
    # Same index matrix as in Atoms._get_index_matrices()
    M = np.indices(nk).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (M + 0.5) / nk - 0.5  # Normal Monkhorst-Pack grid


def gamma_centered(nk):
    """Generate a Gamma-point centered mesh of k-points.

    Reference: https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/gto/cell.py

    Args:
        nk: Number of k-points per axis.

    Returns:
        k-points.
    """
    # Same index matrix as in Atoms._get_index_matrices()
    M = np.indices(nk).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return M / nk  # Gamma-centered grid


def bandpath(kpts):
    """Generate sampled band paths.

    Args:
        kpts: KPoints object.

    Returns:
        Sampled k-points.
    """
    # Convert path to a list and get special points
    path_list = list(kpts.path)
    s_points = SPECIAL_POINTS[kpts.lattice]
    # Commas indicate jumps and are no special points
    N_special = len([p for p in path_list if p != ","])

    # Input handling
    N = kpts.Nk
    if N_special > N:
        log.warning("Sampling is smaller than the number of special points.")
        N = N_special
    for p in path_list:
        if p not in {*s_points, ","}:
            msg = f"{p} is not a special point for the {kpts.lattice} lattice."
            raise KeyError(msg)

    # Calculate distances between special points
    dists = []
    for i in range(len(path_list) - 1):
        if "," not in path_list[i : i + 2]:
            # Use subtract since s_points are lists
            dist = np.subtract(s_points[path_list[i + 1]], s_points[path_list[i]])
            dists.append(norm(kpoint_convert(dist, kpts.a)))
        else:
            # Set distance to zero when jumping between special points
            dists.append(0)

    # Calculate sample points between the special points
    scaled_dists = (N - N_special) * np.array(dists) / sum(dists)
    samplings = np.int64(np.round(scaled_dists))

    # If our sampling does not match the given N add the difference to the longest distance
    if N - N_special - np.sum(samplings) != 0:
        samplings[np.argmax(samplings)] += N - N_special - np.sum(samplings)

    # Generate k-point coordinates
    k_points = [s_points[path_list[0]]]  # Insert the first special point
    for i in range(len(path_list) - 1):
        # Only do something when not jumping between special points
        if "," not in path_list[i : i + 2]:
            s_start = s_points[path_list[i]]
            s_end = s_points[path_list[i + 1]]
            # Get the vector between special points
            k_dist = np.subtract(s_end, s_start)
            # Add scaled vectors to the special point to get the new k-points
            k_points += [
                s_start + k_dist * (n + 1) / (samplings[i] + 1) for n in range(samplings[i])
            ]
            # Append the special point we are ending at
            k_points.append(s_end)
        # If we jump, add the new special point to start from
        elif path_list[i] == ",":
            k_points.append(s_points[path_list[i + 1]])
    return np.asarray(k_points)


def kpoints2axis(kpts):
    """Generate the x-axis for band structures and the respective band path.

    Args:
        kpts: KPoints object.

    Returns:
        k-point axis, special point coordinates, and labels.
    """
    # Convert path to a list and get the special points
    path_list = list(kpts.path)
    s_points = SPECIAL_POINTS[kpts.lattice]

    # Calculate the distances between k-points
    k_dist = kpts.k_scaled[1:] - kpts.k_scaled[:-1]
    dists = norm(kpoint_convert(k_dist, kpts.a), axis=1)

    # Create the labels
    labels = []
    for i in range(len(path_list)):
        # If a jump happened before the current step the special point is already included
        if i > 1 and path_list[i - 1] == ",":
            continue
        # Append the special point if no jump happens
        if "," not in path_list[i : i + 2]:
            labels.append(path_list[i])
        # When jumping join the special points to one label
        elif path_list[i] == ",":
            labels.append("".join(path_list[i - 1 : i + 2]))

    # Get the indices of the special points
    special_indices = [0]  # The first special point is trivial
    for p in labels[1:]:
        # Only search the k-points starting from the previous special point
        shift = special_indices[-1]
        k = kpts.k_scaled[shift:]
        # We index p[0] since p could be a joined label of a jump
        # This expression simply finds the special point in the k_points matrix
        index = np.flatnonzero((k == s_points[p[0]]).all(axis=1))[0] + shift
        special_indices.append(index)
        # Set the distance between special points to zero if we have a jump
        if "," in p:
            dists[index] = 0

    # Insert a zero at the beginning and add up the lengths to create the k-axis
    k_axis = np.append([0], np.cumsum(dists))
    return k_axis, k_axis[special_indices], labels


def get_brillouin_zone(lattice_vectors):
    """Generate the Brillouin zone for given lattice vectors.

    The Brillouin zone can be constructed with a Voronoi decomposition of the reciprocal lattice.

    Reference: http://staff.ustc.edu.cn/~zqj/posts/howto-plot-brillouin-zone

    Args:
        lattice_vectors: Lattice vectors.

    Returns:
        Brillouin zone vertices.
    """
    inv_cell = kpoint_convert(np.eye(3), lattice_vectors)

    px, py, pz = np.tensordot(inv_cell, np.mgrid[-1:2, -1:2, -1:2], axes=(0, 0))
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]
    vor = Voronoi(points)

    bz_ridges = []
    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
    return bz_ridges
