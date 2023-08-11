#!/usr/bin/env python3
"""Parse and sample band paths for band structures."""
import numpy as np
from scipy.linalg import norm, pinv

from .data import SPECIAL_POINTS
from .logger import log


def kpoint_convert(k_points, lattice_vectors):
    """Convert scaled k-points to cartesian coordinates.

    Reference: https://gitlab.com/ase/ase/-/blob/master/ase/dft/kpoints.py

    Args:
        k_points (ndarray): k-points.
        lattice_vectors (ndarray): Lattice vectors.

    Returns:
        ndarray: k-points in cartesian coordinates.
    """
    inv_cell = 2 * np.pi * pinv(lattice_vectors).T
    return k_points @ inv_cell


def bandpath(lattice, lattice_vectors, path, N):
    """Generate sampled band paths.

    Args:
        lattice (str): Lattice type.
        lattice_vectors (ndarray): Lattice vectors.
        path (str): Bandpath.
        N (int): Number of sampling points.

    Returns:
        ndarray: Sampled k-points.
    """
    # Convert path to a list and get special points
    path_list = list(path.upper())
    s_points = SPECIAL_POINTS[lattice]
    # Commas indicate jumps and are no special points
    N_special = len([p for p in path_list if p != ','])

    # Input handling
    if N_special > N:
        log.warning('Sampling is smaler than the number of special points.')
        N = N_special
    for p in path_list:
        if p not in list(s_points) + [',']:
            raise KeyError(f'{p} is not a special point for the {lattice} lattice.')

    # Calculate distances between special points
    dists = []
    for i in range(len(path_list) - 1):
        if ',' not in path_list[i:i + 2]:
            # Use subtract since s_points are lists
            dist = np.subtract(s_points[path_list[i + 1]], s_points[path_list[i]])
            dists.append(norm(kpoint_convert(dist, lattice_vectors)))
        else:
            # Set distance to zero when jumping between special points
            dists.append(0)

    # Calculate sample points between the special points
    scaled_dists = (N - N_special) * np.array(dists) / sum(dists)
    samplings = np.int_(np.round(scaled_dists))

    # Generate k-point coordinates
    k_points = [s_points[path_list[0]]]  # Insert the first special point
    for i in range(len(path_list) - 1):
        s_start = s_points[path_list[i]]
        s_end = s_points[path_list[i + 1]]
        # Only do something when not jumping between special points
        if ',' not in path_list[i:i + 2]:
            for n in range(1, samplings[i] + 1):
                # Get the vector between special points
                k_dist = np.subtract(s_end, s_start)
                # Add the scaled vector to the special point to get a new k-point
                k_points.append(s_start + k_dist * n / (samplings[i] + 1))
            # Append the special point we are ending at
            k_points.append(s_end)
        # If we jump, add the new special point to start from
        elif path_list[i] == ',':
            k_points.append(s_end)
    return np.asarray(k_points)


def kpoints2axis(lattice, lattice_vectors, path, k_points):
    """Generate the x-axis for band structure plots from k-points and the corresponding band path.

    Args:
        lattice (str): Lattice type.
        lattice_vectors (ndarray): Lattice vectors.
        path (str): Bandpath.
        k_points (ndarray): k-points.

    Returns:
        tuple[ndarray, ndarray, list]: k-point axis, special point coordinates, and labels.
    """
    # Convert path to a list and get the special points
    path_list = list(path.upper())
    s_points = SPECIAL_POINTS[lattice]

    # Calculate the distances between k-points
    k_dist = k_points[1:] - k_points[:-1]
    dists = norm(kpoint_convert(k_dist, lattice_vectors), axis=1)

    # Create the labels
    labels = []
    for i in range(len(path_list)):
        # If a jump happened before the current step the special point is already included
        if i > 1 and path_list[i - 1] == ',':
            continue
        # Append the special point if no jump happens
        if ',' not in path_list[i:i + 2]:
            labels.append(path_list[i])
        # When jumping join the special points to one label
        elif path_list[i] == ',':
            labels.append(''.join(path_list[i - 1:i + 2]))

    # Get the indices of the special points
    special_indices = [0]  # The first special point is trivial
    for p in labels[1:]:
        # Only search the k-points starting from the previous special point
        shift = special_indices[-1]
        k = k_points[shift:]
        # We index p[0] since p could be a joined label of a jump
        # This expression simply finds the special point in the k_points matrix
        index = np.flatnonzero((k == s_points[p[0]]).all(axis=1))[0] + shift
        special_indices.append(index)
        # Set the distance between special points to zero if we have a jump
        if ',' in p:
            dists[index] = 0

    # Insert a zero at the beginning and add up the lengths to create the k-axis
    k_axis = np.append([0], dists.cumsum())
    return k_axis, k_axis[special_indices], labels
