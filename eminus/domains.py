#!/usr/bin/env python3
'''Functions to generate masks to restrict real-space fields to domains.'''
import numpy as np
from numpy.linalg import norm

from .tools import center_of_mass


def domain_cuboid(atoms, length, centers=None):
    '''Generate a mask for a cuboidal real-space domain.

    Args:
        atoms: Atoms object.
        length (int | float | list | ndarray): Side length or lengths of the cuboid.

    Keyword Args:
        centers (ndarray): Center of the cuboid.

            Defaults to the geometric center of mass of the system. For multiple coordinates,
            multiple domains will be merged.

    Returns:
        ndarray: Boolean mask.
    '''
    if isinstance(length, (int, float)):
        length = length * np.ones(3)
    if centers is None:
        centers = center_of_mass(atoms.X)
    if centers.ndim == 1:
        mask1 = (np.abs(centers[0] - atoms.r[:, 0]) < length[0])
        mask2 = (np.abs(centers[1] - atoms.r[:, 1]) < length[1])
        mask3 = (np.abs(centers[2] - atoms.r[:, 2]) < length[2])
        mask = mask1 & mask2 & mask3
    else:
        mask = np.zeros(len(atoms.r), dtype=bool)
        for center in centers:
            mask1 = (np.abs(center[0] - atoms.r[:, 0]) < length[0])
            mask2 = (np.abs(center[1] - atoms.r[:, 1]) < length[1])
            mask3 = (np.abs(center[2] - atoms.r[:, 2]) < length[2])
            mask = mask | (mask1 & mask2 & mask3)
    return mask


def domain_isovalue(field, isovalue):
    '''Generate a mask for an isovalue real-space domain.

    Args:
        field (ndarray): Real-space field data.
        isovalue (float): Isovalue for the truncation.

    Returns:
        ndarray: Boolean mask.
    '''
    return np.abs(field) > isovalue


def domain_sphere(atoms, radius, centers=None):
    '''Generate a mask for a spherical real-space domain.

    Args:
        atoms: Atoms object.
        radius (float): Radius of the sphere.

    Keyword Args:
        centers (ndarray): Center of the sphere.

            Defaults to the geometric center of mass of the system. For multiple coordinates,
            multiple domains will be merged.

    Returns:
        ndarray: Boolean mask.
    '''
    if centers is None:
        centers = center_of_mass(atoms.X)
    if centers.ndim == 1:
        mask = norm(centers - atoms.r, axis=1) < radius
    else:
        mask = np.zeros(len(atoms.r), dtype=bool)
        for center in centers:
            mask_tmp = norm(center - atoms.r, axis=1) < radius
            mask = mask | mask_tmp
    return mask


def truncate(field, mask):
    '''Truncate field data for a given mask.

    This will not return a smaller array but set all truncated values to zero.

    Args:
        field (ndarray): Real-space field data.
        mask (ndarray): Boolean mask.

    Returns:
        ndarray: Boolean mask.
    '''
    field_trunc = np.copy(field)
    field_trunc[~mask] = 0
    return field_trunc
