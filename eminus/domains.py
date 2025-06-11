# SPDX-FileCopyrightText: 2022 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Functions to restrict real-space fields to domains."""

import numbers

from . import backend as xp
from .logger import log
from .tools import center_of_mass


@xp.debug
def domain_cuboid(obj, length, centers=None):
    """Generate a mask for a cuboidal real-space domain.

    Args:
        obj: Atoms or SCF object.
        length: Side length or lengths of the cuboid.

    Keyword Args:
        centers: Center of the cuboid.

            Defaults to the geometric center of mass of the system. For multiple coordinates,
            multiple domains will be merged.

    Returns:
        Boolean mask.
    """
    atoms = obj._atoms

    if isinstance(length, numbers.Real):
        length = length * xp.ones(3)
    if centers is None:
        centers = center_of_mass(atoms.pos)
    centers = xp.asarray(centers)
    # Handle each dimension separately and add them together
    if centers.ndim == 1:
        mask1 = xp.abs(centers[0] - atoms.r[:, 0]) < length[0]
        mask2 = xp.abs(centers[1] - atoms.r[:, 1]) < length[1]
        mask3 = xp.abs(centers[2] - atoms.r[:, 2]) < length[2]
        mask = mask1 & mask2 & mask3
    else:
        mask = xp.zeros(atoms.Ns, dtype=bool)
        for center in centers:
            mask1 = xp.abs(center[0] - atoms.r[:, 0]) < length[0]
            mask2 = xp.abs(center[1] - atoms.r[:, 1]) < length[1]
            mask3 = xp.abs(center[2] - atoms.r[:, 2]) < length[2]
            mask = mask | (mask1 & mask2 & mask3)
    return mask


@xp.debug
def domain_isovalue(field, isovalue):
    """Generate a mask for an isovalue real-space domain.

    Args:
        field: Real-space field data.
        isovalue: Isovalue for the truncation.

    Returns:
        Boolean mask.
    """
    if field is None:
        log.warning('The provided field is "None".')
        return None
    return xp.abs(field) > isovalue


@xp.debug
def domain_sphere(obj, radius, centers=None):
    """Generate a mask for a spherical real-space domain.

    Args:
        obj: Atoms or SCF object.
        radius: Radius of the sphere.

    Keyword Args:
        centers: Center of the sphere.

            Defaults to the geometric center of mass of the system. For multiple coordinates,
            multiple domains will be merged.

    Returns:
        Boolean mask.
    """
    atoms = obj._atoms

    if centers is None:
        centers = center_of_mass(atoms.pos)
    centers = xp.asarray(centers)
    if centers.ndim == 1:
        mask = xp.linalg.norm(centers - atoms.r, axis=1) < radius
    else:
        mask = xp.zeros(atoms.Ns, dtype=bool)
        for center in centers:
            mask_tmp = xp.linalg.norm(center - atoms.r, axis=1) < radius
            mask = mask | mask_tmp
    return mask


@xp.debug
def truncate(field, mask):
    """Truncate field data for a given mask.

    This will not return a smaller array but set all truncated values to zero.

    Args:
        field: Real-space field data.
        mask: Boolean mask.

    Returns:
        Truncated field.
    """
    field_trunc = xp.copy(field)
    field_trunc[~mask] = 0
    return field_trunc
