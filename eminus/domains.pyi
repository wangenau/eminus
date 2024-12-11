# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from numpy import bool_, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF
from .typing import Array1D, Array2D

def domain_cuboid(
    obj: Atoms | SCF,
    length: float | Array1D,
    centers: Array1D | Array2D | None = ...,
) -> NDArray[bool_]: ...
def domain_isovalue(
    field: NDArray[floating] | None,
    isovalue: float,
) -> NDArray[bool_]: ...
def domain_sphere(
    obj: Atoms | SCF,
    radius: float,
    centers: Array1D | Array2D | None = ...,
) -> NDArray[bool_]: ...
def truncate(
    field: NDArray[floating],
    mask: NDArray[bool_],
) -> NDArray[floating]: ...
