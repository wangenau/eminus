# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from numpy import bool_, floating
from numpy.typing import NDArray

from ._typing import _Array1D, _Array2D
from .atoms import Atoms
from .scf import SCF

def domain_cuboid(
    obj: Atoms | SCF,
    length: float | _Array1D,
    centers: _Array1D | _Array2D | None = ...,
) -> NDArray[bool_]: ...
def domain_isovalue(
    field: NDArray[floating] | None,
    isovalue: float,
) -> NDArray[bool_]: ...
def domain_sphere(
    obj: Atoms | SCF,
    radius: float,
    centers: _Array1D | _Array2D | None = ...,
) -> NDArray[bool_]: ...
def truncate(
    field: NDArray[floating],
    mask: NDArray[bool_],
) -> NDArray[floating]: ...
