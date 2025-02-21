# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import bool_, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]
type _Array1D = Sequence[float] | _ArrayReal
type _Array2D = Sequence[_Array1D] | _ArrayReal

def domain_cuboid(
    obj: Atoms | SCF,
    length: float | _Array1D,
    centers: _Array1D | _Array2D | None = ...,
) -> NDArray[bool_]: ...
def domain_isovalue(
    field: _ArrayReal | None,
    isovalue: float,
) -> NDArray[bool_]: ...
def domain_sphere(
    obj: Atoms | SCF,
    radius: float,
    centers: _Array1D | _Array2D | None = ...,
) -> NDArray[bool_]: ...
def truncate(
    field: _ArrayReal,
    mask: NDArray[bool_],
) -> _ArrayReal: ...
