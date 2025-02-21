# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import floating, integer
from numpy.typing import NDArray

from .atoms import Atoms

type _Int = integer[Any]
type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]
type _Array1D = Sequence[float] | _ArrayReal
type _Array2D = Sequence[_Array1D] | _ArrayReal

STRUCTURES: dict[str, dict[str, str | list[list[float]]]]

def Cell(
    atom: str | Sequence[str],
    lattice: str | _Array2D,
    ecut: float,
    a: float | _Array1D | _Array2D | None,
    basis: _Array1D | _Array2D | None = ...,
    bands: int | None = ...,
    kmesh: int | Sequence[int] | NDArray[_Int] = ...,
    smearing: float = ...,
    magnetization: float | None = ...,
    verbose: int | str | None = ...,
    **kwargs: Any,
) -> Atoms: ...
