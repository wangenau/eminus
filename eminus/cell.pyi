# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, TypeAlias

from numpy import floating, integer
from numpy.typing import NDArray

from .atoms import Atoms

_Int: TypeAlias = integer
_Float: TypeAlias = floating
_ArrayReal: TypeAlias = NDArray[_Float]
_Array1D: TypeAlias = Sequence[float] | _ArrayReal
_Array2D: TypeAlias = Sequence[_Array1D] | _ArrayReal

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
