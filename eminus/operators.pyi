# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload, TypeVar

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

_AnyWorN = TypeVar('_AnyWorN', NDArray[float64], NDArray[complex128], list[NDArray[complex128]])
_ArrRealorComplex = TypeVar('_ArrRealorComplex', NDArray[float64], NDArray[complex128])

def O(
    atoms: Atoms,
    W: _AnyWorN,
) -> _AnyWorN: ...
def L(
    atoms: Atoms,
    W: _ArrRealorComplex,
    ik: int = ...,
) -> _ArrRealorComplex: ...
def Linv(
    atoms: Atoms,
    W: _ArrRealorComplex,
) -> _ArrRealorComplex: ...
@overload
def I(
    atoms: Atoms,
    W: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
@overload
def I(
    atoms: Atoms,
    W: _ArrRealorComplex,
    ik: int = ...,
) -> _ArrRealorComplex: ...
@overload
def J(
    atoms: Atoms,
    W: list[NDArray[complex128]],
    full: bool = ...,
) -> list[NDArray[complex128]]: ...
@overload
def J(
    atoms: Atoms,
    W: _ArrRealorComplex,
    ik: int = ...,
    full: bool = ...,
) -> _ArrRealorComplex: ...
@overload
def Idag(
    atoms: Atoms,
    W: list[NDArray[complex128]],
    full: bool = ...,
) -> list[NDArray[complex128]]: ...
@overload
def Idag(
    atoms: Atoms,
    W: _ArrRealorComplex,
    ik: int = ...,
    full: bool = ...,
) -> _ArrRealorComplex: ...
@overload
def Jdag(
    atoms: Atoms,
    W: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: _ArrRealorComplex,
    ik: int = ...,
) -> _ArrRealorComplex: ...
def K(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int,
) -> NDArray[complex128]: ...
def T(
    atoms: Atoms,
    W: _AnyWorN,
    dr: Array1D | Array2D,
) -> _AnyWorN: ...
