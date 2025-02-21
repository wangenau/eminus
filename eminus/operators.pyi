# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]
type _Array1D = Sequence[float] | _ArrayReal
type _Array2D = Sequence[_Array1D] | _ArrayReal
_AnyWOrN = TypeVar("_AnyWOrN", _ArrayReal, _ArrayComplex, list[_ArrayComplex])
_ArrayRealOrComplex = TypeVar("_ArrayRealOrComplex", _ArrayReal, _ArrayComplex)

def O(
    atoms: Atoms,
    W: _AnyWOrN,
) -> _AnyWOrN: ...
def L(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
) -> _ArrayRealOrComplex: ...
def Linv(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
) -> _ArrayRealOrComplex: ...
@overload
def I(
    atoms: Atoms,
    W: list[_ArrayComplex],
) -> list[_ArrayComplex]: ...
@overload
def I(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
) -> _ArrayRealOrComplex: ...
@overload
def J(
    atoms: Atoms,
    W: list[_ArrayComplex],
    full: bool = ...,
) -> list[_ArrayComplex]: ...
@overload
def J(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
    full: bool = ...,
) -> _ArrayRealOrComplex: ...
@overload
def Idag(
    atoms: Atoms,
    W: list[_ArrayComplex],
    full: bool = ...,
) -> list[_ArrayComplex]: ...
@overload
def Idag(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
    full: bool = ...,
) -> _ArrayRealOrComplex: ...
@overload
def Jdag(
    atoms: Atoms,
    W: list[_ArrayComplex],
) -> list[_ArrayComplex]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
) -> _ArrayRealOrComplex: ...
def K(
    atoms: Atoms,
    W: _ArrayComplex,
    ik: int,
) -> _ArrayComplex: ...
def T(
    atoms: Atoms,
    W: _AnyWOrN,
    dr: _Array1D | _Array2D,
) -> _AnyWOrN: ...
