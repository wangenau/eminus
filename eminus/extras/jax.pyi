# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from ..atoms import Atoms

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ArrayComplex = NDArray[_Complex]
_ArrayRealOrComplex = TypeVar("_ArrayRealOrComplex", _ArrayReal, _ArrayComplex)

@overload
def I(
    atoms: Atoms,
    W: list[_ArrayRealOrComplex],
) -> list[_ArrayRealOrComplex]: ...
@overload
def I(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
) -> _ArrayRealOrComplex: ...
@overload
def J(
    atoms: Atoms,
    W: list[_ArrayRealOrComplex],
    full: bool = ...,
) -> list[_ArrayRealOrComplex]: ...
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
    W: list[_ArrayRealOrComplex],
    full: bool = ...,
) -> list[_ArrayRealOrComplex]: ...
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
    W: list[_ArrayRealOrComplex],
) -> list[_ArrayRealOrComplex]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
) -> _ArrayRealOrComplex: ...
