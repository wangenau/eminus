# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Literal, overload, TypeAlias, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .utils import handle_k, handle_spin

_Float: TypeAlias = floating
_Complex: TypeAlias = complexfloating
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayComplex: TypeAlias = NDArray[_Complex]
_Array1D: TypeAlias = Sequence[float] | _ArrayReal
_Array2D: TypeAlias = Sequence[_Array1D] | _ArrayReal
_AnyWOrN = TypeVar("_AnyWOrN", _ArrayReal, _ArrayComplex, list[_ArrayComplex])
_ArrayRealOrComplex = TypeVar("_ArrayRealOrComplex", _ArrayReal, _ArrayComplex)

@handle_k
def O(
    atoms: Atoms,
    W: _AnyWOrN,
) -> _AnyWOrN: ...
@handle_spin
def L(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
) -> _ArrayRealOrComplex: ...
@handle_spin
def Linv(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
) -> _ArrayRealOrComplex: ...
@overload
def I(
    atoms: Atoms,
    W: list[_ArrayComplex],
    norm: Literal["backward", "forward"] = ...,
) -> list[_ArrayComplex]: ...
@overload
def I(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
    norm: Literal["backward", "forward"] = ...,
) -> _ArrayRealOrComplex: ...
@overload
def J(
    atoms: Atoms,
    W: list[_ArrayComplex],
    full: bool = ...,
    norm: Literal["backward", "forward"] = ...,
) -> list[_ArrayComplex]: ...
@overload
def J(
    atoms: Atoms,
    W: _ArrayRealOrComplex,
    ik: int = ...,
    full: bool = ...,
    norm: Literal["backward", "forward"] = ...,
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
@handle_spin
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
