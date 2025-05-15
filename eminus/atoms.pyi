# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, overload, Self, TypeAlias, TypeVar

from numpy import bool_, complexfloating, floating, integer
from numpy.typing import NDArray

from . import operators
from .kpoints import KPoints
from .occupations import Occupations
from .utils import BaseObject

_Int: TypeAlias = integer[Any]
_Float: TypeAlias = floating[Any]
_Complex: TypeAlias = complexfloating[Any]
_ArrayComplex: TypeAlias = NDArray[_Complex]
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayInt: TypeAlias = NDArray[_Int]
_Array1D: TypeAlias = Sequence[float] | _ArrayReal
_Array2D: TypeAlias = Sequence[_Array1D] | _ArrayReal
_Array3D: TypeAlias = Sequence[_Array2D] | _ArrayReal
_ArrayRealOrComplex = TypeVar("_ArrayRealOrComplex", _ArrayReal, _ArrayComplex)

class Atoms(BaseObject):
    occ: Occupations
    kpts: KPoints
    is_built: bool
    O = operators.O
    L = operators.L
    Linv = operators.Linv
    K = operators.K
    T = operators.T
    @overload
    def I(self, W: list[_ArrayComplex]) -> list[_ArrayComplex]: ...
    @overload
    def I(
        self,
        W: _ArrayRealOrComplex,
        ik: int = ...,
    ) -> _ArrayRealOrComplex: ...
    @overload
    def J(
        self,
        W: list[_ArrayComplex],
        full: bool = ...,
    ) -> list[_ArrayComplex]: ...
    @overload
    def J(
        self,
        W: _ArrayRealOrComplex,
        ik: int = ...,
        full: bool = ...,
    ) -> _ArrayRealOrComplex: ...
    @overload
    def Idag(
        self,
        W: list[_ArrayComplex],
        full: bool = ...,
    ) -> list[_ArrayComplex]: ...
    @overload
    def Idag(
        self,
        W: _ArrayRealOrComplex,
        ik: int = ...,
        full: bool = ...,
    ) -> _ArrayRealOrComplex: ...
    @overload
    def Jdag(self, W: list[_ArrayComplex]) -> list[_ArrayComplex]: ...
    @overload
    def Jdag(
        self,
        W: _ArrayRealOrComplex,
        ik: int = ...,
    ) -> _ArrayRealOrComplex: ...
    def __init__(
        self,
        atom: str | Sequence[str],
        pos: _Array1D | _Array2D,
        ecut: float | None = ...,
        a: float | _Array1D | _Array2D | None = ...,
        spin: int | None = ...,
        charge: int = ...,
        unrestricted: bool | None = ...,
        center: bool | None = ...,
        verbose: int | str | None = ...,
    ) -> None: ...
    @property
    def atom(self) -> list[str]: ...
    @atom.setter
    def atom(self, value: str | Sequence[str]) -> None: ...
    @property
    def pos(self) -> _ArrayReal: ...
    @pos.setter
    def pos(self, value: _Array1D | _Array2D) -> None: ...
    @property
    def ecut(self) -> float: ...
    @ecut.setter
    def ecut(self, value: float | None) -> None: ...
    @property
    def a(self) -> _ArrayReal: ...
    @a.setter
    def a(self, value: float | _Array1D | _Array2D | None) -> None: ...
    @property
    def spin(self) -> int: ...
    @spin.setter
    def spin(self, value: int | None) -> None: ...
    @property
    def charge(self) -> int: ...
    @charge.setter
    def charge(self, value: int) -> None: ...
    @property
    def unrestricted(self) -> bool: ...
    @unrestricted.setter
    def unrestricted(self, value: bool | None) -> None: ...
    @property
    def center(self) -> bool | str: ...
    @center.setter
    def center(self, value: bool | None) -> None: ...
    @property
    def verbose(self) -> str: ...
    @verbose.setter
    def verbose(self, value: int | str | None) -> None: ...
    @property
    def f(self) -> _ArrayReal: ...
    @f.setter
    def f(self, value: float | _Array1D | _Array2D | _Array3D) -> None: ...
    @property
    def s(self) -> _ArrayInt: ...
    @s.setter
    def s(self, value: int | Sequence[int] | _ArrayInt) -> None: ...
    @property
    def Z(self) -> _ArrayInt: ...
    @Z.setter
    def Z(self, value: int | Sequence[int] | _ArrayInt | str | dict[str, int] | None) -> None: ...
    @property
    def Natoms(self) -> int: ...
    @property
    def Ns(self) -> _ArrayInt: ...
    @property
    def Omega(self) -> float: ...
    @property
    def r(self) -> _ArrayReal | None: ...
    @property
    def active(self) -> list[NDArray[bool_]] | None: ...
    @property
    def G(self) -> _ArrayReal | None: ...
    @property
    def G2(self) -> _ArrayReal | None: ...
    @property
    def G2c(self) -> _ArrayReal | None: ...
    @property
    def Gk2(self) -> _ArrayReal | None: ...
    @property
    def Gk2c(self) -> list[_ArrayReal] | None: ...
    @property
    def Sf(self) -> NDArray[_Complex] | None: ...
    @property
    def dV(self) -> float: ...
    def build(self) -> Self: ...
    kernel = build
    def recenter(self, center: float | _Array1D | None = ...) -> Self: ...
    def set_k(
        self,
        k: _Array2D,
        wk: _Array1D | None = ...,
    ) -> Self: ...
    def clear(self) -> Self: ...
