# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, overload, TypeVar

from numpy import bool_, complexfloating, floating, integer
from numpy.typing import NDArray

from . import operators
from .kpoints import KPoints
from .occupations import Occupations
from .utils import BaseObject

type _Int = integer[Any]
type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayComplex = NDArray[_Complex]
type _ArrayReal = NDArray[_Float]
type _ArrayInt = NDArray[_Int]
type _Array1D = Sequence[float] | _ArrayReal
type _Array2D = Sequence[_Array1D] | _ArrayReal
type _Array3D = Sequence[_Array2D] | _ArrayReal
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
    def I(
        atoms: Atoms,  # noqa: N805
        W: list[_ArrayComplex],
    ) -> list[_ArrayComplex]: ...
    @overload
    def I(
        atoms: Atoms,  # noqa: N805
        W: _ArrayRealOrComplex,
        ik: int = ...,
    ) -> _ArrayRealOrComplex: ...
    @overload
    def J(
        atoms: Atoms,  # noqa: N805
        W: list[_ArrayComplex],
        full: bool = ...,
    ) -> list[_ArrayComplex]: ...
    @overload
    def J(
        atoms: Atoms,  # noqa: N805
        W: _ArrayRealOrComplex,
        ik: int = ...,
        full: bool = ...,
    ) -> _ArrayRealOrComplex: ...
    @overload
    def Idag(
        atoms: Atoms,  # noqa: N805
        W: list[_ArrayComplex],
        full: bool = ...,
    ) -> list[_ArrayComplex]: ...
    @overload
    def Idag(
        atoms: Atoms,  # noqa: N805
        W: _ArrayRealOrComplex,
        ik: int = ...,
        full: bool = ...,
    ) -> _ArrayRealOrComplex: ...
    @overload
    def Jdag(
        atoms: Atoms,  # noqa: N805
        W: list[_ArrayComplex],
    ) -> list[_ArrayComplex]: ...
    @overload
    def Jdag(
        atoms: Atoms,  # noqa: N805
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
    def r(self) -> _ArrayReal: ...
    @property
    def active(self) -> list[NDArray[bool_]]: ...
    @property
    def G(self) -> _ArrayReal: ...
    @property
    def G2(self) -> _ArrayReal: ...
    @property
    def G2c(self) -> _ArrayReal: ...
    @property
    def Gk2(self) -> _ArrayReal: ...
    @property
    def Gk2c(self) -> list[_ArrayReal]: ...
    @property
    def Sf(self) -> NDArray[_Complex]: ...
    @property
    def dV(self) -> float: ...
    def build(self) -> Atoms: ...
    kernel = build
    def recenter(self, center: float | _Array1D | None = ...) -> Atoms: ...
    def set_k(
        self,
        k: _Array2D,
        wk: _Array1D | None = ...,
    ) -> Atoms: ...
    def clear(self) -> Atoms: ...
