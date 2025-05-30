# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self, TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]
_Array1D: TypeAlias = Sequence[float] | _ArrayReal
_Array2D: TypeAlias = Sequence[_Array1D] | _ArrayReal
_Array3D: TypeAlias = Sequence[_Array2D] | _ArrayReal

@dataclass
class Occupations:
    _Nelec: int = ...
    _Nspin: int = ...
    _spin: float = ...
    _charge: int = ...
    _Nstate: int = ...
    _Nempty: int = ...
    _Nk: int = ...
    _bands: int = ...
    _smearing: float = ...
    is_filled: bool = ...
    @property
    def Nelec(self) -> int: ...
    @Nelec.setter
    def Nelec(self, value: int) -> None: ...
    @property
    def Nspin(self) -> int: ...
    @Nspin.setter
    def Nspin(self, value: int | None) -> None: ...
    @property
    def spin(self) -> float: ...
    @spin.setter
    def spin(self, value: float | None) -> None: ...
    @property
    def charge(self) -> int: ...
    @charge.setter
    def charge(self, value: int) -> None: ...
    @property
    def f(self) -> _ArrayReal: ...
    @f.setter
    def f(self, value: float | _Array1D | _Array2D | _Array3D) -> None: ...
    @property
    def Nk(self) -> int: ...
    @Nk.setter
    def Nk(self, value: int) -> None: ...
    @property
    def wk(self) -> _ArrayReal: ...
    @wk.setter
    def wk(self, value: _Array1D) -> None: ...
    @property
    def bands(self) -> int: ...
    @bands.setter
    def bands(self, value: int) -> None: ...
    @property
    def smearing(self) -> float: ...
    @smearing.setter
    def smearing(self, value: float) -> None: ...
    @property
    def magnetization(self) -> float: ...
    @magnetization.setter
    def magnetization(self, value: float | None) -> None: ...
    @property
    def multiplicity(self) -> int: ...
    @property
    def Nstate(self) -> int: ...
    @property
    def Nempty(self) -> int: ...
    @property
    def F(self) -> _ArrayReal: ...
    def fill(
        self,
        f: float | _ArrayReal | None = ...,
        magnetization: float | None = ...,
    ) -> Self: ...
    kernel = fill
    def smear(self, epsilon: _ArrayReal) -> float: ...
