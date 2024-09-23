# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import complex128
from numpy.typing import NDArray

from ..atoms import Atoms
from ..operators import _ArrRealorComplex

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
