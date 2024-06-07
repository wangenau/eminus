# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import complex128, float64
from numpy.typing import NDArray

from ..atoms import Atoms

@overload
def I(
    atoms: Atoms,
    W: NDArray[float64],
    ik: int = ...,
) -> NDArray[float64]: ...
@overload
def I(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int = ...,
) -> NDArray[complex128]: ...
@overload
def I(
    atoms: Atoms,
    W: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
@overload
def J(
    atoms: Atoms,
    W: NDArray[float64],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[float64]: ...
@overload
def J(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[complex128]: ...
@overload
def J(
    atoms: Atoms,
    W: list[NDArray[complex128]],
    full: bool = ...,
) -> list[NDArray[complex128]]: ...
@overload
def Idag(
    atoms: Atoms,
    W: NDArray[float64],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[float64]: ...
@overload
def Idag(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[complex128]: ...
@overload
def Idag(
    atoms: Atoms,
    W: list[NDArray[complex128]],
    full: bool = ...,
) -> list[NDArray[complex128]]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: NDArray[float64],
    ik: int = ...,
) -> NDArray[float64]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int = ...,
) -> NDArray[complex128]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
