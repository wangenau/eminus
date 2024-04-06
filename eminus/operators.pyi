# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

@overload
def O(
    atoms: Atoms,
    W: NDArray[float64],
) -> NDArray[float64]: ...
@overload
def O(
    atoms: Atoms,
    W: NDArray[complex128],
) -> NDArray[complex128]: ...
@overload
def O(
    atoms: Atoms,
    W: list[NDArray[complex128]],
) -> list[NDArray[complex128]]: ...
@overload
def L(
    atoms: Atoms,
    W: NDArray[float64],
    ik: int = ...,
) -> NDArray[float64]: ...
@overload
def L(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int = ...,
) -> NDArray[complex128]: ...
@overload
def Linv(
    atoms: Atoms,
    W: NDArray[float64],
) -> NDArray[float64]: ...
@overload
def Linv(
    atoms: Atoms,
    W: NDArray[complex128],
) -> NDArray[complex128]: ...
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
def K(
    atoms: Atoms,
    W: NDArray[complex128],
    ik: int,
) -> NDArray[complex128]: ...
@overload
def T(
    atoms: Atoms,
    W: NDArray[float64],
    dr: Array1D | Array2D,
) -> NDArray[float64]: ...
@overload
def T(
    atoms: Atoms,
    W: NDArray[complex128],
    dr: Array1D | Array2D,
) -> NDArray[complex128]: ...
@overload
def T(
    atoms: Atoms,
    W: list[NDArray[complex128]],
    dr: Array1D | Array2D,
) -> list[NDArray[complex128]]: ...
