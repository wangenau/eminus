# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ..atoms import Atoms

@overload
def I(
    atoms: Atoms,
    W: NDArray[np.float64],
    ik: int = ...,
) -> NDArray[np.float64]: ...
@overload
def I(
    atoms: Atoms,
    W: NDArray[np.complex128],
    ik: int = ...,
) -> NDArray[np.complex128]: ...
@overload
def I(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]: ...
@overload
def J(
    atoms: Atoms,
    W: NDArray[np.float64],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[np.float64]: ...
@overload
def J(
    atoms: Atoms,
    W: NDArray[np.complex128],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[np.complex128]: ...
@overload
def J(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
    full: bool = ...,
) -> list[NDArray[np.complex128]]: ...
@overload
def Idag(
    atoms: Atoms,
    W: NDArray[np.float64],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[np.float64]: ...
@overload
def Idag(
    atoms: Atoms,
    W: NDArray[np.complex128],
    ik: int = ...,
    full: bool = ...,
) -> NDArray[np.complex128]: ...
@overload
def Idag(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
    full: bool = ...,
) -> list[NDArray[np.complex128]]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: NDArray[np.float64],
    ik: int = ...,
) -> NDArray[np.float64]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: NDArray[np.complex128],
    ik: int = ...,
) -> NDArray[np.complex128]: ...
@overload
def Jdag(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]: ...
