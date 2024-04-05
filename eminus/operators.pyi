# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .typing import Array1D, Array2D

@overload
def O(
    atoms: Atoms,
    W: NDArray[np.float64],
) -> NDArray[np.float64]: ...
@overload
def O(
    atoms: Atoms,
    W: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
@overload
def O(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]: ...
@overload
def L(
    atoms: Atoms,
    W: NDArray[np.float64],
    ik: int = ...,
) -> NDArray[np.float64]: ...
@overload
def L(
    atoms: Atoms,
    W: NDArray[np.complex128],
    ik: int = ...,
) -> NDArray[np.complex128]: ...
@overload
def Linv(
    atoms: Atoms,
    W: NDArray[np.float64],
) -> NDArray[np.float64]: ...
@overload
def Linv(
    atoms: Atoms,
    W: NDArray[np.complex128],
) -> NDArray[np.complex128]: ...
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
def K(
    atoms: Atoms,
    W: NDArray[np.complex128],
    ik: int,
) -> NDArray[np.complex128]: ...
@overload
def T(
    atoms: Atoms,
    W: NDArray[np.float64],
    dr: Array1D | Array2D,
) -> NDArray[np.float64]: ...
@overload
def T(
    atoms: Atoms,
    W: NDArray[np.complex128],
    dr: Array1D | Array2D,
) -> NDArray[np.complex128]: ...
@overload
def T(
    atoms: Atoms,
    W: list[NDArray[np.complex128]],
    dr: Array1D | Array2D,
) -> list[NDArray[np.complex128]]: ...
