# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

def KSO(
    scf: SCF,
    write_cubes: bool = ...,
    **kwargs: Any,
) -> list[NDArray[np.complex128]]: ...
def FO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]] | None = ...,
) -> list[NDArray[np.complex128]]: ...
def FLO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: NDArray[np.float64] | Sequence[NDArray[np.float64]] | None = ...,
) -> list[NDArray[np.complex128]]: ...
def WO(
    scf: SCF,
    write_cubes: bool = ...,
    precondition: bool = ...,
) -> list[NDArray[np.complex128]]: ...
def cube_writer(
    atoms: Atoms,
    orb_type: str,
    orbitals: list[NDArray[np.float64]] | list[NDArray[np.complex128]],
) -> None: ...
