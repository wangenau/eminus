# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

def KSO(
    scf: SCF,
    write_cubes: bool = ...,
    **kwargs: Any,
) -> list[NDArray[complex128]]: ...
def FO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: NDArray[float64] | Sequence[NDArray[float64]] | None = ...,
    guess: str = ...,
) -> list[NDArray[complex128]]: ...
def FLO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: NDArray[float64] | Sequence[NDArray[float64]] | None = ...,
    guess: str = ...,
) -> list[NDArray[complex128]]: ...
def WO(
    scf: SCF,
    write_cubes: bool = ...,
    precondition: bool = ...,
) -> list[NDArray[complex128]]: ...
def SCDM(
    scf: SCF,
    write_cubes: bool = ...,
) -> list[NDArray[complex128]]: ...
def cube_writer(
    atoms: Atoms,
    orb_type: str,
    orbitals: list[NDArray[float64]] | list[NDArray[complex128]],
) -> None: ...
