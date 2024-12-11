# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

def KSO(
    scf: SCF,
    write_cubes: bool = ...,
    **kwargs: Any,
) -> list[NDArray[complexfloating]]: ...
def FO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: NDArray[floating] | Sequence[NDArray[floating]] | None = ...,
    guess: str = ...,
) -> list[NDArray[complexfloating]]: ...
def FLO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: NDArray[floating] | Sequence[NDArray[floating]] | None = ...,
    guess: str = ...,
) -> list[NDArray[complexfloating]]: ...
def WO(
    scf: SCF,
    write_cubes: bool = ...,
    precondition: bool = ...,
) -> list[NDArray[complexfloating]]: ...
def SCDM(
    scf: SCF,
    write_cubes: bool = ...,
) -> list[NDArray[complexfloating]]: ...
def cube_writer(
    atoms: Atoms,
    orb_type: str,
    orbitals: list[NDArray[floating]] | list[NDArray[complexfloating]],
) -> None: ...
