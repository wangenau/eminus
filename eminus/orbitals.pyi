# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, TypeAlias

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

_Float: TypeAlias = floating[Any]
_Complex: TypeAlias = complexfloating[Any]
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayComplex: TypeAlias = NDArray[_Complex]

def KSO(
    scf: SCF,
    write_cubes: bool = ...,
    **kwargs: Any,
) -> list[_ArrayComplex]: ...
def FO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    guess: str = ...,
) -> list[_ArrayComplex]: ...
def FLO(
    scf: SCF,
    write_cubes: bool = ...,
    fods: _ArrayReal | Sequence[_ArrayReal] | None = ...,
    guess: str = ...,
) -> list[_ArrayComplex]: ...
def WO(
    scf: SCF,
    write_cubes: bool = ...,
    precondition: bool = ...,
) -> list[_ArrayComplex]: ...
def SCDM(
    scf: SCF,
    write_cubes: bool = ...,
) -> list[_ArrayComplex]: ...
def cube_writer(
    atoms: Atoms,
    orb_type: str,
    orbitals: Sequence[_ArrayReal] | Sequence[_ArrayComplex],
) -> None: ...
