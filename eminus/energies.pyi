# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, overload

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

@dataclass
class Energy:
    Ekin: float = ...
    Ecoul: float = ...
    Exc: float = ...
    Eloc: float = ...
    Enonloc: float = ...
    Eewald: float = ...
    Esic: float = ...
    Edisp: float = ...
    Eentropy: float = ...
    @property
    def Etot(self) -> float: ...
    def extrapolate(self) -> float: ...

def get_E(scf: SCF) -> float: ...
@overload
def get_Ekin(
    atoms: Atoms,
    Y: NDArray[complex128],
    ik: int,
) -> float: ...
@overload
def get_Ekin(
    atoms: Atoms,
    Y: list[NDArray[complex128]],
) -> float: ...
def get_Ecoul(
    atoms: Atoms,
    n: NDArray[float64],
    phi: NDArray[float64] | None = ...,
) -> float: ...
def get_Exc(
    scf: SCF,
    n: NDArray[float64],
    exc: NDArray[float64] | None = ...,
    n_spin: NDArray[float64] | None = ...,
    dn_spin: NDArray[float64] | None = ...,
    tau: NDArray[float64] | None = ...,
    Nspin: int = ...,
) -> float: ...
def get_Eloc(
    scf: SCF,
    n: NDArray[float64],
) -> float: ...
@overload
def get_Enonloc(
    scf: SCF,
    Y: NDArray[complex128],
    ik: int,
) -> float: ...
@overload
def get_Enonloc(
    scf: SCF,
    Y: list[NDArray[complex128]],
) -> float: ...
def get_Eewald(
    atoms: Atoms,
    gcut: float = ...,
    gamma: float = ...,
) -> float: ...
def get_Esic(
    scf: SCF,
    Y: list[NDArray[complex128]] | None,
    n_single: NDArray[float64] | None = ...,
) -> float: ...
def get_Edisp(
    scf: SCF,
    version: str = ...,
    atm: bool = ...,
    xc: str | None = ...,
) -> float: ...
def get_Eband(
    scf: SCF,
    Y: list[NDArray[complex128]],
    **kwargs: Any,
) -> float: ...
def get_Eentropy(
    scf: SCF,
    epsilon: NDArray[float64],
    Efermi: float,
) -> float: ...
