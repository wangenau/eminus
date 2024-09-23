# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload, TypeVar

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .occupations import Occupations
from .scf import SCF
from .typing import Array1D

_AnyFloat = TypeVar('_AnyFloat', float, float64, NDArray[float64])

def cutoff2gridspacing(E: float) -> float: ...
def gridspacing2cutoff(h: float) -> float: ...
def center_of_mass(
    coords: NDArray[float64],
    masses: Array1D | None = ...,
) -> NDArray[float64]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
def inertia_tensor(
    coords: NDArray[float64],
    masses: Array1D | None = ...,
) -> NDArray[float64]: ...
def get_dipole(
    scf: SCF,
    n: NDArray[float64] | None = ...,
) -> NDArray[float64]: ...
def get_ip(scf: SCF) -> float: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: list[NDArray[complex128]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: NDArray[complex128],
    eps: float = ...,
) -> bool: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: list[NDArray[complex128]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: NDArray[complex128],
    eps: float = ...,
) -> bool: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: list[NDArray[complex128]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: NDArray[complex128],
    eps: float = ...,
) -> bool: ...
def get_isovalue(
    n: NDArray[float64],
    percent: int = ...,
) -> float: ...
def get_tautf(scf: SCF) -> NDArray[float64]: ...
def get_tauw(scf: SCF) -> NDArray[float64]: ...
def get_elf(scf: SCF) -> NDArray[float64]: ...
def get_reduced_gradient(
    scf: SCF,
    eps: float = ...,
) -> NDArray[float64]: ...
def get_spin_squared(scf: SCF) -> float: ...
def get_multiplicity(scf: SCF) -> float: ...
def get_magnetization(scf: SCF) -> float: ...
def get_bandgap(scf: SCF) -> float: ...
def get_Efermi(
    obj: Occupations | SCF,
    epsilon: NDArray[float64] | None = ...,
) -> float: ...
def fermi_distribution(
    E: _AnyFloat,
    mu: float,
    kbT: float,
) -> _AnyFloat: ...
def electronic_entropy(
    E: _AnyFloat,
    mu: float,
    kbT: float,
) -> _AnyFloat: ...
def get_dos(
    epsilon: NDArray[float64],
    wk: NDArray[float64],
    spin: int = ...,
    npts: int = ...,
    width: float = ...,
) -> tuple[NDArray[float64], NDArray[float64]]: ...
