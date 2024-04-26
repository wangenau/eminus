# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload

from numpy import complex128, float64
from numpy.typing import NDArray

from .atoms import Atoms
from .occupations import Occupations
from .scf import SCF
from .typing import Array1D

def cutoff2gridspacing(E: float) -> float: ...
def gridspacing2cutoff(h: float) -> float: ...
def center_of_mass(
    coords: NDArray[float64],
    masses: Array1D | None = ...,
) -> NDArray[float64]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: NDArray[complex128],
) -> NDArray[float64]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: list[NDArray[complex128]],
) -> list[NDArray[float64]]: ...
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
    func: NDArray[complex128],
    eps: float = ...,
) -> bool: ...
@overload
def check_ortho(
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
def check_norm(
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
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: list[NDArray[complex128]],
    eps: float = ...,
) -> list[bool]: ...
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
@overload
def fermi_distribution(
    E: float,
    mu: float,
    kbT: float,
) -> float: ...
@overload
def fermi_distribution(
    E: NDArray[float64],
    mu: float,
    kbT: float,
) -> NDArray[float64]: ...
@overload
def electronic_entropy(
    E: float,
    mu: float,
    kbT: float,
) -> float: ...
@overload
def electronic_entropy(
    E: NDArray[float64],
    mu: float,
    kbT: float,
) -> NDArray[float64]: ...
