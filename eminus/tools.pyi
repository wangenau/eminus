# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import overload

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .occupations import Occupations
from .scf import SCF
from .typing import Array1D

def cutoff2gridspacing(E: float) -> float: ...
def gridspacing2cutoff(h: float) -> float: ...
def center_of_mass(
    coords: NDArray[np.float64],
    masses: Array1D | None = ...,
) -> NDArray[np.float64]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: NDArray[np.complex128],
) -> NDArray[np.float64]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: list[NDArray[np.complex128]],
) -> list[NDArray[np.float64]]: ...
def inertia_tensor(
    coords: NDArray[np.float64],
    masses: Array1D | None = ...,
) -> NDArray[np.float64]: ...
def get_dipole(
    scf: SCF,
    n: NDArray[np.float64] | None = ...,
) -> NDArray[np.float64]: ...
def get_ip(scf: SCF) -> float: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: NDArray[np.complex128],
    eps: float = ...,
) -> bool: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: list[NDArray[np.complex128]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: NDArray[np.complex128],
    eps: float = ...,
) -> bool: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: list[NDArray[np.complex128]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: NDArray[np.complex128],
    eps: float = ...,
) -> bool: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: list[NDArray[np.complex128]],
    eps: float = ...,
) -> list[bool]: ...
def get_isovalue(
    n: NDArray[np.float64],
    percent: int = ...,
) -> float: ...
def get_tautf(scf: SCF) -> NDArray[np.float64]: ...
def get_tauw(scf: SCF) -> NDArray[np.float64]: ...
def get_elf(scf: SCF) -> NDArray[np.float64]: ...
def get_reduced_gradient(
    scf: SCF,
    eps: float = ...,
) -> NDArray[np.float64]: ...
def get_spin_squared(scf: SCF) -> float: ...
def get_multiplicity(scf: SCF) -> float: ...
def get_bandgap(scf: SCF) -> float: ...
def get_Efermi(
    obj: Occupations | SCF,
    epsilon: NDArray[np.float64] | None = ...,
) -> float: ...
@overload
def fermi_distribution(
    E: float,
    mu: float,
    kbT: float,
) -> float: ...
@overload
def fermi_distribution(
    E: NDArray[np.float64],
    mu: float,
    kbT: float,
) -> NDArray[np.float64]: ...
@overload
def electronic_entropy(
    E: float,
    mu: float,
    kbT: float,
) -> float: ...
@overload
def electronic_entropy(
    E: NDArray[np.float64],
    mu: float,
    kbT: float,
) -> NDArray[np.float64]: ...
