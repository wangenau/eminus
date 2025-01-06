# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from ._typing import _Array1D
from .atoms import Atoms
from .occupations import Occupations
from .scf import SCF

_AnyFloat = TypeVar("_AnyFloat", float, floating, NDArray[floating])

def cutoff2gridspacing(E: float) -> float: ...
def gridspacing2cutoff(h: float) -> float: ...
def center_of_mass(
    coords: NDArray[floating],
    masses: _Array1D | None = ...,
) -> NDArray[floating]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: list[NDArray[complexfloating]],
) -> list[NDArray[floating]]: ...
@overload
def orbital_center(
    obj: Atoms | SCF,
    psirs: NDArray[complexfloating],
) -> NDArray[floating]: ...
def inertia_tensor(
    coords: NDArray[floating],
    masses: _Array1D | None = ...,
) -> NDArray[floating]: ...
def get_dipole(
    scf: SCF,
    n: NDArray[floating] | None = ...,
) -> NDArray[floating]: ...
def get_ip(scf: SCF) -> float: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: list[NDArray[complexfloating]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_ortho(
    obj: Atoms | SCF,
    func: NDArray[complexfloating],
    eps: float = ...,
) -> bool: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: list[NDArray[complexfloating]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_norm(
    obj: Atoms | SCF,
    func: NDArray[complexfloating],
    eps: float = ...,
) -> bool: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: list[NDArray[complexfloating]],
    eps: float = ...,
) -> list[bool]: ...
@overload
def check_orthonorm(
    obj: Atoms | SCF,
    func: NDArray[complexfloating],
    eps: float = ...,
) -> bool: ...
def get_isovalue(
    n: NDArray[floating],
    percent: int = ...,
) -> float: ...
def get_tautf(scf: SCF) -> NDArray[floating]: ...
def get_tauw(scf: SCF) -> NDArray[floating]: ...
def get_elf(scf: SCF) -> NDArray[floating]: ...
def get_reduced_gradient(
    scf: SCF,
    eps: float = ...,
) -> NDArray[floating]: ...
def get_spin_squared(scf: SCF) -> float: ...
def get_multiplicity(scf: SCF) -> float: ...
def get_magnetization(scf: SCF) -> float: ...
def get_bandgap(scf: SCF) -> float: ...
def get_Efermi(
    obj: Occupations | SCF,
    epsilon: NDArray[floating] | None = ...,
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
    epsilon: NDArray[floating],
    wk: NDArray[floating],
    spin: int = ...,
    npts: int = ...,
    width: float = ...,
) -> tuple[NDArray[floating], NDArray[floating]]: ...
