# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, overload, TypeVar

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

_AnyW = TypeVar("_AnyW", NDArray[complexfloating], list[NDArray[complexfloating]])

def get_phi(
    atoms: Atoms,
    n: NDArray[floating],
) -> NDArray[floating]: ...
def orth(
    atoms: Atoms,
    W: _AnyW,
) -> _AnyW: ...
def orth_unocc(
    atoms: Atoms,
    Y: _AnyW,
    Z: _AnyW,
) -> _AnyW: ...
def get_n_total(
    atoms: Atoms,
    Y: list[NDArray[complexfloating]],
    n_spin: NDArray[floating] | None = ...,
) -> NDArray[floating]: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: list[NDArray[complexfloating]],
) -> NDArray[floating]: ...
@overload
def get_n_spin(
    atoms: Atoms,
    Y: NDArray[complexfloating],
    ik: int,
) -> NDArray[floating]: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: list[NDArray[complexfloating]],
) -> NDArray[floating]: ...
@overload
def get_n_single(
    atoms: Atoms,
    Y: NDArray[complexfloating],
    ik: int,
) -> NDArray[floating]: ...
def get_grad(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[complexfloating]],
    **kwargs: Any,
) -> NDArray[complexfloating]: ...
def H(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[NDArray[complexfloating]],
    dn_spin: NDArray[floating] | None = ...,
    phi: NDArray[floating] | None = ...,
    vxc: NDArray[complexfloating] | None = ...,
    vsigma: NDArray[complexfloating] | None = ...,
    vtau: NDArray[complexfloating] | None = ...,
) -> NDArray[complexfloating]: ...
def H_precompute(
    scf: SCF,
    W: list[NDArray[complexfloating]],
) -> tuple[
    NDArray[floating],
    NDArray[complexfloating],
    NDArray[complexfloating],
    NDArray[complexfloating],
    NDArray[complexfloating],
]: ...
def Q(
    inp: NDArray[complexfloating],
    U: NDArray[complexfloating],
) -> NDArray[complexfloating]: ...
def get_psi(
    scf: SCF,
    W: list[NDArray[complexfloating]] | None,
    **kwargs: Any,
) -> list[NDArray[complexfloating]]: ...
def get_epsilon(
    scf: SCF,
    W: list[NDArray[complexfloating]] | None,
    **kwargs: Any,
) -> NDArray[floating]: ...
def get_epsilon_unocc(
    scf: SCF,
    W: list[NDArray[complexfloating]] | None,
    Z: list[NDArray[complexfloating]] | None,
    **kwargs: Any,
) -> NDArray[floating]: ...
def guess_random(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[NDArray[complexfloating]]: ...
def guess_pseudo(
    scf: SCF,
    Nstate: int | None = ...,
    seed: int = ...,
    symmetric: bool = ...,
) -> list[NDArray[complexfloating]]: ...
