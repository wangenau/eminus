# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, Protocol

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

# Create a custom Callable type for cost functions
class _CostType(Protocol):
    def __call__(
        self,
        scf: SCF,
        step: int,
    ) -> float: ...

# Create a custom Callable type for gradient functions
class _GradType(Protocol):
    def __call__(
        self,
        scf: SCF,
        ik: int,
        spin: int,
        W: list[NDArray[complexfloating]],
        **kwargs: Any,
    ) -> NDArray[complexfloating]: ...

# Create a custom Callable type for condition functions
class _Conditionype(Protocol):
    def __call__(
        self,
        scf: SCF,
        method: str,
        Elist: list[float],
        linmin: NDArray[floating] | None = ...,
        cg: NDArray[floating] | None = ...,
        norm_g: NDArray[floating] | None = ...,
    ) -> bool: ...

def scf_step(
    scf: SCF,
    step: int,
) -> float: ...
def check_convergence(
    scf: SCF,
    method: str,
    Elist: list[float],
    linmin: NDArray[floating] | None = ...,
    cg: NDArray[floating] | None = ...,
    norm_g: NDArray[floating] | None = ...,
) -> bool: ...
def print_scf_step(
    scf: SCF,
    method: str,
    Elist: list[float],
    linmin: NDArray[floating] | None,
    cg: NDArray[floating] | None,
    norm_g: NDArray[floating] | None,
) -> None: ...
def linmin_test(
    g: NDArray[complexfloating],
    d: NDArray[complexfloating],
) -> float: ...
def cg_test(
    atoms: Atoms,
    ik: int,
    g: NDArray[complexfloating],
    g_old: NDArray[complexfloating],
    precondition: bool = ...,
) -> float: ...
def cg_method(
    scf: SCF,
    ik: int,
    cgform: int,
    g: NDArray[complexfloating],
    g_old: NDArray[complexfloating],
    d_old: NDArray[complexfloating],
    precondition: bool = ...,
) -> tuple[float, float]: ...
def sd(
    scf: SCF,
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> list[float]: ...
def pclm(
    scf: SCF,
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    precondition: bool = ...,
    **kwargs: Any,
) -> list[float]: ...
def lm(
    scf: SCF,
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> list[float]: ...
def pccg(
    scf: SCF,
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
    precondition: bool = ...,
) -> list[float]: ...
def cg(
    scf: SCF,
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> list[float]: ...
def auto(
    scf: SCF,
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> list[float]: ...

IMPLEMENTED: dict[str, Callable[[Any], list[float]]]
