# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, Protocol

from numpy import complexfloating, floating
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

type _Float = floating[Any]
type _Complex = complexfloating[Any]
type _ArrayReal = NDArray[_Float]
type _ComplexReal = NDArray[_Complex]

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
        W: list[_ComplexReal],
        **kwargs: Any,
    ) -> _ComplexReal: ...

# Create a custom Callable type for condition functions
class _Conditionype(Protocol):
    def __call__(
        self,
        scf: SCF,
        method: str,
        Elist: list[float],
        linmin: _ArrayReal | None = ...,
        cg: _ArrayReal | None = ...,
        norm_g: _ArrayReal | None = ...,
    ) -> bool: ...

def scf_step(
    scf: SCF,
    step: int,
) -> float: ...
def check_convergence(
    scf: SCF,
    method: str,
    Elist: list[float],
    linmin: _ArrayReal | None = ...,
    cg: _ArrayReal | None = ...,
    norm_g: _ArrayReal | None = ...,
) -> bool: ...
def print_scf_step(
    scf: SCF,
    method: str,
    Elist: list[float],
    linmin: _ArrayReal | None,
    cg: _ArrayReal | None,
    norm_g: _ArrayReal | None,
) -> None: ...
def linmin_test(
    g: _ComplexReal,
    d: _ComplexReal,
) -> float: ...
def cg_test(
    atoms: Atoms,
    ik: int,
    g: _ComplexReal,
    g_old: _ComplexReal,
    precondition: bool = ...,
) -> float: ...
def cg_method(
    scf: SCF,
    ik: int,
    cgform: int,
    g: _ComplexReal,
    g_old: _ComplexReal,
    d_old: _ComplexReal,
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
