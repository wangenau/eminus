# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any, Protocol

from numpy import complexfloating, floating
from numpy.typing import NDArray

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
        W: list[_ComplexReal],
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

def scf_step_occ(
    scf: SCF,
    W: list[_ComplexReal],
) -> float: ...
def scf_step_unocc(
    scf: SCF,
    Z: list[_ComplexReal],
) -> float: ...
def get_grad_occ(
    scf: SCF,
    ik: int,
    spin: int,
    W: list[_ComplexReal],
    **kwargs: Any,
) -> _ComplexReal: ...
def get_grad_unocc(
    scf: SCF,
    ik: int,
    spin: int,
    Z: list[_ComplexReal],
    **kwargs: Any,
) -> _ComplexReal: ...
def sd(
    scf: SCF,
    W: list[_ComplexReal],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> tuple[list[float], list[_ComplexReal]]: ...
def pclm(
    scf: SCF,
    W: list[_ComplexReal],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    precondition: bool = ...,
    **kwargs: Any,
) -> tuple[list[float], list[_ComplexReal]]: ...
def lm(
    scf: SCF,
    W: list[_ComplexReal],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    **kwargs: Any,
) -> tuple[list[float], list[_ComplexReal]]: ...
def pccg(
    scf: SCF,
    W: list[_ComplexReal],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
    precondition: bool = ...,
) -> tuple[list[float], list[_ComplexReal]]: ...
def cg(
    scf: SCF,
    W: list[_ComplexReal],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> tuple[list[float], list[_ComplexReal]]: ...
def auto(
    scf: SCF,
    W: list[_ComplexReal],
    Nit: int,
    cost: _CostType = ...,
    grad: _GradType = ...,
    condition: _Conditionype = ...,
    betat: float = ...,
    cgform: int = ...,
) -> tuple[list[float], list[_ComplexReal]]: ...

IMPLEMENTED: dict[str, Callable[[Any], tuple[list[float], _ComplexReal]]]
