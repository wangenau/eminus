# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, Callable, Literal, TypeAlias, TypeVar

from numpy import complexfloating, floating, integer
from numpy.typing import NDArray

_Int: TypeAlias = integer
_Float: TypeAlias = floating
_Complex: TypeAlias = complexfloating
_ArrayReal: TypeAlias = NDArray[_Float]
_ArrayComplex: TypeAlias = NDArray[_Complex]
_Array2D: TypeAlias = Sequence[Sequence[float]] | Sequence[_ArrayReal] | _ArrayReal

_F = TypeVar("_F", bound=Callable[..., object])

class BaseObject:
    def view(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    def write(
        self,
        filename: str,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

def dotprod(
    a: _ArrayComplex,
    b: _ArrayComplex,
) -> float: ...
def Ylm_real(
    l: int,
    m: int,
    G: _ArrayReal,
) -> _ArrayReal: ...
def handle_spin(
    func: _F,
) -> _F: ...
def handle_k(
    func: Callable[..., object] | None = ...,
    *,
    mode: Literal["gracefully", "index", "reduce", "skip"] = ...,
) -> Callable[..., Any]: ...
def pseudo_uniform(
    size: Sequence[int] | NDArray[_Int],
    seed: int = ...,
) -> _ArrayComplex: ...
def add_maybe_none(
    a: _ArrayComplex | None,
    b: _ArrayComplex | None,
) -> _ArrayComplex: ...
def molecule2list(molecule: str) -> list[str]: ...
def atom2charge(
    atom: Sequence[str],
    path: str | None = ...,
) -> list[int]: ...
def vector_angle(
    a: _ArrayReal,
    b: _ArrayReal,
) -> _ArrayReal: ...
def get_lattice(lattice_vectors: _Array2D) -> list[_ArrayReal]: ...
