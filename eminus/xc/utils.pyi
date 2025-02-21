# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from numpy import floating
from numpy.typing import NDArray

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]
_DnOrNone = TypeVar("_DnOrNone", _ArrayReal, None)
_TauOrNone = TypeVar("_TauOrNone", _ArrayReal, None)

# Create a custom Callable type for functionals
class _FunctionalType(Protocol):
    def __call__(
        self,
        n: _ArrayReal,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[_ArrayReal, _ArrayReal, None] | tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...

def get_xc(
    xc: str | Sequence[str],
    n_spin: _ArrayReal,
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, float] | None = ...,
    dens_threshold: float = ...,
) -> tuple[_ArrayReal, _ArrayReal, _DnOrNone, _TauOrNone]: ...
def get_exc(
    xc: str | Sequence[str],
    n_spin: _ArrayReal,
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, float] | None = ...,
    dens_threshold: float = ...,
) -> _ArrayReal: ...
def get_vxc(
    xc: str | Sequence[str],
    n_spin: _ArrayReal,
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, float] | None = ...,
    dens_threshold: float = ...,
) -> tuple[_ArrayReal, _DnOrNone, _TauOrNone]: ...
def parse_functionals(xc: str) -> list[str]: ...
def parse_xc_type(xc: str) -> str: ...
def parse_xc_libxc(xc_id: int | str) -> str: ...
def parse_xc_pyscf(xc_id: int | str) -> str: ...
def get_xc_defaults(xc: str | Sequence[str]) -> dict[str, float]: ...
def get_zeta(n_spin: _ArrayReal) -> _ArrayReal: ...
def mock_xc(
    n: _ArrayReal,
    Nspin: int = ...,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, None]: ...

IMPLEMENTED: dict[str, _FunctionalType]
XC_MAP: dict[str, str]
ALIAS: dict[str, str]
