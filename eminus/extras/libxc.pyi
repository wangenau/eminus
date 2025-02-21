# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeVar

from numpy import floating
from numpy.typing import NDArray

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]
_DnOrNone = TypeVar("_DnOrNone", _ArrayReal, None)
_TauOrNone = TypeVar("_TauOrNone", _ArrayReal, None)

def libxc_functional(
    xc: str,
    n_spin: _ArrayReal,
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, float] | None = ...,
) -> tuple[_ArrayReal, _ArrayReal, _DnOrNone, _TauOrNone]: ...
def pyscf_functional(
    xc: str,
    n_spin: _ArrayReal,
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, float] | None = ...,
) -> tuple[_ArrayReal, _ArrayReal, _DnOrNone, _TauOrNone]: ...
