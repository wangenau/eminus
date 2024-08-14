# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import float64
from numpy.typing import NDArray

from ..xc.utils import _DnOrNone, _TauOrNone

def libxc_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, Any] | None = ...,
) -> tuple[NDArray[float64], NDArray[float64], _DnOrNone, _TauOrNone]: ...
def pyscf_functional(
    xc: str,
    n_spin: NDArray[float64],
    Nspin: int,
    dn_spin: _DnOrNone = ...,
    tau: _TauOrNone = ...,
    xc_params: dict[str, Any] | None = ...,
) -> tuple[NDArray[float64], NDArray[float64], _DnOrNone, _TauOrNone]: ...
