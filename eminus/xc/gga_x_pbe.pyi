# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import float64
from numpy.typing import NDArray

def gga_x_pbe(
    n: NDArray[float64],
    mu: float = ...,
    dn_spin: NDArray[float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
def gga_x_pbe_spin(
    n: NDArray[float64],
    zeta: NDArray[float64],
    mu: float = ...,
    dn_spin: NDArray[float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
def pbe_x_base(
    n: NDArray[float64],
    mu: float = ...,
    dn: NDArray[float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
