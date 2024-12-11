# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

def gga_x_pbe(
    n: NDArray[floating],
    mu: float = ...,
    dn_spin: NDArray[floating] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
def gga_x_pbe_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    mu: float = ...,
    dn_spin: NDArray[floating] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
def pbe_x_base(
    n: NDArray[floating],
    mu: float = ...,
    dn: NDArray[floating] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...
