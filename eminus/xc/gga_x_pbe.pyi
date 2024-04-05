# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
from numpy.typing import NDArray

def gga_x_pbe(
    n: NDArray[np.float64],
    mu: float = ...,
    dn_spin: NDArray[np.float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def gga_x_pbe_spin(
    n: NDArray[np.float64],
    zeta: NDArray[np.float64],
    mu: float = ...,
    dn_spin: NDArray[np.float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def pbe_x_base(
    n: NDArray[np.float64],
    mu: float = ...,
    dn: NDArray[np.float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
