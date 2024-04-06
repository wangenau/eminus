# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import float64
from numpy.typing import NDArray

def lda_c_chachiyo_mod(
    n: NDArray[float64],
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], None]: ...
def chachiyo_scaling_mod(
    zeta: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]: ...
def lda_c_chachiyo_mod_spin(
    n: NDArray[float64],
    zeta: NDArray[float64],
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], None]: ...
