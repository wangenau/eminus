# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

from numpy import float64
from numpy.typing import NDArray

def lda_c_chachiyo(
    n: NDArray[float64],
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], None]: ...
def chachiyo_scaling(
    zeta: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]: ...
def lda_c_chachiyo_spin(
    n: NDArray[float64],
    zeta: NDArray[float64],
    weight_function: Callable[[NDArray[float64]], tuple[NDArray[float64], NDArray[float64]]] = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], None]: ...
