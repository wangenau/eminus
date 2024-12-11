# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

from numpy import floating
from numpy.typing import NDArray

def lda_c_chachiyo(
    n: NDArray[floating],
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
def chachiyo_scaling(
    zeta: NDArray[floating],
) -> tuple[NDArray[floating], NDArray[floating]]: ...
def lda_c_chachiyo_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    weight_function: Callable[
        [NDArray[floating]], tuple[NDArray[floating], NDArray[floating]]
    ] = ...,
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
