# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

def lda_c_chachiyo(
    n: NDArray[np.float64],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
def chachiyo_scaling(
    zeta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def lda_c_chachiyo_spin(
    n: NDArray[np.float64],
    zeta: NDArray[np.float64],
    weight_function: Callable[
        [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = ...,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], None]: ...
