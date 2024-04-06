# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from numpy import float64
from numpy.typing import NDArray

def read_gth(
    atom: str,
    charge: int | None = ...,
    psp_path: str = ...,
) -> dict[str, int | float | NDArray[float64]]: ...
def mock_gth() -> dict[str, int | float | NDArray[float64]]: ...
