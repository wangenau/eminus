# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

type _Float = floating[Any]
type _ArrayReal = NDArray[_Float]

def read_gth(
    atom: str,
    charge: int | None = ...,
    psp_path: str = ...,
) -> dict[str, float | _ArrayReal]: ...
def mock_gth() -> dict[str, float | _ArrayReal]: ...
