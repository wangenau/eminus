# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating
_ArrayReal: TypeAlias = NDArray[_Float]

def read_gth(
    atom: str,
    charge: int | None = ...,
    psp_path: str = ...,
) -> dict[str, float | _ArrayReal]: ...
def mock_gth() -> dict[str, float | _ArrayReal]: ...
