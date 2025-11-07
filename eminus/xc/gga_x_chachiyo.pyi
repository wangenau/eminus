# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

_Float: TypeAlias = floating
_ArrayReal: TypeAlias = NDArray[_Float]

def gga_x_chachiyo(
    n: _ArrayReal,
    dn_spin: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
def gga_x_chachiyo_spin(
    n: _ArrayReal,
    zeta: _ArrayReal,
    dn_spin: _ArrayReal,
    **kwargs: Any,
) -> tuple[_ArrayReal, _ArrayReal, _ArrayReal]: ...
