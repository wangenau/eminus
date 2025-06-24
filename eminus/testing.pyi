# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import complexfloating
from numpy.typing import NDArray

_Complex: TypeAlias = complexfloating[Any]
_ArrayComplex: TypeAlias = NDArray[_Complex]

def assert_allclose(
    actual: _ArrayComplex,
    desired: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> None: ...
def assert_array_equal(
    actual: _ArrayComplex,
    desired: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> None: ...
