# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TypeAlias

from numpy import complexfloating
from numpy.typing import NDArray

_Complex: TypeAlias = complexfloating[Any]
_ArrayComplex: TypeAlias = NDArray[_Complex]

class Backend:
    def __getattr__(self, name: str) -> Any: ...
    @staticmethod
    def is_array(value: Any) -> Any: ...
    def sqrtm(self, A: _ArrayComplex) -> _ArrayComplex: ...
    def expm(
        self,
        A: _ArrayComplex,
        *args: Any,
        **kwargs: Any,
    ) -> _ArrayComplex: ...

def is_array(value: Any) -> Any: ...
def sqrtm(A: _ArrayComplex) -> _ArrayComplex: ...
def expm(
    A: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> _ArrayComplex: ...
