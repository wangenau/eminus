# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Protocol, TypeAlias

from numpy import complexfloating
from numpy.typing import NDArray

_Complex: TypeAlias = complexfloating[Any]
_ArrayComplex: TypeAlias = NDArray[_Complex]

class Backend:
    def __getattr__(self, name: str) -> Any: ...
    @staticmethod
    def debug(func: Any) -> Any: ...
    @staticmethod
    def convert(value: Any) -> Any: ...
    @staticmethod
    def is_array(value: Any) -> Any: ...
    def sqrtm(self, A: _ArrayComplex) -> _ArrayComplex: ...
    def expm(
        self,
        A: _ArrayComplex,
        *args: Any,
        **kwargs: Any,
    ) -> _ArrayComplex: ...

# Create a custom Callable type for some decorators
class _HandleType(Protocol):
    def __call__(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

def debug(func: _HandleType) -> _HandleType: ...
def convert(value: Any) -> Any: ...
def is_array(value: Any) -> Any: ...
def sqrtm(A: _ArrayComplex) -> _ArrayComplex: ...
def expm(
    A: _ArrayComplex,
    *args: Any,
    **kwargs: Any,
) -> _ArrayComplex: ...
