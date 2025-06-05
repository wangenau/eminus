# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Protocol

class Backend:
    def __getattr__(self, name: str) -> Any: ...
    @staticmethod
    def debug(func: Any) -> Any: ...
    @staticmethod
    def convert(value: Any) -> Any: ...

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
