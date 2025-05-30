# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

# Add type hints for all properties and methods of the ConfigClass to the module
# This allows IDEs to see that the module has said attribute
backend: str
use_gpu: bool
use_pylibxc: bool
threads: int | None
verbose: int | str
info: Callable[[], None]

class ConfigClass:
    def __init__(self) -> None: ...
    @property
    def backend(self) -> str: ...
    @backend.setter
    def backend(self, value: str) -> None: ...
    @property
    def use_gpu(self) -> bool: ...
    @use_gpu.setter
    def use_gpu(self, value: bool) -> None: ...
    @property
    def use_pylibxc(self) -> bool: ...
    @use_pylibxc.setter
    def use_pylibxc(self, value: bool) -> None: ...
    @property
    def threads(self) -> int | None: ...
    @threads.setter
    def threads(self, value: int) -> None: ...
    @property
    def verbose(self) -> str: ...
    @verbose.setter
    def verbose(self, value: int | str) -> None: ...
    def info(self) -> None: ...
