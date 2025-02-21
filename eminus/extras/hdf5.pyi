# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal

def read_hdf5(filename: str) -> Any: ...
def write_hdf5(
    obj: Any,
    filename: str,
    compression: Literal["gzip", "lzf"] = ...,
    compression_opts: int | None = ...,
) -> None: ...
