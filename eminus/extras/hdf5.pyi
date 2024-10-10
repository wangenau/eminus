# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

def read_hdf5(filename: str) -> Any: ...
def write_hdf5(
    obj: Any,
    filename: str,
) -> None: ...
