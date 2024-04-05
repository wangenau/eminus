# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Any

def read_json(filename: str) -> Any: ...
def write_json(
    obj: Any,
    filename: str,
) -> None: ...
