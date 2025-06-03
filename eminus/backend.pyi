# SPDX-FileCopyrightText: 2025 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

class Backend:
    def __getattr__(self, name: str) -> Any: ...
