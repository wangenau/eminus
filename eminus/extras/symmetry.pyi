# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from ..atoms import Atoms

def symmetrize(
    atoms: Atoms,
    space_group: bool = ...,
    time_reversal: bool = ...,
) -> None: ...
